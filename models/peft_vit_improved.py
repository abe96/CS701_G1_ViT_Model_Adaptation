import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
import random
from typing import Optional, Dict, List
from torchvision.transforms import functional as TF


class LoRALayer(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=32, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices (delta path only; no in-place weight edits)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.lora_dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, in_features)
        result = x @ self.lora_A.T          # (B, L, rank)
        result = self.lora_dropout(result)
        result = result @ self.lora_B.T     # (B, L, out_features)
        return result * self.scaling        # (B, L, out_features)


class AdapterLayer(nn.Module):
    def __init__(self, hidden_dim, adapter_dim, dropout=0.1, gate_init=1e-3):
        super().__init__()
        self.adapter_down = nn.Linear(hidden_dim, adapter_dim)
        self.adapter_up = nn.Linear(adapter_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Init so the path starts near-zero but trainable
        nn.init.xavier_uniform_(self.adapter_down.weight)
        nn.init.zeros_(self.adapter_down.bias)
        nn.init.zeros_(self.adapter_up.weight)
        nn.init.zeros_(self.adapter_up.bias)

        # Gate starts small, not zero → gradients flow immediately
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, x):
        h = self.activation(self.adapter_down(x))
        h = self.dropout(h)
        delta = self.adapter_up(h)
        # FIXED: Cast gate to the same type as the input tensor (for autocast)
        return self.gate.to(x.dtype) * delta


class MultiScaleFeatureFusion(nn.Module):
    """
    Pools along the token axis (sequence length), NOT the hidden dimension.
    Produces a fused feature of size [B, H] and projects back to [B, H].
    """
    def __init__(self, hidden_dim, num_scales=3):
        super().__init__()
        self.num_scales = num_scales
        self.hidden_dim = hidden_dim
        # After pooling and averaging over tokens, each scale yields [B, H].
        # Concatenate -> [B, H * num_scales], then project back to [B, H].
        self.fusion = nn.Linear(hidden_dim * num_scales, hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, L, H]
        B, L, H = x.shape
        x_t = x.transpose(1, 2)  # [B, H, L] for 1D pooling along tokens
        outs: List[torch.Tensor] = []
        for i in range(self.num_scales):
            target_L = max(1, L // (2 ** i))   # L, L/2, L/4, ...
            pooled = F.adaptive_avg_pool1d(x_t, output_size=target_L)  # [B, H, target_L]
            # Average over tokens => [B, H]
            outs.append(pooled.mean(dim=2))
        fused = torch.cat(outs, dim=-1)        # [B, H * num_scales]
        return self.dropout(self.fusion(fused))  # [B, H]


class ImprovedViTPEFT(nn.Module):
    def __init__(
        self,
        num_classes,
        peft_method='adaptformer',
        rank=8,
        alpha=32,
        adapter_dim=64,
        dropout=0.1,
        use_multi_scale=True
    ):
        super().__init__()
        self.peft_method = peft_method
        self.num_classes = num_classes
        self.use_multi_scale = use_multi_scale

        # Pretrained ViT backbone without classification head
        self.backbone = timm.create_model(
            'vit_base_patch16_224.augreg2_in21k_ft_in1k',
            pretrained=True,
            num_classes=0
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        num_blocks_to_unfreeze= 4
        for block in self.backbone.blocks[-num_blocks_to_unfreeze:]:
            for param in block.parameters():
                param.requires_grad = True
        
        hidden_dim = self.backbone.num_features

        # PEFT modules
        self.lora_layers = nn.ModuleDict()
        self.adaptformer_layers = nn.ModuleDict()
        self._lora_hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        if peft_method in ['lora', 'both']:
            self._init_lora_layers(rank, alpha, dropout)
            self._attach_lora_forward_hooks()

        if peft_method in ['adaptformer', 'both']:
            self._init_adaptformer_layers(adapter_dim, dropout)

        # Multi-scale fusion (operates on token features if available)
        if self.use_multi_scale:
            self.multi_scale_fusion = MultiScaleFeatureFusion(hidden_dim)
            classifier_input_dim = hidden_dim
        else:
            classifier_input_dim = hidden_dim

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(classifier_input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self._init_classifier()

    # ---------- LoRA (non-mutating via forward hooks) ----------
    def _init_lora_layers(self, rank, alpha, dropout):
        """
        Create LoRA modules for Linear layers inside attention ('qkv' and 'proj').
        """
        for name, module in self.backbone.named_modules():
            if 'blocks' in name and isinstance(module, nn.Linear) and ('qkv' in name or 'proj' in name):
                layer_name = name.replace('.', '_')
                self.lora_layers[layer_name] = LoRALayer(
                    module.in_features, module.out_features,
                    rank=rank, alpha=alpha, dropout=dropout
                )

    def _lora_forward_hook(self, layer_name):
        """
        Returns a hook that adds the LoRA delta to the Linear output.
        Hook signature: (module, input, output) -> modified_output
        """
        def hook(module, inputs, output):
            # inputs is a tuple; for nn.Linear it's (x,)
            x = inputs[0]  # [B, L, in_features] for attention paths in ViT
            delta = self.lora_layers[layer_name](x)  # [B, L, out_features]
            # Ensure shapes match; if module flattens tokens differently, adapt here.
            return output + delta
        return hook

    def _attach_lora_forward_hooks(self):
        # Register hooks on target Linear modules
        for name, module in self.backbone.named_modules():
            if 'blocks' in name and isinstance(module, nn.Linear) and ('qkv' in name or 'proj' in name):
                layer_name = name.replace('.', '_')
                if layer_name in self.lora_layers:
                    handle = module.register_forward_hook(self._lora_forward_hook(layer_name))
                    self._lora_hook_handles.append(handle)

    # ---------- AdaptFormer ----------
 
    def _init_adaptformer_layers(self, adapter_dim, dropout):
        num_blocks = len(self.backbone.blocks)
        for i in range(num_blocks):
            self.adaptformer_layers[f'adaptformer_{i}'] = AdapterLayer(
                self.backbone.num_features, adapter_dim, dropout
            )

    # ---------- Classifier init ----------
    def _init_classifier(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    # ---------- Forward variants ----------
    def _forward_with_adaptformer(self, x):
        """
        Manually threads through the transformer blocks and adds adapter deltas in parallel.
        Returns token features [B, L, H]; CLS taken by caller if needed.
        """
        # Patch embedding + pos embed
        x = self.backbone.patch_embed(x)                         # [B, L, H]
        # Add positional embeddings (skip class token part for patches)
        x = x + self.backbone.pos_embed[:, 1:, :]
        cls_token = self.backbone.cls_token + self.backbone.pos_embed[:, :1, :]
        x = torch.cat([cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = self.backbone.pos_drop(x)

        # Transformer blocks with adapters
        for i, block in enumerate(self.backbone.blocks):
            block_input = x

            # Attention + residual (timm block API)
            x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
            # MLP + residual
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))

            # Adapter delta on the SAME input (parallel path)
            adapter_name = f'adaptformer_{i}'
            if adapter_name in self.adaptformer_layers:
                x = x + self.adaptformer_layers[adapter_name](block_input)

        x = self.backbone.norm(x)                                # [B, L, H]
        return x

    def _forward_backbone_default(self, x):
        """
        Default backbone forward (no manual block threading).
        Returns global feature [B, H] from CLS.
        """
        feats = self.backbone(x)  # [B, H]
        return feats

    def forward(self, x, use_tta=False, num_tta_crops=5):
        if use_tta and not self.training:
            return self._forward_with_tta(x, num_tta_crops)

        # Choose path
        if self.peft_method == 'lora':
            # LoRA is injected via hooks; just use the backbone forward
            features = self._forward_backbone_default(x)         # [B, H]
            tokens = None
        elif self.peft_method == 'adaptformer':
            tokens = self._forward_with_adaptformer(x)           # [B, L, H]
            features = tokens[:, 0]                              # CLS
        elif self.peft_method == 'both':
            # LoRA hooks active + adapters applied (use adapter path to expose tokens)
            tokens = self._forward_with_adaptformer(x)           # [B, L, H]
            features = tokens[:, 0]
        else:
            # Baseline
            features = self._forward_backbone_default(x)         # [B, H]
            tokens = None

        # Optional multi-scale fusion on token features (only when available)
        if self.use_multi_scale and tokens is not None:
            # tokens: [B, L, H] -> fused [B, H]
            fused = self.multi_scale_fusion(tokens)              # [B, H]
            features = fused

        return self.classifier(features)                          # [B, num_classes]

    def _forward_with_tta(self, x, num_crops=None):
        """
        Performs 10-crop TTA (5 crops + 5 flipped crops).
        x is a batch of [B, C, H, W] (e.g., [B, 3, 256, 256])
        """
        # Target crop size
        crop_size = 224
        
        # 1. Get 5 crops (top-left, top-right, bottom-left, bottom-right, center)
        # five_crop returns a tuple of 5 tensors
        crops = TF.five_crop(x, size=(crop_size, crop_size))
        
        # 2. Get 5 flipped crops
        flipped_x = TF.hflip(x)
        flipped_crops = TF.five_crop(flipped_x, size=(crop_size, crop_size))
        
        # 3. Combine all 10 crops
        # We stack them into a new "batch" dimension
        # (B, C, H, W) -> (10, B, C, H_crop, W_crop)
        # Then view as (10*B, C, H_crop, W_crop) to run the model
        all_crops = torch.stack(crops + flipped_crops) # [10, B, C, 224, 224]
        B = x.shape[0]
        all_crops = all_crops.view(10 * B, x.shape[1], crop_size, crop_size)

        # 4. Run all 10 crops through the model
        # We call self.forward(..., use_tta=False) to prevent infinite recursion
        logits = self.forward(all_crops, use_tta=False) # [10*B, Num_Classes]
        
        # 5. Average the results
        # [10*B, Num_Classes] -> [10, B, Num_Classes] -> [B, Num_Classes]
        logits = logits.view(10, B, self.num_classes)
        return logits.mean(dim=0)

    def get_trainable_params(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def create_model(
    num_classes,
    peft_method='adaptformer',
    rank=8,
    alpha=32,
    adapter_dim=64,
    dropout=0.1,
    use_multi_scale=True
):
    return ImprovedViTPEFT(
        num_classes=num_classes,
        peft_method=peft_method,
        rank=rank,
        alpha=alpha,
        adapter_dim=adapter_dim,
        dropout=dropout,
        use_multi_scale=use_multi_scale
    )
