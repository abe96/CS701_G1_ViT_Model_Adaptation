import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms

from data.dataset import create_data_loaders
from models.peft_vit_improved import create_model
from utils.metrics import AverageMeter, accuracy
from utils.scheduler import create_scheduler


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss.
    This handles both "hard" (class index) and "soft" (mixed) targets.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        assert 0.0 <= smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        
        logprobs = F.log_softmax(x, dim=-1)
        
        if isinstance(target, tuple):
            # --- Soft labels (MixUp/CutMix) ---
    
            targets_a, targets_b, lam = target
            
            # Get dimensions
            num_classes = x.size(-1)
            
            # Create smoothed "a" labels
            smooth_a = torch.full_like(x, self.smoothing / (num_classes - 1))
            smooth_a.scatter_(-1, targets_a.unsqueeze(-1), self.confidence)
            
            # Create smoothed "b" labels
            smooth_b = torch.full_like(x, self.smoothing / (num_classes - 1))
            smooth_b.scatter_(-1, targets_b.unsqueeze(-1), self.confidence)
            
            # Calculate interpolated loss
            loss_a = -torch.sum(smooth_a * logprobs, dim=-1)
            loss_b = -torch.sum(smooth_b * logprobs, dim=-1)
            loss = lam * loss_a + (1.0 - lam) * loss_b
            
        else:
            # --- Hard labels (No MixUp/CutMix) ---
            # target = tensor of shape [B]
            num_classes = x.size(-1)
            
            # Create smoothed "hard" labels
            smooth_hard = torch.full_like(x, self.smoothing / (num_classes - 1))
            smooth_hard.scatter_(-1, target.unsqueeze(-1), self.confidence)
            
            loss = -torch.sum(smooth_hard * logprobs, dim=-1)

        # Return the mean loss over the batch
        return loss.mean()



# Advanced augmentation classes
class CutMix:
    def __init__(self, alpha=1.0, p=0.5):
        self.alpha = alpha
        self.p = p

    def __call__(self, batch):
        if random.random() > self.p:
            return batch
        
        images, labels = batch
        batch_size = images.size(0)
        
        # Generate random indices for mixing
        indices = torch.randperm(batch_size)
        
        # Generate lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Generate bounding box
        W, H = images.size(3), images.size(2)
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply cutmix
        images[:, :, bby1:bby2, bbx1:bbx2] = images[indices, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        # Return labels as a tuple
        return images, (labels, labels[indices], lam)


class MixUp:
    def __init__(self, alpha=0.2, p=0.5):
        self.alpha = alpha
        self.p = p

    def __call__(self, batch):
        if random.random() > self.p:
            return batch
        
        images, labels = batch
        batch_size = images.size(0)
        indices = torch.randperm(batch_size)
        lam = np.random.beta(self.alpha, self.alpha)
        
        mixed_images = lam * images + (1 - lam) * images[indices]
        
        # Return labels as a tuple
        return mixed_images, (labels, labels[indices], lam)





def train_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch,
                use_mixup=False, use_cutmix=False):
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    # --- Create augmentation functions---
    # (Using common defaults, you can tune alphas in config)
    cutmix_fn = CutMix(alpha=1.0, p=1.0) 
    mixup_fn = MixUp(alpha=0.6, p=1.0)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        targets_hard = targets.to(device, non_blocking=True) # Keep original hard targets
        
        # Apply augmentations
        use_mix = use_mixup and random.random() < 0.5
        use_cut = use_cutmix and not use_mix
        
        # targets_for_loss will be (images, hard_labels) OR (images, (labels_a, labels_b, lam))
        targets_for_loss = targets_hard
        
        if use_mix:
            images, targets_for_loss = mixup_fn((images, targets_hard))
        elif use_cut:
            images, targets_for_loss = cutmix_fn((images, targets_hard))
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            # The new criterion can handle both hard and soft targets
            loss = criterion(outputs, targets_for_loss)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate accuracy on the *original* hard targets
        acc1 = accuracy(outputs, targets_hard, topk=(1,))[0]
        
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc@1': f'{top1.avg:.2f}%',
            'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
        })
    
    return losses.avg, top1.avg


def validate_epoch(model, dataloader, criterion, device, use_tta=False):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images, use_tta=use_tta)
            # Validation uses hard targets, so criterion works fine
            loss = criterion(outputs, targets) 
            acc1 = accuracy(outputs, targets, topk=(1,))
            
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc@1': f'{top1.avg:.2f}%'
            })
    
    return losses.avg, top1.avg


def create_cross_validation_splits(dataset, labels, n_splits=2, random_state=42):
    """Create stratified cross-validation splits"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Convert labels to numpy if needed
    if isinstance(labels, torch.Tensor):
        labels_np = labels.cpu().numpy()
    else:
        labels_np = np.array(labels)
    
    splits = list(skf.split(range(len(dataset)), labels_np))
    return splits


def main(config=None):
    """
    Main training function that accepts external config
    If no config is provided, will load default config
    """
    if config is None:
        # Fallback to importing config if none provided
        from configs.config import get_config # Corrected import
        config = get_config('adaptformer_baseline')
        print("Warning: Using fallback config. Consider passing config explicitly.")
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    os.makedirs(config['save_dir'], exist_ok=True)
    device = torch.device(config.get('device', 'cuda') if torch.cuda.is_available() else 'cpu') # Safer get
    print(f"Using device: {device}")

    # Load initial data to get dataset info
    train_loader, val_loader, num_classes = create_data_loaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        config=config # Pass config for augmentations
    )

    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"Val samples: {len(val_loader.dataset)}")
    else:
        print("No validation loader found.")

    # Cross-validation setup
    if config['use_cross_validation']:
        print(f"\nUsing {config['n_folds']}-fold cross validation", flush=True)
        
        # Get full dataset for splitting
        full_train_dataset, train_labels = create_data_loaders(
            config['data_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            return_full=True, # Ask for dataset, not loader
            config=config
        )

        splits = create_cross_validation_splits(full_train_dataset, train_labels, config['n_folds'])
        
        best_cv_acc = 0.0
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(splits):
            print(f"\n{'='*50}", flush=True)
            print(f"FOLD {fold + 1}/{config['n_folds']}", flush=True)
            print(f"{'='*50}", flush=True)
            
            # Create fold datasets
            fold_train_dataset = Subset(full_train_dataset, train_idx)
            fold_val_dataset = Subset(full_train_dataset, val_idx)
            
            fold_train_loader = DataLoader(
                fold_train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=config['num_workers'],
                pin_memory=True
            )
            
            fold_val_loader = DataLoader(
                fold_val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                pin_memory=True
            )
            
            print(f"Fold {fold + 1} - Train: {len(fold_train_dataset)}, Val: {len(fold_val_dataset)}", flush=True)
            
            # Create model for this fold
            model = create_model(
                num_classes=num_classes,
                peft_method=config['peft_method'],
                rank=config['lora_rank'],
                alpha=config['lora_alpha'],
                adapter_dim=config['adapter_dim'],
                use_multi_scale=config['use_multi_scale']
            )
            model = model.to(device)
            
            best_fold_acc = train_fold(
                model, fold_train_loader, fold_val_loader,
                config, device, fold
            )
            
            fold_results.append(best_fold_acc)
            print(f"Fold {fold + 1} best accuracy: {best_fold_acc:.2f}%", flush=True)
            
            if best_fold_acc > best_cv_acc:
                best_cv_acc = best_fold_acc
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'fold': fold + 1,
                    'accuracy': best_fold_acc
                }, os.path.join(config['save_dir'], 'best_cv_model.pth'))
        
        # Print cross-validation results
        print(f"\n{'='*50}")
        print(f"CROSS-VALIDATION RESULTS")
        print(f"{'='*50}")
        for i, acc in enumerate(fold_results):
            print(f"Fold {i + 1}: {acc:.2f}%", flush=True)
        print(f"Mean CV Accuracy: {np.mean(fold_results):.2f}% ± {np.std(fold_results):.2f}%", flush=True)
        print(f"Best CV Accuracy: {best_cv_acc:.2f}%", flush=True)
        
    else:
        # Standard training without cross-validation
        print("\nStarting Standard Training (no cross-validation)", flush=True)
        model = create_model(
            num_classes=num_classes,
            peft_method=config['peft_method'],
            rank=config['lora_rank'],
            alpha=config['lora_alpha'],
            adapter_dim=config['adapter_dim'],
            use_multi_scale=config['use_multi_scale']
        )
        model = model.to(device)
        
        best_acc = train_fold(model, train_loader, val_loader, config, device, fold=0)
        print(f"Final accuracy: {best_acc:.2f}%")
        
        save_path = os.path.join(config['save_dir'], 'best_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'accuracy': best_acc,
            'fold': 0
        }, save_path)
        
        print(f"Model saved in {save_path}")


def train_fold(model, train_loader, val_loader, config, device, fold):
    """Train a single fold with progressive training strategy"""
    trainable, total = model.get_trainable_params()
    print(f"Trainable parameters: {trainable:,} ({trainable/total*100:.2f}%)", flush=True)
    print(f"Total parameters: {total:,}", flush=True)
    
    criterion = LabelSmoothingLoss(smoothing=config['label_smoothing'])
    
    phases = config.get('phases', [
        {'name': 'Full Training', 'epochs': config['epochs'], 'lr': config['lr'], 'freeze_peft': False}
    ])
    
    best_acc = 0.0
    os.makedirs(config['save_dir'], exist_ok=True)
    total_epochs = 0
    start_time = time.time()
    
    for phase_idx, phase in enumerate(phases):
        print(f"\n--- Phase {phase_idx + 1}: {phase['name']} ---")
        
        # Freeze/unfreeze parameters based on phase
        if phase['freeze_peft']:
            # Only train classifier
            for pname, p in model.named_parameters():
                p.requires_grad = ('classifier' in pname)
        else:
            # Train PEFT modules + classifier + last 4 backbone blocks (+ LayerNorms)
            UNFREEZE_LAST_N = 4

            block_indices = []
            for pname, _ in model.named_parameters():
                if '.blocks.' in pname:
                    try:
                        after = pname.split('.blocks.')[1]
                        idx = int(after.split('.')[0])
                        block_indices.append(idx)
                    except Exception:
                        pass
            block_indices = sorted(set(block_indices))
            last_n = set(block_indices[-UNFREEZE_LAST_N:]) if block_indices else set()

            def in_last_n_blocks(n: str) -> bool:
                return any(f'.blocks.{i}.' in n for i in last_n)

            def is_layernorm(n: str) -> bool:
                low = n.lower()
                return ('norm.' in low) or ('layernorm' in low) or ('.ln' in low)

            for pname, p in model.named_parameters():
                train_this = (
                    any(tag in pname for tag in ['lora_', 'adaptformer_', 'classifier', 'multi_scale'])
                    or in_last_n_blocks(pname)
                    or is_layernorm(pname)
                )
                p.requires_grad = train_this

        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=phase['lr'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )

        
       
        scheduler = create_scheduler(
            optimizer,
            epochs=phase['epochs'],
            warmup_epochs=config['warmup_epochs'], # <-- Pass warmup
            scheduler_type='linear_warmup_cosine' # <-- Use the scheduler we fixed
        )
        
        scaler = GradScaler(enabled=torch.cuda.is_available())

        
        for epoch in range(phase['epochs']):
            global_epoch = total_epochs + epoch
            
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, scaler, device,
                global_epoch + 1, # Use 1-based epoch for display
                config['use_mixup'], config['use_cutmix']
            )
            
            # Validation
            # This 'val_loader' will be None if run_training.py disables it
            if val_loader is not None:
                val_loss, val_acc = validate_epoch(
                    model, val_loader, criterion, device,
                    use_tta=config['use_tta'] and phase_idx == len(phases) - 1  # TTA only in final phase
                )
                current_acc = val_acc
                print(f"Epoch {global_epoch + 1}: Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}% | Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f}", flush=True)
            else:
                current_acc = train_acc
                print(f"Epoch {global_epoch + 1}: Train Acc {train_acc:.2f}% | Train Loss {train_loss:.4f}", flush=True)
            
            scheduler.step()
            
            # Save best model
            if current_acc > best_acc:
                best_acc = current_acc
                torch.save({
                    'epoch': global_epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': config,
                    'fold': fold
                }, os.path.join(config['save_dir'], f'best_model_fold_{fold}.pth'))
                print(f"New best accuracy: {best_acc:.2f}%", flush=True)
        
        total_epochs += phase['epochs']
    
    elapsed_time = time.time() - start_time
    print(f"Fold {fold} training completed in {elapsed_time/3600:.2f} hours", flush=True)
    print(f"Best accuracy: {best_acc:.2f}%", flush=True)
    
    return best_acc


if __name__ == "__main__":
    main()