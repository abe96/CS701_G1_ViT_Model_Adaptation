import sys
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import json
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data.dataset import create_data_loaders
from models.peft_vit_improved import create_model
from utils.metrics import AverageMeter, accuracy
from configs.config import get_config, print_config
from utils.scheduler import create_scheduler

try:
    from train_augment import CutMix, MixUp, LabelSmoothingLoss
    print("Successfully imported advanced training components (CutMix, MixUp, correct Criterion).")
except ImportError as e:
    print(f"CRITICAL ERROR: Error importing from train_augment.py: {e}")
    print("This file is required. Please ensure it is in the same directory.")
    sys.exit(1)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Training script with configuration support')
    parser.add_argument('config_name', nargs='?', default='adaptformer_baseline',
                        help='Configuration name (default: adaptformer_baseline)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size from config')
    parser.add_argument('--gpu_memory', type=float, default=None,
                        help='GPU memory in GB for auto batch size calculation')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Check GPU memory if not provided
    gpu_memory_gb = args.gpu_memory
    if gpu_memory_gb is None and torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Detected GPU Memory: {gpu_memory_gb:.1f} GB")

    # Get configuration with proper batch size handling
    config = get_config(
        config_name=args.config_name, 
        gpu_memory_gb=gpu_memory_gb,
        batch_size=args.batch_size
    )
    
    print("\n" + "="*60)
    print(f"RUNNING CONFIGURATION: {args.config_name.upper()}")
    print("="*60)
    print_config(config)

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, num_classes = create_data_loaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        config=config
    )

    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"Val samples: {len(val_loader.dataset)}")

    # Create model
    print("\nCreating model...")
    model = create_model(
        num_classes=num_classes,
        peft_method=config['peft_method'],
        rank=config['lora_rank'],
        alpha=config['lora_alpha'],
        adapter_dim=config['adapter_dim'],
        use_multi_scale=config['use_multi_scale']
    )
    model = model.to(device)

    trainable, total = model.get_trainable_params()
    print(f"Trainable parameters: {trainable:,} ({trainable/total*100:.2f}%)")
    print(f"Total parameters: {total:,}")

    # Import and run training with proper config passing
    if config['use_cross_validation']:
        print("\n Starting Cross-Validation Training...", flush=True)
        from train_augment import main as train_main 
        train_main(config)
    else:
        print("\n Starting Standard Training...", flush=True)
        train_standard(model, train_loader, val_loader, config, device)



def train_standard(model, train_loader, val_loader, config, device):
    """
    Standard training function, FINAL FIXED version:
    - Layer-Wise Learning Rates (LWLR)
    - Re-uses optimizer for "soft" phase changes (preserves momentum)
    - Applies warmup per-phase (prevents LR shock)
    - Reads 'aug_prob' from phase config
    """
    
    # --- Setup components ---
    criterion = LabelSmoothingLoss(smoothing=config['label_smoothing'])
    scaler = GradScaler()

    # --- Create MixUp/CutMix functions ---
    cutmix_fn = CutMix(alpha=1.0)
    mixup_fn = MixUp(alpha=0.6)
    # ---

    phases = config.get('phases', [
        {'name': 'Full Training', 'epochs': config['epochs'], 'lr': config['lr'], 'freeze_peft': False}
    ])

    best_acc = 0.0
    os.makedirs(config['save_dir'], exist_ok=True)
    save_every = config.get('save_every', 5)

    print(f"Training with {len(phases)} phases...", flush=True)
    start_time = time.time()
    
    # --- FIX: Initialize optimizer and freeze status ---
    total_epochs = 0
    optimizer = None
    last_freeze_status = None
    # ---

    for phase_idx, phase in enumerate(phases):
        print(f"\n--- Phase {phase_idx + 1}: {phase['name']} ({phase['epochs']} epochs) ---")
        
        current_aug_prob = phase.get('aug_prob', 0.9) 
        print(f"LR: {phase['lr']}, Freeze PEFT: {phase['freeze_peft']}, Aug Prob: {current_aug_prob}", flush=True)

       #param unfreeze logic
        if phase['freeze_peft']:
            for name, param in model.named_parameters():
                param.requires_grad = ('classifier' in name)
        else:
            UNFREEZE_LAST_N = 4
            block_indices = []
            for name, _ in model.named_parameters():
                if '.blocks.' in name:
                    try:
                        after = name.split('.blocks.')[1]
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

            for name, param in model.named_parameters():
                train_this = (
                    any(tag in name for tag in ['lora_', 'adaptformer_', 'classifier', 'multi_scale']) or
                    in_last_n_blocks(name) or
                    is_layernorm(name)
                )
                param.requires_grad = train_this

        trainable_phase, total = model.get_trainable_params()
        print(f"Trainable params for this phase: {trainable_phase:,} ({trainable_phase/total*100:.2f}%)")

        # 1. Define Layer-Wise LR parameters
        backbone_lr_scale = 0.1 
        
    
        if phase['freeze_peft'] != last_freeze_status:
            print("Hard phase change. Creating new optimizer with Layer-Wise LRs.")
            
            # Create parameter groups for LWLR
            param_groups = [
                {"name": "backbone", "params": [], "lr": phase['lr'] * backbone_lr_scale},
                {"name": "peft_and_head", "params": [], "lr": phase['lr']},
            ]
            
            def is_backbone_layer(n: str, p: torch.nn.Parameter) -> bool:
                if not p.requires_grad: return False
                low = n.lower()
                is_block_or_norm = ('.blocks.' in low) or ('norm.' in low) or ('layernorm' in low) or ('.ln' in low)
                is_peft = any(tag in low for tag in ['lora_', 'adaptformer_'])
                return is_block_or_norm and not is_peft

            # Sort parameters into the two groups
            for name, param in model.named_parameters():
                if not param.requires_grad: continue
                if is_backbone_layer(name, param):
                    param_groups[0]["params"].append(param)
                else:
                    param_groups[1]["params"].append(param)

            print(f"  - Backbone group (LR={phase['lr'] * backbone_lr_scale:.1e}): {len(param_groups[0]['params'])} tensors")
            print(f"  - PEFT/Head group (LR={phase['lr']:.1e}): {len(param_groups[1]['params'])} tensors")

            # Create the optimizer
            optimizer = optim.AdamW(
                param_groups, 
                lr=phase['lr'], 
                weight_decay=config['weight_decay']
            )
            
    
            current_warmup_epochs = config['warmup_epochs'] # Need warmup for new optimizer?
        
        else:
            # "Soft" change (Phase 2 -> 3). RE-USE the optimizer.
            print("Soft phase change. Re-using optimizer and updating LRs.")
            
            if optimizer is None:
                print("Error: Optimizer not initialized. Should not happen.")
                continue

            # Update LRs for the *existing* groups
            for group in optimizer.param_groups:
                if group['name'] == "backbone":
                    group['lr'] = phase['lr'] * backbone_lr_scale
                else: # peft_and_head
                    group['lr'] = phase['lr']

            print(f"  - Updated Backbone group LR to {phase['lr'] * backbone_lr_scale:.1e}")
            print(f"  - Updated PEFT/Head group LR to {phase['lr']:.1e}")

          
            current_warmup_epochs = 0 # NO warmup for a re-used optimizer

        last_freeze_status = phase['freeze_peft']
        
        # --- END of optimizer logic ---

        scheduler = create_scheduler(
            optimizer,
            epochs=phase['epochs'],
            warmup_epochs=current_warmup_epochs, 
            scheduler_type='linear_warmup_cosine' 
        )

        print(f"Optimizer and Scheduler created for phase {phase_idx+1} with {current_warmup_epochs} warmup epochs.", flush=True)
        # --- Epoch Loop for this phase ---
        for epoch in range(phase['epochs']):
            global_epoch = total_epochs + epoch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
           
            print(f"\nEpoch {global_epoch + 1}/{sum(p['epochs'] for p in phases)}")

            model.train()
            train_losses = AverageMeter()
            train_accuracies = AverageMeter()

           
            pbar = tqdm(train_loader, desc=f'Epoch {global_epoch + 1}/{sum(p["epochs"] for p in phases)}')

            for i, (images, targets) in enumerate(pbar):
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

               
                targets_for_loss = targets 

                if (config['use_mixup'] or config['use_cutmix']) and random.random() < current_aug_prob:
                    if config['use_mixup'] and (not config['use_cutmix'] or random.random() < 0.5):
                        images, targets_for_loss = mixup_fn((images, targets))
                    elif config['use_cutmix']:
                        images, targets_for_loss = cutmix_fn((images, targets))
       

                optimizer.zero_grad()

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets_for_loss) 

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                acc1, = accuracy(outputs, targets, topk=(1,))
                train_losses.update(loss.item(), images.size(0))
                train_accuracies.update(acc1.item(), images.size(0))

               
                pbar.set_postfix({
                    'Loss': f'{train_losses.avg:.3f}',
                    'Acc': f'{train_accuracies.avg:.2f}%',
                    'LR_Head': f'{optimizer.param_groups[-1]["lr"]:.6f}' # Show LR for last group (head)
                })

            train_loss = train_losses.avg
            train_acc = train_accuracies.avg
            current_acc = train_acc
            vram_log = ""
            if torch.cuda.is_available():
                # Get the peak reserved memory since the last reset
                peak_mem_gb = torch.cuda.max_memory_reserved() / (1024**3)
                vram_log = f" | Peak VRAM: {peak_mem_gb:.2f} GB"

            if False: # val_loader is not None:
                pass
            else:
                print(f"Epoch {global_epoch+1}: Train Acc {train_acc:.2f}% | Train Loss {train_loss:.4f}{vram_log}", flush=True)

            scheduler.step()

            
            if current_acc > best_acc:
                best_acc = current_acc
                save_path = os.path.join(config['save_dir'], 'best_model.pth')
                try:
                    torch.save({
                        'epoch': global_epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_acc': best_acc,
                        'config': config
                    }, save_path)
                    print(f"New best accuracy: {best_acc:.2f}%. Model saved to {save_path}", flush=True)
                except Exception as e:
                    print(f"Failed to save best model: {e}", flush=True)

            if save_every > 0 and (epoch + 1) % save_every == 0:
                checkpoint_path = os.path.join(config['save_dir'], f'checkpoint_epoch_{global_epoch+1}.pth')
                try:
                    torch.save({
                        'epoch': global_epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_acc': best_acc,
                        'config': config
                    }, checkpoint_path)
                    print(f"Saved periodic checkpoint at epoch {global_epoch+1}", flush=True)
                except Exception as e:
                    print(f"Failed to save checkpoint at epoch {global_epoch+1}: {e}", flush=True)

        total_epochs += phase['epochs']

    
    final_path = os.path.join(config['save_dir'], 'final_model.pth')
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_acc': best_acc,
            'config': config,
            'epochs_completed': total_epochs
        }, final_path)
        print(f"Final model saved at {final_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time/3600:.2f} hours")
    print(f"Best training accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()