"""
Configuration file for easy hyperparameter tuning
"""

# Optimal configurations for different scenarios
CONFIGS = {
    'adaptformer_baseline': {
        'peft_method': 'adaptformer',
        'adapter_dim': 64,
        'lora_rank': 8,
        'lora_alpha': 32,
        # batch_size will be set dynamically
        'epochs': 5,
        'lr': 5e-4,
        'weight_decay': 0.1,
        'warmup_epochs': 3,
        'use_mixup': True,
        'use_cutmix': True,
        'use_tta': True,
        'label_smoothing': 0.1,
        'use_multi_scale': False,
        'use_cross_validation': True,
        'n_folds': 2,
        'phases': [
            {'name': 'Classifier Only', 'epochs': 7, 'lr': 1e-3, 'freeze_peft': True},
            {'name': 'PEFT + Classifier', 'epochs': 5, 'lr': 1e-3, 'freeze_peft': False},
            {'name': 'Fine-tuning', 'epochs': 3, 'lr': 1e-3, 'freeze_peft': False}
        ],
    },

    'lora_experiment': {
        'peft_method': 'lora',
        'adapter_dim': 32,
        'lora_rank': 16,  # Higher rank for LoRA
        'lora_alpha': 32,
        # batch_size will be set dynamically?
        'epochs': 5,
        'lr': 5e-4,
        'weight_decay': 0.1,
        'warmup_epochs': 3,
        'use_mixup': True,
        'use_cutmix': True,
        'use_tta': True,
        'label_smoothing': 0.1,
        'use_multi_scale': True,
        'use_cross_validation': False,
        'n_folds': 1,
        'phases': [
            {'name': 'Classifier Only', 'epochs': 7, 'lr': 1e-3, 'freeze_peft': True},
            {'name': 'PEFT + Classifier', 'epochs': 5, 'lr': 1e-3, 'freeze_peft': False},
            {'name': 'Fine-tuning', 'epochs': 3, 'lr': 1e-3, 'freeze_peft': False}
        ],
    },

    'both_methods': {
        'peft_method': 'adaptformer',
        'adapter_dim': 48,
        'save_every': 1,
        'lora_rank': 16,
        'lora_alpha': 32,  
        'batch_size': 256,
        'epochs': 17,
        'lr': 2e-4,
        'weight_decay': 0.05,
        'warmup_epochs': 2,
        'use_mixup': True,
        'use_cutmix': True,
        'use_tta': False, 
        'label_smoothing': 0.15,
        'use_multi_scale': False,
        'use_cross_validation': False,
        'n_folds': 2,
        
        # --- ADDED: Augmentation parameters ---
        'use_trivial_aug': False,
        'rand_aug_n': 2,  # Number of RandAugment ops
        'rand_aug_m': 9,  # Magnitude of RandAugment ops
        # ---
        
        # --- UPDATED: Training phases ---
        'phases': [
            {'name': 'Classifier Only', 'epochs': 2, 'lr': 1e-3, 'freeze_peft': True, 'aug_prob': 0.0},
            {'name': 'PEFT + Classifier', 'epochs': 10, 'lr': 6e-4, 'freeze_peft': False, 'aug_prob': 0.8},
            {'name': 'Fine-tuning', 'epochs': 7, 'lr': 5e-6, 'freeze_peft': False, 'aug_prob': 0.2} 
        ],
    },
    
}


def get_config(config_name='adaptformer_baseline', batch_size=None, gpu_memory_gb=None):
    """
    Dynamically gets configuration and sets batch size.
    Priority:
    1. Manual --batch_size flag
    2. 'batch_size' in config
    3. Auto-detection by --gpu_memory
    4. Default fallback
    """

    # Base config with sane defaults
    base_config = {
        'data_dir': '/common/home/projectgrps/CS701/CS701G1/cs701-course-data',
        'save_dir': '/common/home/projectgrps/CS701/CS701G1/transfer/src/checkpoints',
        'num_workers': 8,
        'device': 'cuda',
        'peft_method': 'adaptformer',
        'adapter_dim': 64,
        'lora_rank': 8,
        'lora_alpha': 32,
        'epochs': 5,
        'lr': 5e-4,
        'weight_decay': 0.1,
        'warmup_epochs': 3,
        'use_mixup': False,
        'use_cutmix': False,
        'use_tta': False,
        'label_smoothing': 0.1,
        'use_multi_scale': False,
        'use_cross_validation': False,
        'n_folds': 1,
        'phases': [],
        'pretrained_backbone_path': None # Default to None
    }

    # Load config by name
    if config_name in CONFIGS:
        # Need to handle potential circular dependency if **CONFIGS[...] is used above
        temp_config = CONFIGS[config_name].copy()
        if '**CONFIGS' in str(temp_config): # Simple check, might need refinement
             parent_name = temp_config.pop('**CONFIGS') # Assuming simple inheritance for now
             parent_config = CONFIGS[parent_name].copy()
             parent_config.update(temp_config) # Child overrides parent
             base_config.update(parent_config)
        else:
             base_config.update(temp_config)

    else:
        print(f"Config {config_name} not found, using adaptformer_baseline")
        base_config.update(CONFIGS['adaptformer_baseline'])

    # Handle batch size logic
    if batch_size is not None:
        # Manual override takes highest priority
        base_config['batch_size'] = batch_size
        print(f"Using manual batch size override: {batch_size}")
    elif 'batch_size' not in base_config and gpu_memory_gb is not None:
        # Adaptive batch size based on GPU memory, only if not already set
        if gpu_memory_gb >= 46:
            base_config['batch_size'] = 512
        elif gpu_memory_gb >= 24:
            base_config['batch_size'] = 256
        else:
            base_config['batch_size'] = 128
        print(f"Auto-configured batch size based on {gpu_memory_gb}GB GPU: {base_config['batch_size']}")
    elif 'batch_size' not in base_config:
        # Default fallback
        base_config['batch_size'] = 256
        print(f"Using default batch size: {base_config['batch_size']}")

    return base_config


def print_config(config):
    """Pretty print configuration"""
    print("="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    for key, value in config.items():
        if key == 'phases':
            print(f"- {key}:")
            for i, phase in enumerate(value):
                print(f"    Phase {i+1} ({phase['name']}): {phase['epochs']} epochs, LR={phase.get('lr', 'N/A')}") # Added get() for safety
        else:
            print(f"- {key}: {value}")
    print("="*50, flush=True)

