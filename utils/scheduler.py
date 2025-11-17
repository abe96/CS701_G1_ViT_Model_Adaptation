import torch.optim.lr_scheduler as lr_scheduler
import math


def create_scheduler(optimizer, epochs, scheduler_type='cosine_warm_restarts', **kwargs):
    """
    Creates a learning rate scheduler with multiple options.
    
    Args:
        optimizer: The optimizer to attach the scheduler to.
        epochs: Total number of training epochs for this phase.
        scheduler_type: Type of scheduler ('cosine_warm_restarts', 'cosine_annealing', 'step', 'exponential')
        **kwargs: Additional arguments for customization.
    
    Returns:
        A PyTorch learning rate scheduler.
    """
    
    if scheduler_type == 'cosine_warm_restarts':
        # Cosine annealing with warm restarts (default)
        T_0 = kwargs.get('T_0', epochs)
        T_mult = kwargs.get('T_mult', 1)
        eta_min = kwargs.get('eta_min', 1e-6)
        
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
        
    elif scheduler_type == 'cosine_annealing':
        # Simple cosine annealing
        T_max = kwargs.get('T_max', epochs)
        eta_min = kwargs.get('eta_min', 1e-6)
        
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min
        )
        
    elif scheduler_type == 'step':
        # Step decay
        step_size = kwargs.get('step_size', epochs // 3)
        gamma = kwargs.get('gamma', 0.1)
        
        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
        
    elif scheduler_type == 'multistep':
        # Multi-step decay
        milestones = kwargs.get('milestones', [epochs // 2, epochs * 3 // 4])
        gamma = kwargs.get('gamma', 0.1)
        
        scheduler = lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )
        
    elif scheduler_type == 'exponential':
        # Exponential decay
        gamma = kwargs.get('gamma', 0.95)
        
        scheduler = lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma
        )
        
    elif scheduler_type == 'polynomial':
        # Polynomial decay (custom implementation)
        power = kwargs.get('power', 1.0)
        min_lr = kwargs.get('min_lr', 1e-6)
        
        def polynomial_decay(epoch):
            return max(min_lr, (1 - epoch / epochs) ** power)
        
        scheduler = lr_scheduler.LambdaLR(optimizer, polynomial_decay)
        
    elif scheduler_type == 'linear_warmup_cosine':
        # Linear warmup followed by cosine annealing (FIXED)
        warmup_epochs = kwargs.get('warmup_epochs', epochs // 10)
        eta_min = kwargs.get('eta_min', 1e-6)

        def lr_lambda(epoch):
            # epoch is 0-indexed
            if epoch < warmup_epochs:
                # Linear warmup
                if warmup_epochs > 0:
                    return (epoch + 1) / warmup_epochs  # <-- THE FIX
                else:
                    return 1.0  # No warmup
            else:
                # Cosine annealing phase
                cosine_epochs = epochs - warmup_epochs
                if cosine_epochs <= 0:
                    return 1.0 # Warmup was all epochs

                current_cosine_epoch = epoch - warmup_epochs
                
                if cosine_epochs == 1:
                    denominator = 1.0 # Avoid division by zero if 1 cosine epoch
                else:
                    denominator = cosine_epochs - 1 # 0 to N-1 steps
                
                progress = current_cosine_epoch / denominator
                return eta_min + (1 - eta_min) * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    print(f"Created {scheduler_type} scheduler for {epochs} epochs")
    return scheduler


def create_warmup_scheduler(optimizer, warmup_epochs, base_scheduler):
    """
    Creates a warmup scheduler that combines linear warmup with another scheduler.
    
    Args:
        optimizer: The optimizer
        warmup_epochs: Number of warmup epochs
        base_scheduler: The main scheduler to use after warmup
    
    Returns:
        A sequential scheduler
    """
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            if warmup_epochs > 0:
                return (epoch + 1) / warmup_epochs  # <-- THE FIX
            else:
                return 1.0
        return 1.0
    
    warmup_scheduler = lr_scheduler.LambdaLR(optimizer, warmup_lambda)
    
    return lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, base_scheduler],
        milestones=[warmup_epochs]
    )
