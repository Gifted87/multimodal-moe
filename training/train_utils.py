import torch
from fairscale.optim import OSS # Optimizer State Sharding
from fairscale.optim.grad_scaler import ShardedGradScaler # For FSDP + AMP
from torch.optim import AdamW
from transformers import get_scheduler # Hugging Face schedulers
from omegaconf import OmegaConf

def load_config(config_path):
    """ Loads configuration from YAML using OmegaConf """
    base_config = OmegaConf.load('configs/base.yaml') # Load base config first
    if config_path != 'configs/base.yaml':
         override_config = OmegaConf.load(config_path)
         config = OmegaConf.merge(base_config, override_config)
    else:
         config = base_config
    print("Configuration loaded:")
    print(OmegaConf.to_yaml(config))
    return config

def setup_optimizer(model, config):
    """ Sets up the optimizer (AdamW wrapped with OSS if FSDP is enabled) """
    # Filter parameters that require gradients
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.weight", "norm.weight"] # Standard no_decay params
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": config.training.weight_decay,
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]

    if config.training.fsdp_enabled:
        print("Using Fairscale OSS optimizer.")
        optimizer = OSS(
            params=optimizer_grouped_parameters,
            optim=AdamW, # Base optimizer
            lr=config.training.learning_rate,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            eps=config.training.adam_epsilon,
        )
    else:
        print("Using standard PyTorch AdamW optimizer.")
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=config.training.learning_rate,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            eps=config.training.adam_epsilon,
        )
    return optimizer

def setup_scheduler(optimizer, config, num_training_steps):
    """ Sets up the learning rate scheduler """
    print(f"Setting up {config.training.lr_scheduler_type} scheduler with {config.training.warmup_steps} warmup steps.")
    lr_scheduler = get_scheduler(
        name=config.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=num_training_steps,
    )
    return lr_scheduler

def get_grad_scaler(config):
    """ Returns the appropriate GradScaler (Sharded or standard) """
    if config.training.fsdp_enabled and config.training.mixed_precision:
        print("Using ShardedGradScaler for FSDP + AMP.")
        return ShardedGradScaler()
    elif config.training.mixed_precision:
        print("Using standard GradScaler for AMP.")
        return torch.cuda.amp.GradScaler()
    else:
        print("Mixed precision disabled, no GradScaler needed.")
        return None # No scaler needed if AMP is off