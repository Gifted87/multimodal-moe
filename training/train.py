import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset # Basic dataset/loader
from torch.utils.data.distributed import DistributedSampler # For DDP/FSDP

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap # Auto-wrapping for FSDP

import os
import time
import argparse
from tqdm import tqdm
import numpy as np

from models.multimodal_transformer import MultimodalTransformer
from training.train_utils import load_config, setup_optimizer, setup_scheduler, get_grad_scaler
from training.metrics import evaluate_model # Placeholder evaluation

# --- Dummy Dataset ---
# Replace with actual data loading (e.g., using Hugging Face datasets)
class DummyMultimodalDataset(Dataset):
    def __init__(self, config, length=1000):
        self.length = length
        self.config = config
        self.img_size = config.model.image_encoder.img_size
        self.seq_len = config.model.text_encoder.max_seq_len
        self.vocab_size = config.model.text_encoder.vocab_size
        self.num_classes = config.num_classes # Assumes num_classes is in config

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        # Make some padding for variety
        pad_len = np.random.randint(0, self.seq_len // 4)
        if pad_len > 0:
             input_ids[-pad_len:] = 0 # Pad token id
             attention_mask[-pad_len:] = 0

        pixel_values = torch.randn(3, self.img_size, self.img_size)
        image_mask = torch.ones(self.config.model.image_encoder.num_patches + 1) # Placeholder, assumes CLS token
        labels = torch.randint(0, self.num_classes, (1,), dtype=torch.long).squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            #"image_attention_mask": image_mask, # Optional if needed
            "labels": labels
        }

# --- Main Training Function ---
def train(config_path, local_rank):
    """ Main training loop """
    # --- Setup ---
    config = load_config(config_path)
    # Add num_classes to config (needed for dummy dataset/model head)
    config.num_classes = 3 # Example: 3 classes for MNLI

    # DDP/FSDP Setup
    is_distributed = config.training.fsdp_enabled
    if is_distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl") # Initialize distributed environment
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        print(f"Initialized distributed training on rank {rank}/{world_size}.")
    else:
        world_size = 1
        rank = 0
        print("Running non-distributed training.")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config.training.seed + rank) # Seed per process
    np.random.seed(config.training.seed + rank)

    # --- Model ---
    print("Initializing model...")
    # Define FSDP wrapping policy if enabled
    fsdp_params = dict(
        mixed_precision=config.training.mixed_precision,
        flatten_parameters=True, # Generally recommended
        reshard_after_forward=config.training.reshard_after_forward,
        cpu_offload=config.training.cpu_offload,
        # compute_dtype=torch.bfloat16 if config.training.mixed_precision else torch.float32, # For TPUs/newer GPUs
        # bucket_cap_mb= # Optional: Tune bucketing size
    ) if config.training.fsdp_enabled else {}

    # Use auto_wrap with enable_wrap context manager for FSDP
    model = MultimodalTransformer(config)
    if is_distributed:
         with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
             # Wrap layers based on size or class name if needed, or wrap the whole model
             # Wrapping individual large layers (like encoders) can be more efficient
             model = wrap(model) # Wrap the entire model
         model = FSDP(model, **fsdp_params) # Apply FSDP to the wrapped model instance
         print(f"Model wrapped with FSDP on rank {rank}.")
    else:
         model.to(device)
         print(f"Model moved to device {device}.")


    # --- Data ---
    print("Loading data...")
    # Replace DummyDataset with your actual dataset implementation
    train_dataset = DummyMultimodalDataset(config, length=1000 * world_size) # Larger dummy dataset
    eval_dataset = DummyMultimodalDataset(config, length=200 * world_size)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_distributed else None
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None), # Shuffle only if not distributed
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.data.batch_size * 2, # Often larger batch size for eval
        sampler=eval_sampler,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    print(f"Data loaded. Train batches: {len(train_dataloader)}, Eval batches: {len(eval_dataloader)}")


    # --- Optimizer, Scheduler, Scaler ---
    num_training_steps = config.training.max_train_steps
    optimizer = setup_optimizer(model, config)
    lr_scheduler = setup_scheduler(optimizer, config, num_training_steps)
    scaler = get_grad_scaler(config)

    # --- Training Loop ---
    print("Starting training...")
    start_time = time.time()
    global_step = 0
    train_loss = 0.0
    total_aux_loss_epoch = 0.0

    # Resume from checkpoint TODO

    # Calculate effective steps considering gradient accumulation
    num_update_steps_per_epoch = len(train_dataloader) // config.training.gradient_accumulation_steps
    num_epochs = (num_training_steps + num_update_steps_per_epoch -1) // num_update_steps_per_epoch # Ceiling division

    progress_bar = tqdm(range(num_training_steps), desc="Training Steps", disable=(rank != 0))

    for epoch in range(num_epochs):
        model.train()
        if is_distributed:
             train_sampler.set_epoch(epoch) # Ensure proper shuffling each epoch

        epoch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=(rank != 0))
        for step, batch in enumerate(epoch_iterator):
             # Move batch to device
             batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

             # Forward pass with Automatic Mixed Precision context
             with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                 outputs = model(**batch,
                                 return_loss=True,
                                 enable_checkpointing=config.training.gradient_checkpointing)
                 loss = outputs["loss"]
                 aux_loss = outputs.get("aux_loss", torch.tensor(0.0))

                 if loss is None:
                      print("Warning: Loss is None. Skipping step.")
                      continue # Should not happen if labels are provided

                 # Normalize loss for gradient accumulation
                 if config.training.gradient_accumulation_steps > 1:
                      loss = loss / config.training.gradient_accumulation_steps

             # Accumulate losses for logging
             train_loss += loss.item()
             total_aux_loss_epoch += aux_loss.item() / config.training.gradient_accumulation_steps # Log avg aux loss

             # Backward pass
             if scaler:
                 scaler.scale(loss).backward()
             else:
                 loss.backward()

             # Optimizer step (perform update after accumulation_steps)
             if (step + 1) % config.training.gradient_accumulation_steps == 0:
                 # Unscale gradients before clipping and optimizer step
                 if scaler:
                     if config.training.gradient_clipping > 0:
                          scaler.unscale_(optimizer) # Unscale for clipping
                          torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clipping)
                     scaler.step(optimizer)
                     scaler.update()
                 else:
                      if config.training.gradient_clipping > 0:
                           torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clipping)
                      optimizer.step()

                 # Zero gradients and update scheduler AFTER optimizer step
                 optimizer.zero_grad()
                 lr_scheduler.step()
                 global_step += 1
                 progress_bar.update(1)

                 # --- Logging ---
                 if global_step % config.training.log_steps == 0 and rank == 0:
                     avg_loss = train_loss / config.training.log_steps
                     avg_aux_loss = total_aux_loss_epoch / config.training.log_steps
                     elapsed_time = time.time() - start_time
                     lr = lr_scheduler.get_last_lr()[0]
                     print(f"Step: {global_step}/{num_training_steps} | LR: {lr:.2e} | Avg Loss: {avg_loss:.4f} | Avg Aux Loss: {avg_aux_loss:.4f} | Time: {elapsed_time:.2f}s")
                     # Reset accumulators for next logging interval
                     train_loss = 0.0
                     total_aux_loss_epoch = 0.0
                     # Log to TensorBoard/W&B here

                 # --- Evaluation ---
                 if global_step % config.training.eval_steps == 0:
                     if rank == 0: print(f"\n--- Evaluating at Step {global_step} ---")
                     eval_metrics = evaluate_model(model, eval_dataloader, config, device)
                     if rank == 0:
                          print(f"Evaluation results: {eval_metrics}")
                          # Log eval metrics
                          # Save checkpoint based on eval performance if needed
                     model.train() # Set back to train mode

                 # --- Save Checkpoint ---
                 if global_step % config.training.save_steps == 0 and rank == 0:
                     save_dir = os.path.join(config.training.output_dir, f"checkpoint-{global_step}")
                     os.makedirs(save_dir, exist_ok=True)
                     # Save model state (handle FSDP saving)
                     # FSDP requires specific saving logic to gather state dict
                     if is_distributed:
                          # Use FSDP state_dict saving method
                          # Requires consolidating state from all ranks
                          # See Fairscale/PyTorch FSDP documentation for recommended saving patterns
                          print("Saving FSDP model state... (Implementation required)")
                          # state_dict = model.state_dict() # Needs FSDP consolidation wrapper
                          # if rank == 0: torch.save(state_dict, os.path.join(save_dir, "pytorch_model.bin"))
                     else:
                          torch.save(model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

                     # Save optimizer, scheduler, config etc.
                     torch.save(optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))
                     torch.save(lr_scheduler.state_dict(), os.path.join(save_dir, "scheduler.pt"))
                     OmegaConf.save(config, os.path.join(save_dir, "training_config.yaml"))
                     print(f"Checkpoint saved to {save_dir}")


                 if global_step >= num_training_steps:
                     break # Exit inner loop

        if global_step >= num_training_steps:
             print("Maximum training steps reached.")
             break # Exit outer loop

    print("Training finished.")
    total_time = time.time() - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")

    # Final save
    if rank == 0:
        save_dir = os.path.join(config.training.output_dir, "final_model")
        os.makedirs(save_dir, exist_ok=True)
        # Save final model state (handle FSDP)
        print("Saving final model state... (FSDP handling required)")
        # ... FSDP save logic ...
        OmegaConf.save(config, os.path.join(save_dir, "model_config.yaml"))
        print(f"Final model saved to {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multimodal MoE Model")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to the configuration file")
    parser.add_argument("--local_rank", type=int, default=os.getenv('LOCAL_RANK', 0), help="Local rank for distributed training")
    args = parser.parse_args()

    train(args.config, args.local_rank)