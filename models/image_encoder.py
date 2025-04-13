import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer, EncoderBlock # Use torchvision's ViT structure
# Or use Hugging Face transformers ViTModel for more features
from transformers import ViTConfig, ViTModel

from .lora_layer import LoRALinear # Import LoRA helper

class ImageEncoderViT(nn.Module):
    def __init__(self,
                 hidden_dim=768, # Base ViT dimension, adjust if needed
                 num_layers=12,
                 num_heads=12,
                 patch_size=16,
                 img_size=224,
                 use_lora=False,
                 lora_rank=8,
                 dropout_rate=0.1):
        super().__init__()

        # Configure ViT using Hugging Face implementation for robustness
        config = ViTConfig(
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            image_size=img_size,
            patch_size=patch_size,
            intermediate_size=hidden_dim * 4, # Standard FFN expansion
            hidden_dropout_prob=dropout_rate,
            attention_probs_dropout_prob=dropout_rate,
        )
        self.vit = ViTModel(config, add_pooling_layer=False) # Get sequence of patch embeddings

        # Apply LoRA if specified
        if use_lora:
            print(f"Applying LoRA (rank={lora_rank}) to ViT...")
            # Apply LoRA only to attention and feed-forward layers
            LoRALinear.apply_to_model(self.vit.encoder, rank=lora_rank, alpha=lora_rank, # Common practice: alpha=rank
                                      layers_to_adapt=["query", "key", "value", "dense", "intermediate", "output"])
            print("LoRA applied to ViT.")


    def forward(self, images):
        # images: [batch_size, channels, height, width]
        outputs = self.vit(pixel_values=images)
        # Return sequence of embeddings (patch embeddings + CLS token if configured)
        # Shape: [batch_size, num_patches + 1, hidden_dim]
        return outputs.last_hidden_state