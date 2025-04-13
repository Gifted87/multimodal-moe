import torch
import torch.nn as nn
from omegaconf import DictConfig # For config handling

from .text_encoder import TextEncoderTransformer
from .image_encoder import ImageEncoderViT
from .fusion import FusionBlock
from .moe_layer import MultimodalMoE # May use MoE also in encoders if desired

class MultimodalTransformer(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        # --- Modality Encoders ---
        self.text_encoder = TextEncoderTransformer(
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.text_encoder.num_layers,
            num_heads=config.model.num_heads,
            vocab_size=config.model.text_encoder.vocab_size,
            max_seq_len=config.model.text_encoder.max_seq_len,
            alibi_enabled=config.model.text_encoder.alibi_enabled,
            dropout_rate=config.model.dropout_rate
        )

        self.image_encoder = ImageEncoderViT(
            hidden_dim=config.model.hidden_dim, # Ensure ViT output dim matches text
            num_layers=config.model.image_encoder.num_layers,
            num_heads=config.model.num_heads,
            patch_size=config.model.image_encoder.patch_size,
            img_size=config.model.image_encoder.img_size,
            use_lora=config.model.image_encoder.use_lora,
            lora_rank=config.model.image_encoder.lora_rank,
            dropout_rate=config.model.dropout_rate
        )

        # Optional: Project image features to text hidden dim if they differ
        # Example: if ViT uses 768 and text uses 2048
        # self.image_proj = nn.Linear(vit_hidden_dim, config.model.hidden_dim)

        # --- Fusion Layers ---
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(
                hidden_dim=config.model.hidden_dim,
                num_heads=config.model.num_heads,
                use_moe_ffn=config.model.fusion.use_moe_in_fusion,
                moe_config=config.model.moe, # Pass MoE config directly
                dropout_rate=config.model.dropout_rate
            ) for _ in range(config.model.fusion.num_layers)
        ])

        # --- Output Heads (Example: for classification or generation) ---
        # This part depends heavily on the downstream task (GLUE, VQA, Captioning)
        # Example: Text classification head (e.g., for GLUE MNLI)
        self.text_pooler = nn.Sequential(
             nn.Linear(config.model.hidden_dim, config.model.hidden_dim),
             nn.Tanh(),
             nn.Dropout(config.model.dropout_rate)
        ) # Pool the [CLS] token or average embeddings
        self.classifier_head = nn.Linear(config.model.hidden_dim, config.num_classes) # num_classes from config

        # Example: Contrastive loss head (CLIP-style)
        # self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


    def forward(self,
                input_ids=None,          # Text: [batch, seq_len_t]
                attention_mask=None,   # Text mask: [batch, seq_len_t]
                pixel_values=None,       # Image: [batch, channels, height, width]
                image_attention_mask=None, # Image mask (if patches can be padded) [batch, seq_len_i]
                labels=None,             # Labels for loss calculation [batch]
                return_loss=True,
                enable_checkpointing=False # For gradient checkpointing
               ):

        # 1. Encode Text
        if input_ids is not None:
            if enable_checkpointing:
                 # Apply checkpointing to text encoder layers if needed
                 text_features = torch.utils.checkpoint.checkpoint_sequential(
                     self.text_encoder.layers, self.config.training.gradient_checkpointing_chunks, # Chunks = number of layers to group
                     self.text_encoder.embedding(input_ids) # Start with embeddings
                 )
                 text_features = self.text_encoder.norm(text_features) # Apply final norm
            else:
                 text_features = self.text_encoder(input_ids, attention_mask=attention_mask)
                 # Shape: [batch, seq_len_t, hidden_dim]
        else:
            text_features = None

        # 2. Encode Image
        if pixel_values is not None:
             if enable_checkpointing:
                 # Apply checkpointing to image encoder layers
                 image_features = self.image_encoder.vit.embeddings(pixel_values) # Get initial embeddings
                 image_features = torch.utils.checkpoint.checkpoint_sequential(
                      self.image_encoder.vit.encoder.layer, self.config.training.gradient_checkpointing_chunks,
                      image_features
                 )
                 # Apply final norm if ViT model doesn't do it (Hugging Face ViT usually includes it)
             else:
                 image_features = self.image_encoder(pixel_values)
                 # Shape: [batch, seq_len_i, hidden_dim] (seq_len_i = num_patches + 1 for CLS)

             # Optional projection if dimensions mismatch
             # if hasattr(self, 'image_proj'):
             #    image_features = self.image_proj(image_features)
        else:
             image_features = None


        # 3. Fuse Modalities (if both are present)
        if text_features is not None and image_features is not None:
            for i, fusion_block in enumerate(self.fusion_blocks):
                 if enable_checkpointing:
                     # Checkpoint each fusion block
                     text_features, image_features = torch.utils.checkpoint.checkpoint(
                         fusion_block, text_features, image_features, text_mask=attention_mask, image_mask=image_attention_mask
                     )
                 else:
                     text_features, image_features = fusion_block(
                         text_features, image_features,
                         text_mask=attention_mask, image_mask=image_attention_mask
                     )

        # 4. Prepare Output and Calculate Loss (Task-specific)
        loss = None
        logits = None
        total_aux_loss = torch.tensor(0.0, device=text_features.device if text_features is not None else image_features.device)

        # Collect MoE auxiliary losses
        for module in self.modules():
            if isinstance(module, FusionBlock):
                 aux_losses = module.get_aux_losses()
                 for aux_loss in aux_losses:
                      total_aux_loss += aux_loss
            elif isinstance(module, MultimodalMoE): # If MoE used elsewhere
                 total_aux_loss += module.get_aux_loss()


        if return_loss and labels is not None:
            # Example: Text Classification (using pooled text output)
            if text_features is not None:
                pooled_output = self.text_pooler(text_features[:, 0]) # Use CLS token output
                logits = self.classifier_head(pooled_output) # [batch, num_classes]
                loss_fct = nn.CrossEntropyLoss()
                main_loss = loss_fct(logits.view(-1, self.config.num_classes), labels.view(-1))
                loss = main_loss + total_aux_loss # Add MoE aux loss
            else:
                # Handle cases where loss depends on image or combined features
                print("Warning: Loss calculation not defined for this input combination.")
                loss = total_aux_loss # Return only aux loss if main loss cannot be calculated

        # Construct output object (similar to Hugging Face models)
        output = {
            "loss": loss,
            "logits": logits,
            "text_features": text_features,
            "image_features": image_features,
            "aux_loss": total_aux_loss # Return unscaled aux loss for logging
        }
        # Filter out None values
        return {k: v for k, v in output.items() if v is not None}