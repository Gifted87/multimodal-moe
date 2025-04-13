import torch
import torch.nn as nn
import math

from .moe_layer import MultimodalMoE, Expert # Import MoE components
from .text_encoder import AlibiAttention # Can reuse ALiBi attention if needed

class CrossModalAttention(nn.Module):
    """ Performs attention from one modality (query) to another (key/value) """
    def __init__(self, hidden_dim, num_heads, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        self.scale = self.head_dim ** -0.5

        # Separate projections for query (modality A) and key/value (modality B)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.kv_proj = nn.Linear(hidden_dim, hidden_dim * 2, bias=False) # Project K and V together
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query_modality, kv_modality, attention_mask=None):
        # query_modality: [batch, seq_len_q, hidden_dim] (e.g., text)
        # kv_modality: [batch, seq_len_kv, hidden_dim] (e.g., image patches)
        # attention_mask: Mask for kv_modality (e.g., image padding), shape [batch, seq_len_kv]

        batch_size, seq_len_q, _ = query_modality.shape
        _, seq_len_kv, _ = kv_modality.shape

        # Project Q, K, V
        q = self.q_proj(query_modality) # [batch, seq_len_q, hidden_dim]
        k, v = self.kv_proj(kv_modality).chunk(2, dim=-1) # [batch, seq_len_kv, hidden_dim] each

        # Reshape for multi-head attention
        q = q.reshape(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        # Shapes: [batch, num_heads, seq_len_*, head_dim]

        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # [batch, num_heads, seq_len_q, seq_len_kv]

        # Apply attention mask (for key/value modality padding)
        if attention_mask is not None:
            # Expand mask: [batch, 1, 1, seq_len_kv]
            mask = attention_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
            attn_scores = attn_scores.masked_fill(~mask, float('-inf')) # True means keep

        # Softmax and Dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights) # [batch, num_heads, seq_len_q, seq_len_kv]

        # Weighted sum
        attn_output = torch.matmul(attn_weights, v) # [batch, num_heads, seq_len_q, head_dim]

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len_q, -1) # [batch, seq_len_q, hidden_dim]
        output = self.out_proj(attn_output)

        return output


class FusionBlock(nn.Module):
    """
    A block that performs cross-modal attention followed by an optional MoE FFN.
    Can attend Text -> Image and Image -> Text.
    """
    def __init__(self,
                 hidden_dim,
                 num_heads,
                 use_moe_ffn=True,
                 moe_config=None, # Dict with MoE params if use_moe_ffn is True
                 ffn_dim=None,
                 dropout_rate=0.1):
        super().__init__()
        self.use_moe_ffn = use_moe_ffn

        self.cross_attn_t2i = CrossModalAttention(hidden_dim, num_heads, dropout_rate)
        self.norm_t2i_in = nn.LayerNorm(hidden_dim)
        self.norm_t2i_out = nn.LayerNorm(hidden_dim)
        self.dropout_t2i = nn.Dropout(dropout_rate)

        self.cross_attn_i2t = CrossModalAttention(hidden_dim, num_heads, dropout_rate)
        self.norm_i2t_in = nn.LayerNorm(hidden_dim)
        self.norm_i2t_out = nn.LayerNorm(hidden_dim)
        self.dropout_i2t = nn.Dropout(dropout_rate)

        if use_moe_ffn:
            assert moe_config is not None, "MoE config required if use_moe_ffn is True"
            self.moe_ffn_text = MultimodalMoE(hidden_dim=hidden_dim, **moe_config)
            self.moe_ffn_image = MultimodalMoE(hidden_dim=hidden_dim, **moe_config)
            self.norm_moe_text = nn.LayerNorm(hidden_dim)
            self.norm_moe_image = nn.LayerNorm(hidden_dim)
            self.dropout_moe_text = nn.Dropout(dropout_rate)
            self.dropout_moe_image = nn.Dropout(dropout_rate)
        else:
            # Use standard FFN if MoE is disabled
            if ffn_dim is None: ffn_dim = hidden_dim * 4
            self.ffn_text = nn.Sequential(
                nn.Linear(hidden_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout_rate),
                nn.Linear(ffn_dim, hidden_dim), nn.Dropout(dropout_rate)
            )
            self.ffn_image = nn.Sequential(
                 nn.Linear(hidden_dim, ffn_dim), nn.GELU(), nn.Dropout(dropout_rate),
                 nn.Linear(ffn_dim, hidden_dim), nn.Dropout(dropout_rate)
            )
            self.norm_ffn_text = nn.LayerNorm(hidden_dim)
            self.norm_ffn_image = nn.LayerNorm(hidden_dim)
            self.dropout_ffn_text = nn.Dropout(dropout_rate)
            self.dropout_ffn_image = nn.Dropout(dropout_rate)

    def forward(self, text_features, image_features, text_mask=None, image_mask=None):
        # text_features: [batch, seq_len_t, hidden_dim]
        # image_features: [batch, seq_len_i, hidden_dim]
        # text_mask: [batch, seq_len_t]
        # image_mask: [batch, seq_len_i]

        # 1. Text attends to Image
        res_text = text_features
        text_norm = self.norm_t2i_in(text_features)
        image_norm = self.norm_i2t_in(image_features) # Normalize image features once here
        t2i_attn_out = self.cross_attn_t2i(text_norm, image_norm, attention_mask=image_mask)
        text_features = res_text + self.dropout_t2i(t2i_attn_out)
        text_features = self.norm_t2i_out(text_features) # Normalize after residual

        # 2. Image attends to Text
        res_image = image_features
        # Use already normalized text_norm from above
        i2t_attn_out = self.cross_attn_i2t(image_norm, text_norm, attention_mask=text_mask)
        image_features = res_image + self.dropout_i2t(i2t_attn_out)
        image_features = self.norm_i2t_out(image_features) # Normalize after residual

        # 3. Apply FFN (MoE or Dense)
        if self.use_moe_ffn:
            # Text FFN
            res_text = text_features
            text_norm = self.norm_moe_text(text_features)
            text_features = res_text + self.dropout_moe_text(self.moe_ffn_text(text_norm))

            # Image FFN
            res_image = image_features
            image_norm = self.norm_moe_image(image_features)
            image_features = res_image + self.dropout_moe_image(self.moe_ffn_image(image_norm))
        else:
            # Text FFN
            res_text = text_features
            text_norm = self.norm_ffn_text(text_features)
            text_features = res_text + self.dropout_ffn_text(self.ffn_text(text_norm))

            # Image FFN
            res_image = image_features
            image_norm = self.norm_ffn_image(image_features)
            image_features = res_image + self.dropout_ffn_image(self.ffn_image(image_norm))


        return text_features, image_features

    def get_aux_losses(self):
        """ Collect auxiliary losses from MoE layers in this block """
        losses = []
        if self.use_moe_ffn:
            losses.append(self.moe_ffn_text.get_aux_loss())
            losses.append(self.moe_ffn_image.get_aux_loss())
        # Filter out zero tensors or None values if any MoE layer wasn't used
        return [loss for loss in losses if loss is not None and loss.item() != 0]