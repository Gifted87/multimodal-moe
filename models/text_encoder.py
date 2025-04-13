import torch
import torch.nn as nn
import math

# ALiBi Implementation (adapted for standard MultiheadAttention)
# Adds a bias to the attention scores based on relative position
def get_alibi_slopes(num_heads):
    """ Gets the ALiBi slopes for each head """
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(num_heads).is_integer():
        return get_slopes_power_of_2(num_heads)
    else:
        # Fallback for non-power-of-2 heads (e.g., interpolation or simple range)
        closest_power_of_2 = 2**math.floor(math.log2(num_heads))
        return get_slopes_power_of_2(closest_power_of_2) + \
               get_slopes_power_of_2(2*closest_power_of_2)[0:num_heads-closest_power_of_2]


class AlibiAttention(nn.Module):
    """ MultiheadAttention layer modified to incorporate ALiBi """
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(hidden_dim, hidden_dim * 3) # Combined QKV projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Calculate ALiBi slopes (once)
        self.slopes = torch.tensor(get_alibi_slopes(num_heads))
        self._alibi_bias_cache = {} # Cache for efficiency

    def get_alibi_bias(self, seq_len, device):
        if seq_len not in self._alibi_bias_cache:
            # Precompute ALiBi bias matrix: shape [num_heads, seq_len, seq_len]
            relative_pos = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
            relative_pos = relative_pos.abs().unsqueeze(0).expand(self.num_heads, -1, -1) # [num_heads, seq_len, seq_len]

            slopes = self.slopes.to(device).view(self.num_heads, 1, 1)
            alibi_bias = slopes * relative_pos * -1 # Negative sign as per ALiBi paper
            self._alibi_bias_cache[seq_len] = alibi_bias.detach() # Detach as it's not learned

        return self._alibi_bias_cache[seq_len]


    def forward(self, x, attention_mask=None):
        # x shape: [batch_size, seq_len, hidden_dim]
        # attention_mask: [batch_size, seq_len] (optional, for padding)
        batch_size, seq_len, _ = x.shape

        # 1. Project Q, K, V
        qkv = self.qkv_proj(x).chunk(3, dim=-1) # (q, k, v) each [batch, seq_len, hidden_dim]
        q, k, v = map(lambda t: t.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2), qkv)
        # Now shape [batch_size, num_heads, seq_len, head_dim]

        # 2. Calculate Attention Scores with ALiBi bias
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale # [batch, num_heads, seq_len, seq_len]

        # Add ALiBi bias
        alibi_bias = self.get_alibi_bias(seq_len, x.device) # [num_heads, seq_len, seq_len]
        attn_scores = attn_scores + alibi_bias.unsqueeze(0) # Add broadcasted bias

        # Apply padding mask (if provided)
        if attention_mask is not None:
            # Expand mask: [batch, 1, 1, seq_len]
            mask = attention_mask.unsqueeze(1).unsqueeze(2).to(torch.bool)
            # Mask needs to be compatible with attn_scores [batch, num_heads, seq_len, seq_len]
            # Where mask value is True, set score to -inf
            attn_scores = attn_scores.masked_fill(~mask, float('-inf')) # Assuming True means "keep", False means "mask"

        # 3. Apply Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1) # [batch, num_heads, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)

        # 4. Weighted Sum (Attend to Values)
        attn_output = torch.matmul(attn_weights, v) # [batch, num_heads, seq_len, head_dim]

        # 5. Reshape and Project Output
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1) # [batch, seq_len, hidden_dim]
        output = self.out_proj(attn_output)

        return output


class TransformerLayer(nn.Module):
    """ A standard Transformer Encoder Layer with optional ALiBi """
    def __init__(self, hidden_dim, num_heads, ffn_dim=None, dropout_rate=0.1, use_alibi=True):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = hidden_dim * 4

        self.use_alibi = use_alibi
        if use_alibi:
            self.self_attn = AlibiAttention(hidden_dim, num_heads, dropout=dropout_rate)
        else:
            # Use standard MHA if ALiBi is disabled (requires position embeddings externally)
            self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout_rate, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout_rate)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attention_mask=None):
        # x shape: [batch_size, seq_len, hidden_dim]
        # attention_mask: [batch_size, seq_len] (for ALiBiAttention) or [batch_size, seq_len, seq_len] (for nn.MHA)

        # Self-Attention block
        residual = x
        x = self.norm1(x)
        if self.use_alibi:
            attn_output = self.self_attn(x, attention_mask=attention_mask) # Pass mask if needed
        else:
             # Standard MHA needs key_padding_mask
             # Convert seq_len mask to key_padding_mask (True where padded)
             key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
             attn_output, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)

        x = residual + self.dropout(attn_output)

        # Feed-Forward block
        residual = x
        x = self.norm2(x)
        ffn_output = self.ffn(x)
        x = residual + self.dropout(ffn_output)

        return x


class TextEncoderTransformer(nn.Module):
    def __init__(self,
                 hidden_dim=768,
                 num_layers=12,
                 num_heads=12,
                 vocab_size=32000,
                 max_seq_len=512, # Needed if NOT using ALiBi
                 alibi_enabled=True,
                 dropout_rate=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.alibi_enabled = alibi_enabled

        if not alibi_enabled:
            # Use learned positional embeddings if ALiBi is off
            self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
            print("Using Learned Positional Embeddings for Text Encoder.")
        else:
            print("Using ALiBi for Text Encoder (No explicit Position Embeddings).")


        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, num_heads, ffn_dim=hidden_dim*4, dropout_rate=dropout_rate, use_alibi=alibi_enabled)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim) # Final normalization

    def forward(self, input_ids, attention_mask=None):
        # input_ids: [batch_size, seq_len]
        # attention_mask: [batch_size, seq_len] (1 for real tokens, 0 for padding)

        seq_len = input_ids.shape[1]
        x = self.embedding(input_ids) # [batch_size, seq_len, hidden_dim]

        if not self.alibi_enabled:
            positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0) # [1, seq_len]
            x = x + self.position_embedding(positions) # Add positional embeddings

        # Apply Transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask) # Pass mask

        x = self.norm(x) # Final layer norm
        return x # [batch_size, seq_len, hidden_dim]