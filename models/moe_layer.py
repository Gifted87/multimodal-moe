import torch
import torch.nn as nn
from fairscale.nn.moe import MOELayer, Top2Gate # Using Top2Gate directly if k=2 is fixed, or Router for flexibility
from fairscale.nn.misc import TopKGate # Base for other gates if needed
# Let's stick to the user's NoisyTopkRouter reference if available or implement its logic
# Assuming NoisyTopkRouter exists in Fairscale or we implement a similar concept:
# If fairscale.nn.gate.NoisyTopkRouter isn't standard, we might need to implement it
# or use Top2Gate/TopKGate and add noise manually during routing if essential.
# For simplicity here, let's assume Top2Gate covers the k=2 case efficiently.
# If NoisyTopK specifically needed, implementation would involve adding noise before top-k selection.

# Placeholder for NoisyTopkRouter if not directly in fairscale
# Concept: Add Gaussian noise to logits before applying top-k
class SimpleNoisyTopKRouter(nn.Module):
    def __init__(self, d_model, num_experts, k=2, noise_epsilon=1e-2, capacity_factor=1.2):
        super().__init__()
        self.k = k
        self.num_experts = num_experts
        self.epsilon = noise_epsilon
        self.capacity_factor = capacity_factor # Note: capacity factor is usually handled by MOELayer itself
        self.gate_linear = nn.Linear(d_model, num_experts)

    def forward(self, x):
        # x shape: [batch_size * seq_len, d_model] or [tokens, d_model]
        logits = self.gate_linear(x)

        if self.training and self.epsilon > 0:
            noise = torch.randn_like(logits) * self.epsilon
            noisy_logits = logits + noise
        else:
            noisy_logits = logits

        raw_weights = torch.softmax(noisy_logits, dim=-1)
        top_k_weights, top_k_indices = torch.topk(raw_weights, self.k, dim=-1)

        # Normalize top-k weights
        normalized_weights = top_k_weights / torch.sum(top_k_weights, dim=-1, keepdim=True)

        return normalized_weights, top_k_indices # Shape: [tokens, k], [tokens, k]


# Basic FeedForward Expert
class Expert(nn.Module):
    """ A simple FeedForward Network expert """
    def __init__(self, hidden_dim, ffn_dim=None, dropout_rate=0.1):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = hidden_dim * 4 # Standard FFN expansion
        self.fc1 = nn.Linear(hidden_dim, ffn_dim)
        self.act = nn.GELU() # Or ReLU, SiLU etc.
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(ffn_dim, hidden_dim)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class MultimodalMoE(nn.Module):
    """
    Mixture of Experts Layer using Fairscale MOELayer.
    Routes tokens to a subset of experts.
    """
    def __init__(self,
                 hidden_dim,
                 num_experts=256,
                 top_k=2,
                 gate_type="noisy_topk", # Or 'top2' etc.
                 noise_epsilon=1e-2,
                 capacity_factor=1.2,
                 aux_loss_factor=0.01,
                 expert_class=Expert,
                 **expert_kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.k = top_k
        self.aux_loss_factor = aux_loss_factor

        if gate_type == "noisy_topk":
            # Use placeholder or actual if available
            self.gate = SimpleNoisyTopKRouter(hidden_dim, num_experts, top_k, noise_epsilon, capacity_factor)
        elif gate_type == "top2" and top_k == 2:
             # Fairscale's optimized Top2Gate
             self.gate = Top2Gate(hidden_dim, num_experts, capacity_factor=capacity_factor)
        else:
             # Fallback to generic TopKGate or raise error
             print(f"Warning: Using generic TopKGate for k={top_k}. NoisyTopkRouter preferred.")
             self.gate = TopKGate(hidden_dim, num_experts, k=top_k, capacity_factor=capacity_factor)


        self.experts = nn.ModuleList([
            expert_class(hidden_dim=hidden_dim, **expert_kwargs) for _ in range(num_experts)
        ])

        # MOELayer handles the expert dispatch and combines results
        # It also calculates the auxiliary loss internally
        self.moe_layer = MOELayer(
            gate=self.gate,
            experts=self.experts,
            model_dim=hidden_dim,
            num_local_experts=num_experts # Assuming non-distributed experts for simplicity here; FSDP handles model sharding
        )
        self.aux_loss = None # To store aux loss calculated by MOELayer

    def forward(self, x):
        # x shape: [batch, seq_len, hidden_dim]
        original_shape = x.shape
        # MOELayer expects input shape [num_tokens, model_dim]
        x = x.reshape(-1, self.hidden_dim)

        # Fairscale's MOELayer returns (output, aux_loss)
        output, aux_loss = self.moe_layer(x)
        self.aux_loss = aux_loss * self.aux_loss_factor # Store scaled loss

        # Reshape back to original
        return output.reshape(original_shape)

    def get_aux_loss(self):
      """ Retrieve the auxiliary loss for the last forward pass. """
      # Make sure loss is on the correct device and requires grad if needed downstream
      # The loss is usually detached within MOELayer for standard usage.
      return self.aux_loss if self.aux_loss is not None else torch.tensor(0.0, device=self.gate.gate_linear.weight.device)