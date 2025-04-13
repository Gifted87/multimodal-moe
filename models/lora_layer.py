import torch
import torch.nn as nn
import math

# Simple LoRA implementation for Linear layers
class LoRALinear(nn.Module):
    def __init__(self, linear_layer: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.rank = rank
        self.alpha = alpha

        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Freeze original weights
        self.weight = linear_layer.weight
        self.weight.requires_grad = False
        if linear_layer.bias is not None:
            self.bias = linear_layer.bias
            self.bias.requires_grad = False
        else:
            self.bias = None

        self.scaling = self.alpha / self.rank
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize A with Kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Original output
        frozen_output = nn.functional.linear(x, self.weight, self.bias)

        # LoRA adaptation
        lora_output = (x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling

        return frozen_output + lora_output

    def merge(self):
        """ Merges LoRA weights into the original weights. """
        if not self.merged:
            delta_w = self.lora_B @ self.lora_A * self.scaling
            self.weight.data += delta_w
            self.merged = True # Prevent merging multiple times

    # Utility to easily replace linear layers
    @staticmethod
    def apply_to_model(model, rank, alpha=1.0, layers_to_adapt=["query", "key", "value", "dense", "fc1", "fc2"]):
        """ Recursively replaces Linear layers with LoRALinear """
        for name, module in model.named_children():
            if isinstance(module, nn.Linear):
                # Check if the layer name contains keywords for adaptation
                if any(layer_name in name for layer_name in layers_to_adapt):
                    print(f"Applying LoRA to: {name}")
                    setattr(model, name, LoRALinear(module, rank, alpha))
            else:
                LoRALinear.apply_to_model(module, rank, alpha, layers_to_adapt)