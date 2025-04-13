import torch
import unittest
from omegaconf import OmegaConf

from models.moe_layer import MultimodalMoE, SimpleNoisyTopKRouter, Expert

class TestMoESparsity(unittest.TestCase):

    def setUp(self):
        # Minimal config for testing MoE layer
        self.config = OmegaConf.create({
            "model": {
                "hidden_dim": 64,
                "moe": {
                    "num_experts": 8,
                    "top_k": 2,
                    "gate_type": "noisy_topk",
                    "noise_epsilon": 0.01, # Keep low for testing stability
                    "capacity_factor": 1.0, # Use 1.0 for easier capacity checks
                    "aux_loss_factor": 0.01
                }
            }
        })
        self.hidden_dim = self.config.model.hidden_dim
        self.num_experts = self.config.model.moe.num_experts
        self.top_k = self.config.model.moe.top_k

        # Instantiate MoE layer
        self.moe_layer = MultimodalMoE(
             hidden_dim=self.hidden_dim,
             num_experts=self.num_experts,
             top_k=self.top_k,
             gate_type=self.config.model.moe.gate_type,
             noise_epsilon=self.config.model.moe.noise_epsilon,
             capacity_factor=self.config.model.moe.capacity_factor,
             aux_loss_factor=self.config.model.moe.aux_loss_factor
        ).cuda() # Test on GPU if available

        self.device = next(self.moe_layer.parameters()).device

    def test_output_shape(self):
        """ Test if the output shape is correct """
        batch_size = 4
        seq_len = 10
        dummy_input = torch.randn(batch_size, seq_len, self.hidden_dim, device=self.device)
        output = self.moe_layer(dummy_input)
        self.assertEqual(output.shape, dummy_input.shape)

    def test_expert_routing_indices(self):
        """ Test if the gate returns the correct number of expert indices """
        num_tokens = 50
        dummy_tokens = torch.randn(num_tokens, self.hidden_dim, device=self.device)

        # Access the gate directly to check its output
        # Note: MOELayer might encapsulate this, adjust if needed based on Fairscale version
        if isinstance(self.moe_layer.gate, SimpleNoisyTopKRouter):
             weights, selected_experts = self.moe_layer.gate(dummy_tokens)
             self.assertEqual(selected_experts.shape, (num_tokens, self.top_k))
             # Check indices are within range
             self.assertTrue(torch.all(selected_experts >= 0))
             self.assertTrue(torch.all(selected_experts < self.num_experts))
        else:
             # If using Fairscale's gate directly within MOELayer, this test needs adaptation
             # to potentially inspect internal state or rely on aux loss behavior.
             print("Skipping direct gate output test for non-SimpleNoisyTopKRouter.")


    def test_aux_loss_calculation(self):
         """ Test if auxiliary loss is calculated and has the right shape """
         batch_size = 4
         seq_len = 10
         dummy_input = torch.randn(batch_size, seq_len, self.hidden_dim, device=self.device)
         _ = self.moe_layer(dummy_input) # Forward pass to trigger loss calculation
         aux_loss = self.moe_layer.get_aux_loss()

         self.assertIsNotNone(aux_loss)
         self.assertTrue(isinstance(aux_loss, torch.Tensor))
         self.assertEqual(aux_loss.shape, torch.Size([])) # Should be a scalar
         self.assertTrue(aux_loss.item() >= 0) # Loss should be non-negative

    # Add more tests:
    # - Test load balancing (harder, might need to check expert usage stats over many batches)
    # - Test capacity factor enforcement (check if tokens are dropped if capacity exceeded)
    # - Test behavior in eval mode (e.g., noise disabled)

if __name__ == '__main__':
    unittest.main()