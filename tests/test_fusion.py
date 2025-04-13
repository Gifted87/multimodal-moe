import torch
import unittest
from omegaconf import OmegaConf

from models.fusion import FusionBlock
from models.multimodal_transformer import MultimodalTransformer # To test integrated model

class TestFusion(unittest.TestCase):

    def setUp(self):
        self.config = OmegaConf.create({
            "model": {
                "hidden_dim": 64,
                "num_heads": 4,
                "dropout_rate": 0.0, # Disable dropout for deterministic tests
                "fusion": {
                    "num_layers": 1, # Test a single fusion block
                    "use_moe_in_fusion": False # Test without MoE first
                },
                 # Add dummy encoder/MoE configs if needed by MultimodalTransformer init
                 "text_encoder": {"num_layers": 1, "vocab_size": 100, "max_seq_len": 16, "alibi_enabled": True},
                 "image_encoder": {"num_layers": 1, "patch_size": 8, "img_size": 32, "use_lora": False},
                 "moe": {"num_experts": 4, "top_k": 2}
            },
            "num_classes": 3 # Add dummy num_classes
        })
        self.hidden_dim = self.config.model.hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Test standalone FusionBlock
        self.fusion_block = FusionBlock(
            hidden_dim=self.hidden_dim,
            num_heads=self.config.model.num_heads,
            use_moe_ffn=self.config.model.fusion.use_moe_in_fusion,
            # moe_config=self.config.model.moe, # Provide if use_moe_ffn=True
            dropout_rate=self.config.model.dropout_rate
        ).to(self.device)

        # Test full model integration
        self.full_model = MultimodalTransformer(self.config).to(self.device)


    def test_fusion_block_shapes(self):
        """ Test output shapes of the FusionBlock """
        batch_size = 4
        seq_len_t = 10
        seq_len_i = 12 # e.g., 11 patches + 1 CLS token
        text_feat = torch.randn(batch_size, seq_len_t, self.hidden_dim, device=self.device)
        img_feat = torch.randn(batch_size, seq_len_i, self.hidden_dim, device=self.device)

        fused_text, fused_image = self.fusion_block(text_feat, img_feat)

        self.assertEqual(fused_text.shape, text_feat.shape)
        self.assertEqual(fused_image.shape, img_feat.shape)

    def test_full_model_forward_pass(self):
        """ Test if the full model can perform a forward pass """
        batch_size = 2
        seq_len_t = self.config.model.text_encoder.max_seq_len
        img_size = self.config.model.image_encoder.img_size
        vocab_size = self.config.model.text_encoder.vocab_size

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len_t), device=self.device)
        attention_mask = torch.ones(batch_size, seq_len_t, device=self.device)
        pixel_values = torch.randn(batch_size, 3, img_size, img_size, device=self.device)
        labels = torch.randint(0, self.config.num_classes, (batch_size,), device=self.device)

        # Run forward pass requesting loss
        try:
             outputs = self.full_model(
                 input_ids=input_ids,
                 attention_mask=attention_mask,
                 pixel_values=pixel_values,
                 labels=labels,
                 return_loss=True
             )
             self.assertIn("loss", outputs)
             self.assertIn("logits", outputs)
             self.assertIsNotNone(outputs["loss"])
             self.assertEqual(outputs["logits"].shape, (batch_size, self.config.num_classes))

        except Exception as e:
             self.fail(f"Full model forward pass failed with exception: {e}")

    # Add tests for fusion with MoE enabled
    # Add tests for attention masks

if __name__ == '__main__':
    unittest.main()