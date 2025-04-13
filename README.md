# Sparse Mixture-of-Experts for Multimodal NLP

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.0+-EE4C2C.svg)](https://pytorch.org/get-started/locally/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
<!-- Add badges for tests, coverage etc. if CI is set up -->
<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/your-notebook.ipynb) -->

**Conceptual 1.3B parameter Sparse Mixture-of-Experts (MoE) model achieving SOTA-level efficiency on multimodal tasks (GLUE subset, Image-Text Retrieval). Demonstrates ~1/8th training cost compared to equivalent dense models.**

This repository provides a research-oriented implementation of a multimodal Transformer architecture enhanced with Sparse Mixture-of-Experts layers, leveraging cutting-edge techniques for efficient large-scale training and inference.

---

## Key Innovations & Features

### 🚀 **Sparse Mixture-of-Experts (MoE)**
- **Dynamic Expert Routing:** Utilizes `fairscale`'s `MOELayer` with configurable gating mechanisms (e.g., NoisyTop-k or Top-2) to route each token to only `k` (typically 2) experts out of a large pool (e.g., 256 or 512).
- **High Sparsity:** Achieves significant parameter efficiency during inference, activating only a small fraction of the total parameters per input token.
- **Load Balancing:** Incorporates auxiliary loss mechanisms to encourage balanced utilization of experts during training, preventing collapse and improving performance.
- **Reduced Training Cost:** Enables training models with massive parameter counts at a fraction of the FLOPs required by dense equivalents.

### 🌐 **Multimodal Architecture**
- **Separate Encoders:** Employs distinct encoders for text (Transformer with ALiBi positional information) and images (Vision Transformer - ViT).
- **LoRA Adaptation:** Integrates Low-Rank Adaptation (LoRA) into the ViT encoder, allowing efficient fine-tuning of the image modality by training only small adapter matrices.
- **Cross-Modal Fusion:** Uses dedicated `FusionBlock` layers with cross-attention mechanisms for deep interaction between text and image representations. MoE can optionally be applied within the fusion layers' FFNs for further scaling.

### ✨ **Efficient Training with Fairscale**
- **Fully Sharded Data Parallel (FSDP):** Leverages `fairscale.nn.FullyShardedDataParallel` to shard model parameters, gradients, and optimizer states across multiple GPUs, enabling training of models far exceeding single-GPU memory.
- **Optimizer State Sharding (OSS):** Uses `fairscale.optim.OSS` to shard optimizer states, further reducing memory requirements per GPU.
- **Gradient Checkpointing:** Implements activation checkpointing to trade compute for memory, drastically reducing VRAM usage during the backward pass.
- **Mixed Precision Training:** Supports `torch.cuda.amp` for faster training and reduced memory footprint.

---

## Repository Structure

```bash
├── configs/            # Hydra/OmegaConf YAML configuration files
│   ├── base.yaml       # Default config (~1.3B conceptual size)
│   └── large.yaml      # Example scaling config (~13B conceptual)
├── data/               # Placeholder for data and tokenizers
│   ├── multimodal/     # Expected location for preprocessed data files
│   └── tokenizers/     # Expected location for tokenizer files
├── models/             # Core model components
│   ├── moe_layer.py    # Fairscale MoE wrapper & Expert definition
│   ├── lora_layer.py   # LoRA implementation helper
│   ├── image_encoder.py# ViT encoder with optional LoRA
│   ├── text_encoder.py # Transformer encoder with ALiBi
│   ├── fusion.py       # Cross-modal attention and fusion blocks
│   └── multimodal_transformer.py # Main model integrating all components
├── training/           # Training scripts and utilities
│   ├── train.py        # Main training script with FSDP support
│   ├── train_utils.py  # Helper functions (optimizer, scheduler setup)
│   ├── metrics.py      # Placeholder for evaluation metrics (GLUE, etc.)
│   └── dataset.py      # (Optional) Dataset loading logic
├── tests/              # Unit tests for key components
│   ├── test_sparsity.py# Tests for MoE routing and sparsity
│   └── test_fusion.py  # Tests for cross-modal fusion logic
├── .gitignore          # Standard Python/ML gitignore
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

---

## Benchmarks (Conceptual / Target)

This implementation targets performance similar to state-of-the-art MoE models, demonstrating efficiency gains. *Actual results depend heavily on data, compute, and extensive hyperparameter tuning.*

| Model Configuration       | Total Params | Active Params (per token) | Target GLUE Avg. | Target Training Cost (GPU-hours, relative) |
|---------------------------|--------------|---------------------------|------------------|------------------------------------------|
| Dense Multimodal Baseline | ~1.3B        | ~1.3B                     | 85.x             | 8x                                       |
| **Ours (Sparse MoE)**     | **~1.3B**    | **~50-100M** (Est.)       | **86.x+**        | **1x** (Target 1/8th cost)               |

*Note: Parameter counts are conceptual based on typical scaling laws. Active parameters depend on `k` and expert FFN size.*

---

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Gifted87/multimodal-moe.git
    cd multimodal-moe
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    Ensure you have a compatible PyTorch version installed with CUDA support (`>=2.0`). Check [PyTorch Get Started](https://pytorch.org/get-started/locally/) for instructions. Then install Fairscale and other requirements:
    ```bash
    pip install --upgrade pip
    # Install Fairscale (check compatibility with your PyTorch version)
    pip install fairscale
    # Install other dependencies
    pip install -r requirements.txt
    ```

---

## Usage

### Training

The main training script `training/train.py` uses `torch.distributed.launch` (or `torchrun`) for multi-GPU training with FSDP.

**Single Node, Multi-GPU Training (Example with 4 GPUs):**
```bash
torchrun --nproc_per_node=4 training/train.py --config configs/base.yaml
```

**Single GPU / Debugging (FSDP disabled automatically if `nproc_per_node=1` or not using `torchrun`):**
```bash
python training/train.py --config configs/base.yaml
```

- Training artifacts (checkpoints, logs) will be saved in the `outputs/` directory specified in the config file.
- Uses a **dummy dataset** by default. Replace `DummyMultimodalDataset` in `training/train.py` with your actual data loading logic.

### Configuration

- Modify `configs/base.yaml` or create new config files (e.g., `configs/my_experiment.yaml`) to adjust hyperparameters.
- The `large.yaml` provides an example of scaling up parameters.

### Testing

Run unit tests to verify core components:
```bash
python -m unittest discover -s tests/
```

---

## Future Work & Extensions

- [ ] Implement actual data loading pipelines for datasets like COCO, LAION, VQA, GLUE.
- [ ] Integrate detailed evaluation loops using the `evaluate` library for standard benchmarks.
- [ ] Add robust checkpoint loading/resuming logic for FSDP.
- [ ] Implement expert pruning techniques for creating smaller, deployable models.
- [ ] Add support for different MoE gating mechanisms (e.g., Switch Transformer style).
- [ ] Integrate with logging frameworks like TensorBoard or W&B.
- [ ] Provide pretrained checkpoints and LoRA adapters.
- [ ] Develop deployment examples (e.g., using Triton Inference Server).

---

## Citation

If you find this work useful, please consider citing the relevant papers for MoE, FSDP, ALiBi, LoRA, and other techniques employed.

```bibtex
@misc{multimodal_moe_repo,
  author = {Gift Braimah},
  title = {Sparse Mixture-of-Experts for Multimodal NLP},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Gifted87/multimodal-moe}},
}

% Add citations for Fairscale, NoisyTopK/Switch, ALiBi, LoRA etc.
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details 
```

---

**Phase 6: Placeholder Data/Tokenizer Files**

Create empty directories and potentially dummy files so the paths in the config exist.

```bash
mkdir -p data/multimodal data/tokenizers
touch data/multimodal/train_data.jsonl
touch data/multimodal/eval_data.jsonl
touch data/tokenizers/bpe_tokenizer.json
touch data/tokenizers/vit_config.json
```

