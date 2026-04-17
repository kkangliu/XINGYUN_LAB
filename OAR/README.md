<div align="center">

# 🎯 Outcome-Grounded Advantage Reshaping (OAR)

**Fine-Grained Credit Assignment for Mathematical Reasoning**

[![ACL 2026](https://img.shields.io/badge/ACL-2026-blue.svg)](https://arxiv.org/abs/2601.07408)
[![arXiv](https://img.shields.io/badge/arXiv-2601.07408-b31b1b.svg)](https://arxiv.org/abs/2601.07408)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

[📄 Paper](https://arxiv.org/abs/2601.07408) | [🏠 XINGYUN LAB](https://github.com/XINGYUN-AI-LAB) | [💬 Issues](https://github.com/XINGYUN-AI-LAB/XINGYUN_LAB/issues)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Method](#-method)
  - [OAR-P: Perturbation-based](#oar-p-perturbation-based)
  - [OAR-G: Gradient-based](#oar-g-gradient-based)
  - [Dual-Tier Advantage Reshaping](#dual-tier-advantage-reshaping)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Repository Structure](#-repository-structure)
- [Results](#-results)
- [Citation](#-citation)
- [Contact](#-contact)

---

## 🌟 Overview

**OAR** addresses a critical limitation in **Group Relative Policy Optimization (GRPO)** for reasoning tasks: its **coarse-grained credit assignment**. GRPO propagates group-level rewards uniformly to every token in a sequence, overlooking the varying contributions of individual steps in multi-step reasoning.

### The Problem
In mathematical reasoning tasks, not all tokens contribute equally:
- Some tokens are **pivotal** (key reasoning steps)
- Others are **low-impact** (connecting words, formatting)
- GRPO treats them all the same ❌

### Our Solution
OAR reallocates advantage based on **outcome-sensitivity**:
- **OAR-P**: Uses counterfactual token perturbations (performance ceiling ⭐)
- **OAR-G**: Employs input-gradient sensitivity (negligible overhead ⚡)
- **Conservative reshaping**: Suppresses noise, boosts pivotal tokens 🎯

---

## ✨ Key Features

- 🎯 **Fine-grained credit assignment** for token-level optimization
- 🚀 **Two strategies**:
  - **OAR-P**: High accuracy via perturbation analysis
  - **OAR-G**: Efficient gradient-based approximation
- 📊 **Significant gains** over GRPO baselines on mathematical reasoning
- 🔧 **Easy integration** with existing RL training pipelines
- ⚡ **Negligible overhead** with OAR-G variant

---

## 🧠 Method

### Overview Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Sequence                            │
│  [Context] → [Reasoning Steps] → [Conclusion] → [Answer]   │
└─────────────────────────────────────────────────────────────┘
                           ↓
        ┌──────────────────────────────────────┐
        │   Importance Scoring Module          │
        │  ┌──────────────┐  ┌──────────────┐ │
        │  │   OAR-P      │  │   OAR-G      │ │
        │  │ Perturbation │  │  Gradient    │ │
        │  │   Analysis   │  │  Sensitivity │ │
        │  └──────────────┘  └──────────────┘ │
        └──────────────────────────────────────┘
                           ↓
        ┌──────────────────────────────────────┐
        │  Dual-Tier Advantage Reshaping       │
        │  • Noise Suppression (< threshold)   │
        │  • Signal Boosting (≥ threshold)     │
        └──────────────────────────────────────┘
                           ↓
        ┌──────────────────────────────────────┐
        │   Token-Level Weighted Advantages    │
        │   Applied to Policy Gradient         │
        └──────────────────────────────────────┘
```

### OAR-P: Perturbation-based

**OAR-P** estimates token importance through **counterfactual analysis**:

1. **Mask each token** in the reasoning sequence
2. **Measure the impact** on final answer distribution
3. **Quantify importance** via KL divergence:

```
I(t) = KL(P_original || P_masked_t)
```

**Pros**: High accuracy, performance ceiling
**Cons**: Computational overhead (multiple forward passes)

### OAR-G: Gradient-based

**OAR-G** uses **input-gradient sensitivity** as an efficient proxy:

1. Add **Gaussian noise** to reasoning token embeddings
2. Compute **KL divergence** w.r.t. clean output
3. **Backpropagate** to get token-level gradients:

```
Saliency(t) = |∇_{e_t} KL(P_clean || P_noisy)|
```

**Pros**: Negligible overhead (~1 backward pass)
**Cons**: Approximation of true importance

### Dual-Tier Advantage Reshaping

Conservative reshaping scheme balances stability and sensitivity:

```python
# Normalize importance scores
norm_importance = (scores - min) / (max - min)

# Noise suppression (below threshold)
if norm_importance[i] < threshold:
    weight[i] = max(norm_importance[i] / threshold, 0.1)

# Signal boosting (above threshold)
else:
    relative_pos = (norm_importance[i] - threshold) / (1.0 - threshold)
    weight[i] = 1.0 + beta * relative_pos
```

**Hyperparameters**:
- `threshold`: Percentile for noise/signal separation (default: 0.25)
- `beta`: Boosting coefficient for high-impact tokens (default: 1.0)

---

## 🛠️ Installation

### Prerequisites
- Python >= 3.8
- PyTorch >= 2.0
- CUDA (recommended for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/XINGYUN-AI-LAB/XINGYUN_LAB.git
cd XINGYUN_LAB/OAR

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install numpy scipy
```

---

## 🚀 Quick Start

### Basic Usage

```python
from oar import calculate_token_importance_gradient, calculate_oar_hybrid_weights

# 1. Calculate token importance (OAR-G)
raw_scores = calculate_token_importance_gradient(
    model=model,
    input_segments=input_segments,
    device=device,
    pfinal_mode="answer_mean",  # or "last", "answer_joint"
    noise_std=1e-3
)

# 2. Apply dual-tier reshaping
weights = calculate_oar_hybrid_weights(
    inputs=inputs,
    model=model,
    threshold=0.25,
    beta=1.0
)

# 3. Weight advantages for policy gradient
weighted_advantages = advantages * weights
```

### Training Example

```python
# Standard GRPO training
loss = -min(
    ratio * advantages,
    clip(ratio, 1-eps, 1+eps) * advantages
)

# With OAR enhancement
weights = calculate_oar_hybrid_weights(inputs, model)
weighted_advantages = advantages * weights
loss = -min(
    ratio * weighted_advantages,
    clip(ratio, 1-eps, 1+eps) * weighted_advantages
)
```

### Configuration

```python
# OAR-G (recommended for efficiency)
args.method = 'gradient'
args.pfinal_mode = 'answer_mean'
args.threshold = 0.25
args.beta = 1.0

# OAR-P (for maximum accuracy)
args.method = 'perturbation'
args.pfinal_mode = 'last'
args.threshold = 0.25
args.beta = 1.0
```

---

## 📁 Repository Structure

```
OAR/
├── main.py                          # Core OAR implementation
│   ├── calculate_token_importance_perturbation()  # OAR-P
│   ├── calculate_token_importance_gradient()      # OAR-G
│   ├── calculate_oar_hybrid_weights()             # Reshaping logic
│   └── compute_loss()                             # Integration with GRPO
└── README.md                        # This file
```

---

## 📊 Results

### Performance on Mathematical Reasoning Benchmarks

| Method | GSM8K | MATH | Average |
|--------|-------|------|---------|
| GRPO (baseline) | 75.2 | 42.8 | 59.0 |
| **OAR-P** | **79.6** | **47.3** | **63.5** |
| **OAR-G** | **78.9** | **46.8** | **62.9** |

**Key Findings**:
- ✅ **OAR-P** sets performance ceiling (+4.4% avg improvement)
- ✅ **OAR-G** achieves comparable gains with negligible overhead (+3.9%)
- ✅ Both significantly outperform GRPO baselines

### Efficiency Comparison

| Method | Forward Passes | Backward Passes | Relative Time |
|--------|----------------|-----------------|---------------|
| GRPO | 1 | 1 | 1.0x |
| OAR-P | N+1 (N=tokens) | 1 | ~5.0x |
| **OAR-G** | 1 | 1 | **~1.1x** ⚡ |

---

## 📝 Citation

If you find our work useful, please cite:

```bibtex
@inproceedings{li2026outcome,
  title={Outcome-Grounded Advantage Reshaping for Fine-Grained Credit Assignment in Mathematical Reasoning},
  author={Li, Ziheng and Kang, Liu and Xiao, Feng and Xing, Luxi and Si, Qingyi and Li, Zhuoran and Gong, Weikang and Yang, Deqing and Xiao, Yanghua and Guo, Hongcheng},
  booktitle={Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2026}
}
```

---

## 📧 Contact

For questions or issues:
- 📮 Open an [issue](https://github.com/XINGYUN-AI-LAB/XINGYUN_LAB/issues)
- 📧 Contact the corresponding author listed in the paper

---

<div align="center">

**[XINGYUN AI LAB](https://github.com/XINGYUN-AI-LAB)** - AI and NLP Research Laboratory

*Last Updated: April 2026*

</div>
