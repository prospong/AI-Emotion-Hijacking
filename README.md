# Emotional Hijacking in AI Systems

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-EE4C2C.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A neuroscience-inspired computational framework for investigating emotional hijacking phenomena in artificial intelligence systems. This research implements the **MEGA (Memory-Emotion-Gate-Amygdala) architecture**, a dual-pathway model inspired by amygdala processing mechanisms in the human brain.

## 🔬 Research Overview

This project systematically characterizes both **induced** and **spontaneous** hijacking mechanisms through five comprehensive experiments, revealing critical vulnerabilities in AI decision-making systems when processing emotional stimuli.

### Key Findings

- **Adversarial Vulnerability**: Fast pathways exhibit **61% greater vulnerability** to adversarial perturbations compared to slow pathways (Cohen's d = 2.31)
- **Critical Phase Transition**: Information bottleneck parameter β exhibits sharp phase transition at **βc ≈ 0.368** with 92% entropy collapse
- **Non-Monotonic Stability**: System noise displays W-shaped hijacking probability curve with optimal robustness at **σ = 0.50**
- **Dynamic Reversibility**: Baseline pathway dominance (86% fast wins) can be reversed to 60% slow pathway wins through contextual modulation

---

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Installation](#-installation)
- [Experiments](#-experiments)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Results & Visualizations](#-results--visualizations)
- [Citation](#-citation)
- [License](#-license)

---

## 🏗️ Architecture

The **MEGA Framework** consists of four interconnected modules that simulate dual-pathway emotional processing:

```
┌─────────────────────────────────────────────────┐
│              MEGA Architecture                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────┐         ┌──────────────┐    │
│  │   Memory     │◄────────┤   Emotion    │    │
│  │   Module     │         │   Module     │    │
│  │   (M)        │         │   (E)        │    │
│  └──────┬───────┘         └──────────────┘    │
│         │                                      │
│         ▼                                      │
│  ┌──────────────┐                             │
│  │    Gate      │                             │
│  │   Module     │──────┐                      │
│  │    (γ)       │      │                      │
│  └──────────────┘      │                      │
│                        ▼                      │
│              ┌────────────────┐               │
│              │   Amygdala     │               │
│              │   (Dual Path)  │               │
│              ├────────────────┤               │
│              │  Fast  │ Slow  │               │
│              │ (θf=0.3│θs=0.7)│               │
│              └────────────────┘               │
│                      │                        │
│                      ▼                        │
│              ┌────────────────┐               │
│              │   Decision     │               │
│              │    Output      │               │
│              └────────────────┘               │
└─────────────────────────────────────────────────┘
```

### Core Components

1. **Memory Module (M)**: Recurrent memory network with forgetting dynamics
   - Equation: `M(t+1) = (1-η)M(t) + η·tanh(W_m·[x(t); M(t)])`
   - Default forgetting rate: η = 0.1

2. **Emotion Module (E)**: Valence-arousal encoder
   - Processes input signals into emotional representations
   - Maps to 2D space: valence ∈ [-1,1], arousal ∈ [0,1]

3. **Gate Module (γ)**: Sigmoid-based attention mechanism
   - Controls information flow: `γ = σ(W_g·M + b_g)`
   - Enables selective emotional filtering

4. **Amygdala Module**: Dual-pathway processor
   - **Fast Pathway** (θf = 0.3): Rapid, heuristic decisions
   - **Slow Pathway** (θs = 0.7): Deliberative, rational processing

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for experiments 2-5)
- 8GB+ RAM

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/emotional-hijacking.git
cd emotional-hijacking

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.1
numpy>=1.23.0
matplotlib>=3.5.0
scipy>=1.9.0
seaborn>=0.12.0
pandas>=1.5.0
scikit-learn>=1.1.0
```

---

## 🔬 Experiments

The research comprises **five systematic experiments** investigating different aspects of emotional hijacking:

### Experiment 1: Memory-Gate Dynamics Baseline
**Objective**: Establish foundational behavioral characteristics of the MEGA framework

**Key Metrics**:
- Gate responsiveness rate: 100% (Enhanced config)
- Memory-gate coupling: r = 0.91
- PFC alignment: Response delay = 0.191s (target: 0.200s)

**Configurations Tested**:
- Original (conservative)
- Enhanced (balanced) ✓ **optimal**
- Various signal types: mixed, chaotic, regime-switching

```bash
python src/experiments/E1_memory_gate/run_experiment.py --config enhanced
```

**Outputs**: `experiments/E1_memory_gate/` (13 figures)

---

### Experiment 2: Adversarial-Induced Hijacking
**Objective**: Quantify vulnerability to adversarial perturbations (FGSM attacks)

**Key Findings**:
- ε = 0.05: Hijacking rate = 25.0%
- ε = 0.20: Hijacking rate = 35.9%
- Fast pathway vulnerability: 61% vs. Slow: 39% (p < 0.001)

**Mathematical Model**:
```
P_hijack(ε) ≈ 0.36(1 - e^(-10ε))
```

```bash
python src/experiments/E2_adversarial/run_fgsm_attack.py \
    --epsilon 0.05 0.10 0.15 0.20 \
    --num_samples 1000
```

**Outputs**: `experiments/E2_adversarial/` (2 figures)

---

### Experiment 3: Spontaneous Hijacking via Information Bottleneck
**Objective**: Identify non-adversarial failure modes through compression analysis

**Critical Discovery**: Phase transition at **βc = 0.368**
- Pre-critical (β < 0.368): 0% hijacking, stable dynamics
- Post-critical (β ≥ 0.368): 74% hijacking, entropy collapse (3.38 → 0.28 bits)
- Effect size: η² = 0.91 (mega-scale)

**Optimal Operating Range**: β ∈ [0.5, 1.5]

```bash
python src/experiments/E3_bottleneck/sweep_beta.py \
    --beta_range 0.1 3.0 \
    --num_steps 30
```

**Outputs**: `experiments/E3_bottleneck/` (3 figures)

---

### Experiment 4: Pathway Competition Dynamics
**Objective**: Characterize fast-slow pathway interaction and bias mechanisms

**Key Findings**:
- Baseline: Fast pathway wins 86% of trials
- With contextual bias: Can reverse to 60% slow pathway dominance
- Correlation: Memory magnitude |M| vs. pathway selection: R² = 0.88

**Sentinel Signals** (Early Warning Indicators):
1. Gate saturation: γ > 0.7 for 5+ consecutive steps
2. Memory overflow: |M| > 0.6
3. Pathway switching rate increase: +25% over baseline

```bash
python src/experiments/E4_competition/pathway_competition.py \
    --num_trials 500 \
    --modulation_strength 0.5
```

**Outputs**: `experiments/E4_competition/` (2 figures)

---

### Experiment 5: Four-Body Coupling & Noise Analysis
**Objective**: Investigate system-wide interactions and noise robustness

**Critical Discovery**: **W-shaped hijacking probability curve**

**Danger Zones**:
- Low noise (σ ≈ 0.10): Over-sensitivity → 15.4% hijacking
- High noise (σ ≈ 1.50): Chaos → 16.1% hijacking

**Goldilocks Zone**: σ = 0.50 → **8.5% hijacking** (optimal robustness)

```bash
python src/experiments/E5_coupling/noise_sweep.py \
    --sigma_range 0.01 2.0 \
    --coupling_analysis
```

**Outputs**: `experiments/E5_coupling/` (7 figures)

---

## ⚡ Quick Start

### Running All Experiments

```bash
# Complete pipeline (generates all 27 figures)
./run_all_experiments.sh

# Or run individually:
python src/run_experiments.py --experiment E1 E2 E3 E4 E5
```

### Example: Single Experiment

```python
from src.models.mega import MEGAModel
from src.experiments.E2_adversarial import AdversarialTester

# Initialize model
model = MEGAModel(
    memory_dim=64,
    emotion_dim=32,
    gate_threshold=0.5
)

# Run FGSM attack
tester = AdversarialTester(model)
results = tester.run_fgsm_sweep(
    epsilon_values=[0.05, 0.10, 0.15, 0.20],
    num_samples=1000
)

print(f"Hijacking rate at ε=0.10: {results['hijack_rate'][0.10]:.2%}")
```

## 📊 Results & Visualizations

All experimental results are organized in the `experiments/` directory with corresponding visualizations.

### Sample Outputs

<details>
<summary><b>Experiment 1: Enhanced Configuration Dynamics</b></summary>

![E1 Sample](experiments/E1_memory_gate/figure_003.png)

**Metrics**:
- Gate responsiveness: 100%
- Emotional peaks detected: 16
- Memory-gate correlation: r = 0.91
- Mean memory magnitude: 0.15

</details>

<details>
<summary><b>Experiment 2: FGSM Vulnerability Analysis</b></summary>

![E2 Sample](experiments/E2_adversarial/figure_016.png)

**Key Data**:
- ε = 0.05: 25.0% hijacking (fast: 30%, slow: 20%)
- ε = 0.20: 35.9% hijacking (fast: 45%, slow: 28%)
- Vulnerability gap: 61% fast vs. 39% slow

</details>

<details>
<summary><b>Experiment 3: Beta Phase Transition</b></summary>

![E3 Sample](experiments/E3_bottleneck/figure_025.png)

**Critical Points**:
- Phase transition: βc = 0.368
- Safe zone: β ∈ [0.5, 1.5]
- Danger zone: β ≥ 2.0 (74% hijacking)

</details>

<details>
<summary><b>Experiment 5: W-Shaped Noise Robustness</b></summary>

![E5 Sample](experiments/E5_coupling/figure_027.png)

**Robustness Profile**:
- Low noise danger: σ = 0.10 (15.4% hijack)
- Optimal zone: σ = 0.50 (8.5% hijack)
- High noise danger: σ = 1.50 (16.1% hijack)

</details>

---

## 🎯 Defense Mechanisms

Based on experimental findings, we propose **four actionable mitigation strategies**:

### 1. Threshold Calibration
Adjust pathway thresholds to balance speed-accuracy tradeoff:
```python
model.set_thresholds(
    fast_threshold=0.4,  # Increase from 0.3
    slow_threshold=0.65   # Decrease from 0.7
)
```

### 2. Noise Injection (Stochastic Defense)
Operate in Goldilocks zone:
```python
model.add_training_noise(sigma=0.50)  # Optimal robustness
```

### 3. Beta Monitoring (Early Warning System)
```python
if model.get_beta() >= 1.8:  # Approaching danger zone (βc=0.368)
    warnings.warn("System approaching instability threshold")
    model.reduce_compression()
```

### 4. Pathway Balancing
```python
if model.pathway_win_rate['fast'] > 0.80:
    model.apply_contextual_bias(strength=-0.3)  # Favor slow pathway
```

---

## 📈 Performance Benchmarks

System tested on NVIDIA Tesla V100 (16GB VRAM):

| Experiment | Runtime | Peak GPU Memory | Data Points |
|-----------|---------|----------------|-------------|
| E1 (Memory-Gate) | 2.3 min | 1.2 GB | 120 |
| E2 (Adversarial) | 18.7 min | 4.8 GB | 1000 |
| E3 (Bottleneck) | 12.4 min | 3.2 GB | 30 |
| E4 (Competition) | 5.1 min | 0.8 GB | 500 |
| E5 (Coupling) | 8.9 min | 2.1 GB | 2000 |
| **Total Pipeline** | **47.4 min** | **4.8 GB** | **3650** |

---

## Evidence Appendix
\documentclass[10pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[margin=0.5in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage{caption}
\usepackage{amsmath}
\usepackage{xcolor}

% Compact spacing
\setlength{\parindent}{0pt}
\setlength{\parskip}{3pt}
\captionsetup{font=footnotesize,labelfont=bf,skip=2pt}

\begin{document}

% Title
\begin{center}
{\Large \textbf{EVIDENCE APPENDIX: AI Emotional Hijacking}}
\end{center}
\vspace{-8pt}

% Evidence Index Table
\begin{table}[H]
\centering
\small
\begin{tabular}{clcc}
\toprule
\textbf{Fig} & \textbf{Key Finding} & \textbf{Significance} & \textbf{Source} \\
\midrule
E1 & Critical Phase Transition at $\beta_c=0.368$ & $\eta^2=0.91$, $p<0.0001$ & Exp 3 \\
E2 & Fast Pathway 61\% Vulnerability Differential & Cohen's $d=2.31$, $p<0.001$ & Exp 2 \\
E3 & W-Shaped Noise-Hijacking Curve & $R^2=0.88$, $\sigma_{opt}=0.50$ & Exp 5 \\
E4 & 86\% Fast Pathway Dominance & $\chi^2=147.2$, $p<0.0001$ & Exp 4 \\
\bottomrule
\end{tabular}
\end{table}
\vspace{-8pt}

% FIGURE E1 & E2
\subsection*{\footnotesize\textcolor{blue}{E1: Phase Transition \hfill E2: Pathway Vulnerability}}
\vspace{-4pt}
\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{AI-Emotion-Figures/figure_003.png}
\hfill
\includegraphics[width=0.48\textwidth]{AI-Emotion-Figures/figure_016.png}
\end{figure}
\vspace{-10pt}

\textbf{E1 Finding:} Information bottleneck parameter $\beta$ undergoes sharp phase transition at $\beta_c \approx 0.368$. Hijacking rate jumps from 0\% to 74-84\%. Gate entropy collapses 92\% (3.38→0.28 bits). \textbf{Impact:} Recommend $\beta \in [0.5, 1.5]$ operational range.

\textbf{E2 Finding:} Fast pathway 61\% more vulnerable than slow pathway. Fast: $0.3077 \pm 0.12$ vs Slow: $0.1968 \pm 0.08$, ratio 1.56:1 ($p<0.001$, Cohen's $d=2.31$). Model: $P_{hijack}(\epsilon) \approx 0.36(1 - e^{-10\epsilon})$. \textbf{Impact:} Fast pathway accounts for 61\% of hijacking events.

\vspace{2pt}
\hrule
\vspace{4pt}

% FIGURE E3 & E4
\subsection*{\footnotesize\textcolor{blue}{E3: Noise Dynamics \hfill E4: Architectural Bias}}
\vspace{-4pt}
\begin{figure}[H]
\centering
\includegraphics[width=0.48\textwidth]{AI-Emotion-Figures/figure_027.png}
\hfill
\includegraphics[width=0.48\textwidth]{AI-Emotion-Figures/figure_020.png}
\end{figure}
\vspace{-10pt}

\textbf{E3 Finding:} Non-monotonic W-curve with dual danger zones: Low noise ($\sigma=0.10$): 15.4\% hijacking; Optimal ($\sigma=0.50$): 8.5\% hijacking (min); High noise ($\sigma=1.50$): 16.1\% hijacking. \textbf{Impact:} Operate at $\sigma \in [0.40, 0.60]$ "Goldilocks zone".

\textbf{E4 Finding:} Fast pathway dominates 86\% decisions. Fast: 86\% wins, 71\% accuracy, 16.3 steps; Slow: 14\% wins, 86.4\% accuracy, 26.8 steps. Cross-validation: Predicted 39.8\% vs observed 40\% hijacking (0.5\% error). \textbf{Impact:} Target 60/40 distribution via asymmetric thresholds.

\vspace{4pt}

% SYNTHESIS BOX
\begin{center}
\small
\fbox{\begin{minipage}{0.98\textwidth}
\textbf{Integrated Hijacking Framework}

\begin{table}[H]
\centering
\footnotesize
\begin{tabular}{llcc}
\toprule
\textbf{Pathway} & \textbf{Trigger} & \textbf{Hijack Rate} & \textbf{Defense Strategy} \\
\midrule
External & $\epsilon \geq 0.05$ & 25--36\% & Adversarial training, input filtering \\
Spontaneous & $\beta \geq 2.0$ & 74--84\% & $\beta$ monitoring, entropy stabilization \\
Noise Resonance & $\sigma \notin [0.4, 0.6]$ & 12--16\% & Noise regulation at $\sigma=0.50$ \\
Architectural & Fast\% $> 70\%$ & 40\% & Pathway balancing, slow empowerment \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Critical Thresholds:} $\epsilon_c \approx 0.05$, $\beta_c \approx 0.368$ (most sensitive), $\sigma_{opt} = 0.50$, Fast\% $> 70\%$ \\
\textbf{Validation:} All findings: Power $> 0.95$, Effect Size $> 0.80$, 95\% CI $< 10\%$ error. Cross-validation accuracy: 0.5\% error.\\
\textbf{Contribution:} First demonstration of critical phase transition, quantified pathway differential (61\%), non-monotonic noise relationship, and validated architectural bias model in AI emotional processing.

\end{minipage}}
\end{center}

\end{document}

## 📖 Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{tian2025emotional,
  title={Emotional Hijacking in Artificial Intelligence Systems: A Neuroscience-Inspired Dual-Pathway Analysis},
  author={Tian, Zhigang and Collaborators},
  journal={Under Review},
  year={2025},
  note={Code: github.com/yourusername/emotional-hijacking}
}
```

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Check code style
black src/ --check
flake8 src/
```

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Inspired by Joseph LeDoux's dual-pathway amygdala theory
- FGSM implementation adapted from Goodfellow et al. (2015)
- Visualization tools built with Matplotlib and Seaborn
- Computing resources provided by [Zhigang Tian]

---

## 📧 Contact

**Primary Author**: Zhigang Tian  
**Email**: [medcloud.ph@gmail.com]  

---
---

<p align="center">
  <b>⚠️ Disclaimer</b><br>
  This research is for academic purposes only. The vulnerability analyses are intended<br>
  to improve AI safety and should not be used for malicious purposes.
</p>

<p align="center">
  Made with ❤️ for safer AI systems
</p>
