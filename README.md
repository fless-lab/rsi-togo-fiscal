# RSI: Rule-State Inference

### A Bayesian Framework for Compliance Monitoring in Rule-Governed Domains
*Evidence from Francophone African Fiscal Systems*

**Abdou-Raouf Atarmla** — Togo DataLab / INPT Rabat

[![arXiv](https://img.shields.io/badge/arXiv-2603.21610-b31b1b.svg)](https://arxiv.org/abs/2603.21610)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fless-lab/rsi-framework/blob/main/experiments/togo-fiscal/walkthrough.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is RSI?

Most ML frameworks for compliance monitoring learn rules from data. **RSI inverts this paradigm**: known regulatory rules are encoded as Bayesian priors, and compliance monitoring is cast as posterior inference over a latent rule-state space.

RSI operates at **two complementary levels**:

- **Population-level Bayesian inference** estimates the compliance rate for each rule across the population, with calibrated uncertainty. *"63% of taxpayers comply with VAT, ± 3%."*

- **Entity-level deterministic scoring** identifies non-compliant entities per rule using continuous compliance signals, without any labeled data. The aggregation strategy (flag if 1 rule violated, 2+, etc.) is a domain policy choice.

> **On individual Bayesian inference:** The current version provides deterministic entity scoring. *Sequential RSI*, where the posterior at period *t* becomes the prior at *t+1*, will enable full entity-level Bayesian tracking over time. This is the natural next step and is under active development.

**Three theoretical guarantees** are proven:
- **T1** — Regulatory changes absorbed in O(1) time (no retraining)
- **T2** — Bernstein-von Mises posterior consistency (uncertainty shrinks at 1/√N)
- **T3** — Monotone ELBO convergence under mean-field VI

---

## Key Results (Togo Fiscal Instance)

### Per-Rule Performance (zero-shot, no labels)

| Rule | RSI F1 | RSI AUC | RBS F1 | Advantage |
|------|--------|---------|--------|-----------|
| R1_TVA | 0.865 | 0.874 | 0.765 | +0.100 |
| R2_IS | 0.739 | 0.892 | 0.740 | -0.001 |
| R3_IMF | 0.545 | 0.914 | 0.541 | +0.005 |
| R4_TPU | 0.825 | 0.691 | 0.720 | +0.105 |
| R5_IRPP | 0.690 | 0.879 | 0.690 | +0.000 |
| R6_PAT | 0.841 | 0.876 | 0.783 | +0.058 |
| R7_DECL | 0.630 | 0.879 | 0.621 | +0.009 |
| R8_BANK | 0.791 | 0.867 | 0.739 | +0.053 |
| **Mean** | **0.741** | **0.859** | **0.700** | **+0.041** |

- **T1:** RSI absorbs regulatory changes in ~0.002ms vs ~1,000ms for retraining (around 500,000x speedup; exact ratio is hardware-dependent, the O(1) guarantee is machine-independent)
- **T2:** Posterior uncertainty decreases at 1/√N (ratios 1.86-2.03 vs theoretical 2.0)
- **T3:** ELBO gain = +815, monotone, converges in 1 sweep
- **Missing data:** At 50% missing, RSI advantage over RBS widens to +0.25-0.37 per rule

---

## Quick Start

```bash
pip install numpy scipy scikit-learn pandas
python experiments/togo-fiscal/run.py
```

---

## Dataset: RSI-Togo-Fiscal-Synthetic v2.0

2,000 synthetic enterprises across 2 regulatory periods, grounded in real OTR fiscal rules (2022-2025):

- **8 fiscal rules** (6 threshold-based + 2 universal)
- **4 market segments** (informal 45%, small formal 25%, medium 18%, large 12%)
- **Realistic noise:** under-declaration ratio ~0.70, 18% missing data
- **Regulatory change event:** VAT threshold 60M → 100M FCFA (Law n°2024-007)
- **Fully reproducible:** seed=42

---

## Applying RSI to a New Domain

RSI is domain-agnostic. The core (`src/core.py`) never changes. You only write a domain adapter. Example for medical protocol compliance:

```python
from core import PopulationRSI, EntityScorer

# 1. Define rules as priors (institutional knowledge)
priors = {
    'HbA1c_control':   {'alpha': 6, 'beta': 4, 'sigma_drift': 1.0},
    'Annual_eye_exam': {'alpha': 5, 'beta': 5, 'sigma_drift': 0.5},
    'Metformin_first': {'alpha': 7, 'beta': 3, 'sigma_drift': 0.5},
}

# 2. Compliance signals: 1 = compliant, 0 = non-compliant, NaN = unknown
signals = {
    'HbA1c_control':   {'patient_001': 1, 'patient_002': 0, 'patient_003': float('nan')},
    'Annual_eye_exam': {'patient_001': float('nan'), 'patient_002': 1, 'patient_003': 0},
    'Metformin_first': {'patient_001': 1, 'patient_002': 1, 'patient_003': 0},
}

# 3. Applicability: which rules apply to which entities
applicability = {
    'HbA1c_control':   {'patient_001': True, 'patient_002': True, 'patient_003': True},
    'Annual_eye_exam': {'patient_001': True, 'patient_002': True, 'patient_003': False},
    'Metformin_first': {'patient_001': True, 'patient_002': True, 'patient_003': True},
}

# 4. Run
rsi = PopulationRSI(priors)
results = rsi.fit(signals, applicability)

# Population result: E[c_HbA1c] = 0.63 ± 0.04
# "63% of diabetic patients have controlled HbA1c, with calibrated uncertainty"
for rule, post in results['population'].items():
    print(f"{rule}: E[c]={post['E_c']:.3f} ± {post['std_c']:.3f}")
```

Any domain with (i) known rules, (ii) observable signals, (iii) definable applicability is a candidate. Immediate extensions include environmental regulation, anti-money-laundering, and legal contract monitoring.

---

## Citing

```bibtex
@article{atarmla2026rsi,
  title   = {Rule-State Inference ({RSI}): A Bayesian Framework
             for Compliance Monitoring in Rule-Governed Domains},
  author  = {Atarmla, Abdou-Raouf},
  journal = {arXiv preprint arXiv:2603.21610},
  year    = {2026},
  url     = {https://arxiv.org/abs/2603.21610}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

## Contact

Abdou-Raouf Atarmla — [achilleatarmla@gmail.com](mailto:achilleatarmla@gmail.com) | [abdou-raouf.atarmla@datalab.gouv.tg](mailto:abdou-raouf.atarmla@datalab.gouv.tg) | [atarmla.abdouraouf@ine.inpt.ac.ma](mailto:atarmla.abdouraouf@ine.inpt.ac.ma)