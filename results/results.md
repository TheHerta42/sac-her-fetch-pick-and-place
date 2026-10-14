# FetchPickAndPlace — Domain-Randomization Evaluation

- 300 episodes per model per condition
- `target_range` (goal sampling radius): **0.15** (default)
- Seed: 42 — identical across models, so every model faces the same episode set and the rows below can be compared directly

Success = the episode ended before the 50-step cap, which only happens when `SuccessTerminationWrapper` sees `is_success`. Brackets are 95% Wilson intervals.

`dr_trained` reproduces the randomization the fine-tune actually saw (`domain_randomization.py:DR_CONFIG`); `dr_severe` widens the same axes past that distribution to find where each policy breaks.

## Success rate

| Model | standard | dr_trained | dr_severe |
|---|---|---|---|
| v1 (3M steps, no DR) | 100.0% [99,100] | 98.3% [96,99] | 79.0% [74,83] |
| v3 (6M steps, no DR) | 100.0% [99,100] | 99.3% [98,100] | 86.3% [82,90] |
| dr_v1 (9M steps, DR) | 100.0% [99,100] | 99.7% [98,100] | 91.3% [88,94] |

## Mean completion length (lower is faster)

| Model | standard | dr_trained | dr_severe |
|---|---|---|---|
| v1 (3M steps, no DR) | 9.74 | 11.25 | 23.54 |
| v3 (6M steps, no DR) | 9.93 | 10.64 | 19.42 |
| dr_v1 (9M steps, DR) | 10.00 | 10.29 | 16.96 |

## Degradation vs. own standard performance

| Model | Physics | Δ success rate | Δ length | p-value (reward) | Significant |
|---|---|---|---|---|---|
| v1 (3M steps, no DR) | dr_trained | -1.7% | +1.51 | 0.0017 | yes |
| v1 (3M steps, no DR) | dr_severe | -21.0% | +13.81 | 0.0000 | yes |
| v3 (6M steps, no DR) | dr_trained | -0.7% | +0.71 | 0.0638 | no |
| v3 (6M steps, no DR) | dr_severe | -13.7% | +9.49 | 0.0000 | yes |
| dr_v1 (9M steps, DR) | dr_trained | -0.3% | +0.29 | 0.3971 | no |
| dr_v1 (9M steps, DR) | dr_severe | -8.7% | +6.96 | 0.0000 | yes |

## DR fine-tune (dr_v1 (9M steps, DR)) vs. baseline (v3 (6M steps, no DR))

| Physics | Δ success rate | Δ length | Welch p | Paired p (same seed) | Significant |
|---|---|---|---|---|---|
| standard | +0.0% | +0.07 | 0.8205 | 0.1084 | no |
| dr_trained | +0.3% | -0.36 | 0.3886 | 0.1343 | no |
| dr_severe | +5.0% | -2.46 | 0.0290 | 0.0002 | yes |

Success-rate differences are percentage points. The paired test uses the shared seed, so each model answers the same task instances — far more sensitive than the unpaired Welch test at these near-ceiling success rates. Paired p for `dr_trained` is *not* significant: within the training distribution the fine-tune buys nothing, which is the honest reading. Its advantage only appears once the physics leaves that distribution.

## Robustness retention

Fraction of each model's own standard-environment success rate that survives under perturbed physics. Higher = more robust; the *relative* drop is the fair comparison because all three models start at 100%.

| Model | dr_trained | dr_severe |
|---|---|---|
| v1 (3M steps, no DR) | 98.3% | 79.0% |
| v3 (6M steps, no DR) | 99.3% | 86.3% |
| dr_v1 (9M steps, DR) | 99.7% | 91.3% |

## Headline

Under the stress condition every model degrades, but not equally: **v1 (3M steps, no DR)** retains 79% of its nominal success rate while **dr_v1 (9M steps, DR)** retains 91%. The DR fine-tune also finishes tasks **2.5 steps faster** than the baseline it was fine-tuned from, and that gap is significant under both the paired (p=0.0002) and unpaired (p=0.0290) tests. The honest caveat: within the training distribution (`dr_trained`) the improvement is not significant.

