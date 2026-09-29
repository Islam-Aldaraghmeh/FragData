# Training audit

This audit leaves the original data, checkpoint, notebook, and metrics unchanged.
The diagnostic controls evaluate validation only. They are diagnostic checks,
not independently confirmed performance improvements.

## Checkpoint and optimization

The original run reproduced **bitwise-identical selected weights**, with maximum
absolute parameter difference **0.0**. Every recorded original training loss and
validation cosine also reproduced exactly. The selected checkpoint really is
epoch 1; the saved file is not accidentally the final, overfit checkpoint.

The model has **140,545 parameters**, trained on **210 distinct molecules / 626
spectra**. Its 68,220 candidate rows are latent alternatives, not 68,220 independent
intensity labels. They map to 13,152 supported peak rows.

| Checkpoint | Post-epoch train mean cosine | Validation mean cosine |
|---|---:|---:|
| Epoch 1, saved | 0.5461 | 0.5095 |
| Epoch 3 | 0.6991 | 0.4715 |
| Epoch 9 | 0.7749 | 0.4751 |

The model can learn the training spectra, but the additional fit does not transfer
to new molecules. Training for longer alone is not a remedy. These are
post-epoch scores, unlike the original history's training loss, which averages
scores obtained while weights change within an epoch.

## Three bounded controls

All controls reuse exactly the existing train and validation splits, seed,
architecture, max pooling, cosine loss, batch size 16, maximum 40 epochs, and
patience 8. The original checkpoint is the only model evaluated on test.

| Model/control | Selected epoch | Train mean cosine | Validation mean cosine |
|---|---:|---:|---:|
| Original | 1 | 0.5461 | 0.5095 |
| Slower/regularized optimization: lr 0.0001, weight decay 0.001 | 8 | 0.5939 | 0.5133 |
| Remove precursor fingerprint | 1 | 0.5365 | 0.5061 |
| Remove both fingerprints; retain eight descriptors | 40 | 0.5474 | 0.5429 |

The slower/regularized control changes two optimizer settings together; it does
not isolate the effect of either one. Its improvement is small and overfitting
still develops. Removing precursor fingerprints alone does not resolve the
problem, so precursor memorization alone is not an adequate explanation.

The descriptors-only control generalizes better on validation with a much
smaller train/validation gap. This supports simplifying or regularizing the
high-dimensional fingerprint mapping for this small data regime. It does not
prove fragment structure is useless. Its selected epoch is the imposed
40-epoch boundary, so it should not be described as having converged. Its
zeroed fingerprint columns retain the original architecture, but only eight
input dimensions vary.

## Collision-energy sensitivity

For every validation spectrum, the audit fixes all candidate structures and
locations, predicts once at CE 20 and once at CE 60, and compares the two
independently base-peak-normalized predictions on the same full peak axis.
This isolates sensitivity of the supplied scoring function. It does not
assert that both experimental spectra would contain exactly the same peaks.

| Model/control | Mean cosine between CE 20 and 60 predictions | Mean absolute intensity change |
|---|---:|---:|
| Original | 0.99999893 | 0.0007175 |
| Slower/regularized optimization | 0.99999534 | 0.0013821 |
| Remove precursor fingerprint | 0.99999162 | 0.0019483 |
| Descriptors only | 0.97164098 | 0.0823961 |

Intensity change is on the [0,1] scale: the original changes by only about
0.072 percentage points on average, versus 8.24 percentage points for the
descriptors-only control. Thus the saved model barely uses CE to change the
predicted relative intensities. The descriptors-only control can respond to
CE in the same architecture and loss. This does not establish that its
energy response is chemically correct, but suggests a learnability/input
representation issue rather than a missing energy input or inference bug.

## Hard-max gradient coverage

At the saved checkpoint, 13,152 of 68,220 training candidate rows (**19.28%**)
are the maximum in their peak. Only those maxima can receive a score gradient.
Among candidate rows in ambiguous peaks, **15.99%** are maxima. All but one
of the winning rows have an absolute cosine-loss score gradient greater than
1e-12. Validation shows the same pattern (18.52% of all candidates, 15.57%
within ambiguous peaks).

This is the expected effect of the requested hard maximum, not a pooling bug:
most alternatives receive no direct intensity gradient at a given checkpoint.
The audit does not establish how much of the held-out error this causes, nor
whether the winning latent candidate is chemically correct.

Reproduction: `.venv/bin/python results/experiment_2/error_analysis/training_audit.py`.
Full numerical history and control settings are in `training_audit.json`.
