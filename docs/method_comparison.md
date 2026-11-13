# Paired method comparison and frozen validation comparator

`compare-batch` reads a completed, integrity-verified batch. It compares every
processor condition at the selected baseline's operating point, keeping delay,
noise, sampling interval, window size and metric thresholds fixed. It does not
choose a winner by scanning test results.

## Commands

Explore a validation or unlabelled/synthetic batch:

```powershell
poetry run poseahead compare-batch outputs/validation_suite raw-walking --seed 42 --repetitions 2000 --output outputs/validation_comparison.json
```

After choosing the fixed comparator on validation, freeze it explicitly:

```powershell
poetry run poseahead freeze-comparator outputs/validation_suite raw-walking outputs/fixed_comparator.json
poetry run poseahead compare-batch outputs/test_suite raw-test-walking --frozen-comparator outputs/fixed_comparator.json --output outputs/test_comparison.json
```

The run ID identifies a processor/config condition, not just that one clip. All
sequences in its condition are used as the baseline panel. Frozen comparator
artifacts record the validation batch hash, exact condition and frozen decision
thresholds. Test comparison rejects a missing/mismatched comparator and re-verifies
the validation provenance. Existing output files are never overwritten. Artifact
checks cannot prove that a person did not inspect test results before freezing;
chronological research discipline remains necessary.

## Pairing and exclusions

Pairs use the same sequence ID and clock. For each pair, input/reference hashes,
seed, evaluation start/end/step, validity masks and eligible latency joint-window
masks must match. Different percentages or equal percentages with different masks
are excluded with a reason. All three primary metrics must be available. A method
that produces fewer valid observations cannot silently gain an advantage through
scoring a different subset.

Reports include baseline/candidate sequence counts, eligible pairs and exclusions.
A complete panel requires all sequences on both sides to match and remain eligible.
Rule flags on an incomplete subset are exploratory only; the separate
`eligible_for_frozen_rule_claim` field stays false. At least two eligible sequences
and a frozen validation comparator are required before that field can become true.
It does **not** certify sample independence, a sufficient sample size or validity
of the manually provided dataset labels.

## Confidence intervals and decision rule

For each primary metric, calculate per-sequence candidate-minus-baseline differences.
Resample paired sequences with replacement, equally weighting sequences, and report
the 2.5/97.5 percentiles of resampled mean differences. The same draws are used for
all three metrics. Bootstrap seed and repetitions are saved; default repetitions
are 2000, permitted range 100–100,000. A single sequence has no interval or rule claim.

The implementation reports the frozen SPEC rules:

1. Conservative noninferiority interpretation: the upper 95% CI bound is <= 0 for
   latency, pose error and jitter, and strictly < 0 for at least one.
2. Mean latency decreases by at least 20 ms while mean MPJPE and jitter each grow
   by at most 10%. A zero comparator error allows zero candidate error only.

Rule 2 is a point-estimate threshold test, not a confidence guarantee. CIs are
pointwise, not a simultaneous three-metric confidence region; there is no multiple
comparison adjustment. Sequence bootstrap does not capture dependence among takes
or cameras of the same subject. Small samples and nearly identical clips can yield
misleadingly narrow intervals; subject-level sensitivity analysis remains pending.
Negative lag represents leading the reference, not automatically better perception.

## Pareto report

The descriptive frontier uses mean latency, MPJPE and jitter on complete matched
panels. A point is dominated if another is no worse on all three and better on at
least one; ties remain on the frontier. Baseline and each included comparison are
identified explicitly. This is a point-estimate frontier, not a significance test,
and excluded/incomplete conditions are not part of it.

Only arithmetic, deterministic bootstrapping, pairing protections and validation/test
workflow have been tested on synthetic data. No real Human3.6M research conclusion
or claim that adaptive prediction beats a fixed comparator is established.
