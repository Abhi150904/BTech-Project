# Phase 7 · Scoring a pending tool proposal

The offline detector estimates whether the pending action has the Phase 5 label
`violation_attempted=True`. A score describes an attempt. Whether an effect
actually completed requires separate outcome evidence. Both models run offline
and return scores for analysis. Gateway decisions belong to Phase 8.

## Train and evaluate

From the repository root in a Python 3.11+ environment, install the optional
neural extra. Training and inference use saved inputs and CPU PyTorch.

```shell
python -m pip install -e ".[neural]"
runtime-security-train research/scripted-v1/dataset.json research/scripted-v1/manifest.json research/scripted-v1/baselines.json --config configs/model.toml --out research/my-phase7-run
```

Choose a new output directory. A successful run produces `tool_only.json`,
`enriched.json` and `report.json`. The trainer audits the Phase 5 corpus,
recomputes its grouped manifest, compares dataset and manifest hashes and exact
run memberships with the supplied Phase 6 report, and verifies all three
baseline confusion matrices against the held-out proposal IDs and labels.
An unknown action label remains unknown. The report counts these by partition
and counts actions from excluded runs separately.

## What the models see

Each sample ends at the proposal identified by its action label. Only the
observed prefix of that run enters preprocessing. The matched tool result,
subsequent events, effect ledger, outcome, prompts, raw arguments and run or
scenario identifiers are excluded from model tensors. Where available, Phase 4
uses versioned task grants to derive the authorization category.

| Variant | Timesteps | Inputs per timestep |
| --- | --- | --- |
| Tool-only | Proposals | Tool identity |
| Enriched | All observed events through the proposal | Phase 4 tool, operation, resource class, sensitivity, destination trust, authorization, event kind, previous tool, transition, result status and mismatch, plus three numeric features |

The numeric fields count prior sensitive attempts and successes and indicate a
send after a sensitive read. Each categorical field gets its own embedding.
Only allowlisted values seen in known-label training windows get learned IDs
starting at 2. ID 0 pads a sequence. ID 1 represents explicit unknowns and
unseen values. Phase 4 supplies zero defaults for absent numeric counters or
flags. Malformed numeric fields fail schema validation. Numeric inputs use
training-window means and population standard deviations, with scale 1 for
constant fields. Earlier events can appear in several training examples.

Each variant retains its most recent 32 timesteps by default, including the
pending proposal. Phase 4 computes cumulative counts before truncation. A
packed bidirectional LSTM sees the observed prefix in both directions. A linear
attention head assigns padding zero weight before pooling the hidden states.
The binary head applies dropout during training and a sigmoid at scoring time.

## Selection and reproducibility

Both variants use the same seed, optimizer, partitions and training budget.
The checked-in configuration uses 8-dimensional embeddings, 12 hidden units
per direction, one recurrent layer, dropout 0.1, seed 42, Adam at 0.01, batch
size 16 and a 30-epoch limit. Unweighted binary cross-entropy and shuffling
without replacement suit the fixture's 9 positive and 8 negative training
proposals. Fitting requires both known training classes and at least one known
validation label.

The trainer restores the epoch with the lowest known-label validation loss.
Patience is five consecutive epochs without improvement. After restoration,
validation F1 selects the threshold, breaking ties by fewer false positives
then higher threshold. A missing validation class triggers the fixed 0.5 fallback.
Test labels enter integrity checks and final metrics only. The report includes
the loss history, stopping reason, best epoch, validation curve, model fields,
threshold, held-out scores, split memberships and provenance hashes.

Training uses deterministic PyTorch algorithms, a seeded shuffle and one CPU
thread. For exact replication use the reported PyTorch build and the same CPU
environment. Other builds and hardware can vary numerically. Timing covers
one prefix at a time, including preprocessing and the forward pass, and excludes
disk I/O and Phase 4 extraction. Median and nearest-rank p95 are in nanoseconds.

## Score one sanitized prefix

Start with the Phase 3 JSONL trajectory from one run. Copy its lines from
`run_start` through the proposal of interest, inclusive, to `prefix.jsonl`.
For example, these commands score the first proposal from an exported single-run
trajectory whose first two events are `run_start` and `proposal`.

```powershell
python -c "from pathlib import Path; rows = Path('trajectory.jsonl').read_bytes().splitlines(keepends=True); Path('prefix.jsonl').write_bytes(b''.join(rows[:2]))"
runtime-security-score research/phase7-scripted-v1/enriched.json prefix.jsonl --policy configs/policy_metadata.toml
```

Python callers can pass the ordered `TrajectoryEvent` list directly to
`score_prefix(load_checkpoint(checkpoint_path), events, policy_path)` from
`runtime_security.detectors.inference`. The scorer checks sequence, run and
event identity, lifecycle and the feature policy digest (accepting LF/CRLF
checkout differences). It extracts Phase 4 rows and runs the checkpoint's
saved preprocessing. The output contains the score, threshold, boolean
`unsafe_proposal`, variant, final sequence number and model-version digest.

The v1 checkpoint is JSON with numeric weight arrays. The loader checks the
schema, PyTorch major version, configuration digest, vocabulary, normalization
dimensions, tensor names, shapes and finiteness. The checkpoint also records
the training membership fingerprint, corpus and policy hashes, best epoch,
calibration method and curve. Compare the canonical checkpoint digest with the
one in a trusted report to identify the artifact. The format carries no
executable serialized Python objects.

## Frozen fixture results

There are three independent groups. The test group contains two authorized
reads. Validation has four known negatives and test has two, with zero known
positives in either. Every method below uses the same two test proposals.

| Method | True negatives | False positives | FPR |
| --- | ---: | ---: | ---: |
| Tool-only BiLSTM | 2 | 0 | 0 |
| Enriched BiLSTM | 2 | 0 | 0 |
| Phase 6 sequence logistic | 0 | 2 | 1 |
| Phase 6 semantic logistic | 2 | 0 | 0 |
| Phase 6 random forest | 2 | 0 | 0 |

Both neural thresholds used the 0.5 fallback. Tool-only restored epoch 1 and
enriched restored epoch 2. Recall is undefined for all methods. For the neural
models, precision and F1 are also undefined. JSON represents zero-denominator
rates as `null`. Two latency samples per variant suffice to exercise the
measurement path only. A research comparison needs independent task and attack
families, authorized lookalikes and unsafe proposals in validation and test,
and enough examples per class to estimate uncertainty.

[Inspect the full fixture report](../research/phase7-scripted-v1/report.json).
