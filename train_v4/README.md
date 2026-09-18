# Train v4 — video drifting with a Flash-WAM-style action objective

V4 extends `train_v3` with a Flash-WAM-style action objective: two-timestep
action consistency plus the native action flow-matching regularizer. Each
microbatch uses the complete aligned GT video/action sequence of its sampled
source. The normalized actions are not clipped; the initial padded action
frame receives ordinary noise/timesteps and participates in valid-channel loss. Teacher-generated pairs remain exclusive to the video
kernel/execution branches. GT current video is detached and never conditions
student video generation (apart from the observed initial frame).
Both terms are masked to the 16 used RoboTwin channels. Their weights are
`1.0` and `0.01`, respectively.

Every logged update is appended to `metrics.jsonl`; the console line also
includes action consistency and action flow-matching. Formal profiles require
W&B to remain disabled, and a local metrics-write failure stops every rank
instead of continuing without diagnostics.

Action noise/state and the FM target retain FP32 precision. Packed model
forwards still compute in BF16. Intermediate teacher Euler states retain all
channels until the final masked loss. Only the video/probe branch retains its
initial-observation convention and clipped historical action conditions.
See DECISIONS.md for the full-sequence upstream action parity audit.
Each successful optimizer update logs `duration_seconds`; regular metrics also
include `timing/optimizer_step_seconds`. This measures data loading, all
microbatches, backward, Adam and EMA, excluding initialization/checkpoint I/O.

The action pass runs after the video and execution graphs are released. All
four loss terms update the same online student in one optimizer step, and the
EMA target is updated once after that optimizer step.

The existing dataset, teacher bank, and signature cache paths are intentionally
reused, including the cache name ending in `_v3`. W&B is disabled and cannot
be enabled through either formal v4 launch profile.

## Inherited clean-50 experiment

High-frequency training inputs use local storage under `/root/autodl-tmp`;
model checkpoints and training outputs remain on shared FS under
`/root/autodl-fs`. Run from `/root/driftwam` in the `lingbot-distill` conda
environment.

## One-command formal run

```bash
conda run -n lingbot-distill --no-capture-output \
  bash /root/driftwam/train_v4/run_flashwam500_clean50.sh
```

The first invocation automatically precomputes any missing 50-step paired
teacher actions and writes `CACHE_COMPLETE.json`. This cache is shared by
later instances. After completion, the same command starts the training run.
The default uses four GPUs and 250 optimizer updates.
It enables the lower-memory hard-matched execution objective with
`EXECUTION_RESPONSE_MODE=recompute_selected` and weight `0.1`.

Default artifacts:

```text
dataset: /root/autodl-tmp/robotwin-lerobot/lerobot_robotwin_eef_clean_50
teacher videos: /root/autodl-tmp/robotwin-lerobot/teacher_video_bank_clean_50
teacher actions: /root/autodl-tmp/robotwin-lerobot/teacher_signature_cache_clean_50_seed42_v3
output: /root/autodl-fs/wam/runs/driftwam_v4_action_response_clean50_seed42
```

The profile preserves global batch 32 on any positive GPU count that divides
32. For example, `NGPU=1` uses accumulation 32, `NGPU=2` uses 16, and the
default `NGPU=4` uses 8. A single A800 run is therefore supported but has much
lower throughput. Override `CACHE_NGPU` independently for cache preparation.

## Recommended launch sequence

Run a cache-only pass when you want to inspect it separately:

```bash
conda run -n lingbot-distill --no-capture-output \
  bash /root/driftwam/train_v4/precompute_teacher_signatures.sh
```

For the four-GPU memory check, run this bounded stress test first:

```bash
conda run -n lingbot-distill --no-capture-output \
  bash /root/driftwam/train_v4/smoke_test_4gpu.sh
```

It preserves global batch 32 (4 GPUs, batch 1, accumulation 8), all four
losses, full histories, precision, and the sync threshold. Default: two
optimizer updates. The first uses `[longest, 24, 24, 24, 24, 24, 24, 24]`;
the second uses `[24, 24, 24, 24, 24, 24, 24, longest]`. Here `24` is the
configured unsynced-history boundary and `longest` is discovered from valid
teacher-bank chunks. `MAX_TRAIN_STEPS=3` adds one
all-zero-history update; `1` stops after the first update. Sources are sampled
only from eligible chunks, with a shared history length across ranks. Sources
are distinct within each global microbatch when enough are available; rare
longest chunks repeat across ranks for this diagnostic only, with independent
per-rank noise. In the current bank, only two sources reach 50 latent history
frames, so each is assigned to two ranks; 298 sources support 24 frames.

The test logs each update and adds only `memory/peak_allocated_gib` and
`memory/peak_reserved_gib`: the maximum over all ranks for the entire update,
including optimizer/EMA. It aborts on a non-finite update, requires completed
caches, uses a separate output directory, and writes no training checkpoints.
The diagnostic sampling order is recorded as `history_stress_test=true` in
run configs; it cannot resume and does not replace formal random sampling.
Model loading, one teacher-only bandwidth calibration batch, and CUDA kernel
compilation still occur before the updates. This is a memory/execution check,
not a convergence or final quality evaluation.

For an optional one-update, one-GPU smoke test:

```bash
conda run -n lingbot-distill --no-capture-output \
  bash /root/driftwam/train_v4/smoke_test.sh
```

Finally launch the formal profile with the one-command invocation above.

Cache precomputation is resumable. For a bounded diagnostic subset, set
`CACHE_START_INDEX`, `CACHE_END_INDEX`, or `CACHE_MAX_SOURCES`. A subset does
not publish `CACHE_COMPLETE.json` and therefore cannot unlock formal training.

## Useful overrides

```bash
MAX_TRAIN_STEPS=10 OUTPUT_DIR=/root/autodl-fs/wam/runs/my_v4_run \
  bash /root/driftwam/train_v4/run_flashwam500_clean50.sh
```

For a pre-existing cache, set `PRECOMPUTE_TEACHER_SIGNATURES=0` to make a
missing or incomplete cache fail immediately. Resume requires
`ALLOW_RESUME=1`, either `RESUME_FROM_PATH` or `RESUME_FROM_STEP`, and the
same `OUTPUT_DIR` as the original run. The saved `action_supervision_source`
must be `ground_truth`; older teacher-pair runs cannot be resumed as this
objective. Their weights can still be selected explicitly as initialization.
This explicit gate prevents stale
environment variables from silently turning a fresh run into a resume.

Checkpoints default to every **50 optimizer steps**, retaining only the latest
complete checkpoint (online student, EMA, optimizer and per-rank RNG state).
At each save boundary rank 0 **deletes the previous checkpoint first**, then all
ranks save the current in-memory state to a temporary directory and publish its
completion markers. Old directories are renamed out of the resumable namespace
before deletion. A deletion error prevents the new write from starting.

This user-selected disk policy creates a recovery gap: if the process or new
save fails after deletion, no previous checkpoint is guaranteed to remain.
Both deletion and checkpoint writing pause optimizer updates. Logs record the
deletion time separately and the total time for deletion plus saving.

Before loading models, disk preflight counts FP32 tensor payloads plus 8 GiB of
headroom. With retention set to one, a fresh run requires about **83.8 GiB** free
for the current 5B model (75.8 GiB checkpoint + 8 GiB headroom). On resume it also
credits eligible old checkpoint files that will be deleted before replacement.
Saving more often does not multiply this bounded peak. Insufficient space fails
preflight. Unpublished temporary saves are cleaned up on handled write errors.

## Execution-response modes

The action execution objective can be switched without changing the offline
teacher-action cache:

```bash
# Current kernel-only baseline.
EXECUTION_RESPONSE_MODE=off bash /root/driftwam/train_v4/smoke_test.sh

# Reuse all K x M differentiable responses; fastest if it fits in memory.
EXECUTION_RESPONSE_MODE=reuse_all bash /root/driftwam/train_v4/smoke_test.sh

# Detached K x M kernel probes, then recompute only K selected responses.
EXECUTION_RESPONSE_MODE=recompute_selected \
  bash /root/driftwam/train_v4/smoke_test.sh
```

Set `EXECUTION_LOSS_WEIGHT` to change the auxiliary velocity-MSE multiplier.
Both enabled modes use the same argmax teacher matching, valid action mask,
noise, timestep, and teacher-response targets.
In `recompute_selected`, small contexts evaluate the selected candidates in
one batch, without activation recomputation. The BF16-equivalent context-token
budget is 9792, corresponding to four candidates with 16 history frames plus
two current frames at the formal 120-video/16-action-token layout. Larger
contexts evaluate one candidate at a time and recompute each response's entire
cache lifecycle during backward. Both paths preserve the same matching,
noise, history, targets and gradients (up to floating-point batching roundoff).
This keeps the longer-history memory safeguard while avoiding unnecessary
recomputation on the common short histories.
