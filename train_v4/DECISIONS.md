# Train v4: Flash-WAM-Style Action Objective Extension

V4 retains the v3 video drifting and action-execution objectives and adds a
Flash-WAM-style action objective: two-timestep action consistency plus a small
native action flow-matching regularizer. For each sampled source, the full GT
video and its aligned GT action sequence are taken from the native latent
dataset. Action positions start at zero and no separate sampled-chunk history
is prepended. The frozen teacher advances the noisy action from the sampled start noise level to
the paired end level; the online student prediction at the start is matched to
the frozen EMA target prediction at the end in `x0` space using pseudo-Huber
loss. Only the 16 used RoboTwin channels and valid positions contribute.

The fixed action settings are two DDIM levels (stride 500), pseudo-Huber
constant `0.001`, consistency weight `1.0`, and flow-matching weight `0.01`.
The flow target is the native `noise - clean_action` velocity and reuses the
same online-student forward. Both action losses use the detached GT pair;
current GT video does not enter student video generation, video drifting,
or execution (the observed initial frame remains a valid condition).
Teacher-generated video/action pairs remain the kernel/execution probes.
The action graph is built only after video and execution backward release
their graphs. The video branch still samples causal chunks; the action branch
supervises the full source on every microbatch. Source ordering and the overall
video objective remain those of v4, so this is action-protocol parity, not a
claim that the entire experiment reproduces Flash-WAM.

The fixed `action_supervision_source=ground_truth` is saved in run configs,
manifest, and trainer state. Resume rejects missing or different supervision
sources; old teacher-pair weights may only be used as explicit initialization.

## Action parity audit (2026-09-13)

GT actions, Gaussian noise, noisy action state, and the FM target remain FP32;
only the private packed-forward inputs are cast to the model compute dtype.
Euler and x0 velocity increments use the velocity dtype as in Flash-WAM's
StepMixin. Intermediate Euler states are not remasked by the loss-validity
mask: inactive channels may affect EMA outputs. The initial padded action frame
is normalized by the upstream dataset rules, receives ordinary noise and
sampled timesteps, and contributes wherever the native validity mask is true.
Neither v4's action training data nor the local Flash-WAM training dataset clips
normalized GT actions. The video/probe branch preserves its existing clipped
historical action condition independently; changing that cache-backed condition
would alter a separate objective.

The regression test executes the pinned upstream DataMixin and StepMixin
(commit `5b8df13e9db24fb15ce42ff5ccc60a4015195960`) with the repaired native
attention. It covers a full five-frame sequence, the initial action frame,
a partial final chunk, and GT values outside [-1.5, 1.5]. CPU FP32/BF16 and CUDA
BF16 action consistency/FM values and combined parameter gradients agree within
reduction roundoff. A dataset parity check also compares upstream action
normalization/padding and masks. This does not claim parity with upstream's
leaking attention, input-dict mutation, or BF16 parameter accumulation; those
correctness and precision repairs are shared with the local baseline.

The pre-existing offline action-probe generator masks inactive channels after
each denoising step; the upstream server masks them after the rollout. These
cached probes belong to the drifting/execution definition and are unchanged;
they must not be presented as bitwise upstream server rollouts.

The numerical contract is `action_objective=gt_fp32_full_sequence_unclipped_v3`;
strict resume rejects checkpoints predating this correction. Long full action
sequences trigger gradient synchronization even when the sampled video history
is short, preventing deferred full gradients from exhausting GPU memory. This
changes communication scheduling, not loss weights or the effective batch.

## Inherited Train v3: Action-Response-Guided Video Drifting

The following describes the video objective inherited and revised in `train_v4`.
The objective is
task-success-oriented video distillation. Matching every teacher mode is not a
requirement; teacher diversity supplies candidate behaviors from which the
kernel selects locally compatible attraction and repulsion directions.

## Candidates and paired teacher actions

For one condition, `M` is the number of teacher videos and `K` is the number
of student videos. The fixed experiment uses `M = K = 4`. Teacher video
`T_m`, where `m` is the teacher index, is paired with action `a_m`. The action
is generated once using the frozen LingBot-VA weights with the local 50-step
probe sampler and stored
in the teacher-signature cache:

```text
a_m = G_A^50(T_m, condition)
```

The cache is an offline data artifact, not a second distilled model. Formal
training requires a complete cache and never computes a 50-step action online.

Student video `S_k`, where `k` is the student index, is generated exactly once
per training micro-batch. No student action signature is generated.

## One-step action-response geometry

For every micro-batch, `epsilon_A` is newly sampled Gaussian action noise and
`t_f` is the newly sampled action diffusion timestep for latent frame `f`.
They are shared across all teacher probes and all video candidates within the
same condition. For each paired teacher action:

```text
a_tilde_m = AddNoise(a_m, epsilon_A, t)
r_T_m     = v_A(a_tilde_m, t | T_m, condition)
r_S_km    = v_A(a_tilde_m, t | S_k, condition)
```

Here `v_A` is the frozen LingBot-VA action-velocity predictor. A video-context
forward only inserts the already generated `T_m` or `S_k` into its causal KV
context; it does not regenerate a video.

Let `Omega_A` be the valid elements from the native RoboTwin 16-channel mask
and the data validity mask. The positive action-response distance is

```text
D_A_ST(k,m) = mean over Omega_A of (r_S_km - r_T_m)^2.
```

First compute the positive kernel weights `w_pos` defined below. Let `m_j`
be student `j`'s detached maximum-weight teacher. The negative distance uses
that compared candidate's probe on both endpoints:

```text
m_j = argmax_m stop_gradient(w_pos(j,m))
D_A_SS(k,j) = mean over Omega_A of
              (r_S(k,m_j) - r_S(j,m_j))^2.
```

This replaces the former average over all `M` probes. It uses `m_j`, not the
query student's `m_k`, and never compares endpoints under different probes.
The positive formula is unchanged, and all required responses already exist.
Matching is also computed when execution is off; when execution is enabled,
it reuses the same matching. Positive weights do not depend on repulsion, so
there is no circular definition.

The negative action distance is generally directed: `D_A_SS(k,j)` need not
equal `D_A_SS(j,k)`. Action and joint-distance diagnostics include both
non-self directions; symmetric video-distance statistics remain unchanged.
These detached distances are normalized-velocity MSE values, not separate
optimized losses. They do not decode physical actions or run a trajectory.

## Video geometry and kernel

Let `D_V_ST(k,m)` and `D_V_SS(k,j)` be the existing normalized layer-19 video
feature distances. Let `h_V` and `h_A` be fixed positive video and action
bandwidths. The positive teacher weights are

```text
w_pos(k,m) = softmax_m(
    -D_V_ST(k,m) / h_V - D_A_ST(k,m) / h_A
).
```

The negative student weights exclude `j = k`:

```text
w_neg(k,j) = softmax_{j != k}(
    -D_V_SS(k,j) / h_V - D_A_SS(k,j) / h_A
).
```

Let `phi(T_m)` and `phi(S_k)` be normalized video features, and let `beta` be
the repulsion multiplier (`beta = 1` in the fixed experiment). The detached
drifting field is

```text
Delta_k = sum_m w_pos(k,m) * (phi(T_m) - phi(S_k))
          - beta * sum_{j != k} w_neg(k,j) * (phi(S_j) - phi(S_k)).
```

After batch-global RMS normalization of `Delta`, the base optimized loss is
the stopped-target video feature regression loss:

```text
L = MSE(phi(S), stop_gradient(phi(S) + Delta / RMS(Delta))).
```

Kernel action responses remain detached measurements. An optional hard-matched
execution objective uses the maximum positive-kernel teacher for each student:

```text
m_star(k) = argmax_m stop_gradient(w_pos(k,m))
L_exec = mean_k MSE_valid(r_S(k,m_star), stop_gradient(r_T(m_star)))
L = L_drift + lambda_exec * L_exec
```

`reuse_all` preserves the existing `K x M` student-response graphs.
`recompute_selected` keeps those probes detached and then recomputes only the
`K` selected student responses, while reusing noisy actions, timesteps, and
teacher targets. That execution branch itself has no physical-action decode or
multi-step rollout; the separate deferred v4 action objective supplies consistency and
flow-matching gradients after its graph is released. There is still no RL
post-training stage or 50-step online action rollout.

## Bandwidth calibration

Bandwidths are calibrated once unless explicitly supplied or restored from a
checkpoint. `h_V` is the median distinct teacher-teacher video distance. To
obtain `h_A`, the teacher videos are temporarily used as the comparison
candidate set and their responses are evaluated under all paired teacher-action
probes. For the directed pair `(i,j)`, compare `r_j(T_i)` with `r_j(T_j)`.
The median over all `i != j`, including both directions, gives `h_A`.
Both bandwidths then remain fixed for the run. This definition differs from
the former probe-averaged calibration; fresh runs calibrate the new geometry.
An old checkpoint's stored bandwidths are still restored unchanged by the
existing resume path, so resuming it is not an exact continuation of the old
objective after this geometry change.

## Fixed clean-50 experiment

- Teacher candidates: `M = 4`.
- Student candidates: `K = 4`.
- Student video steps: `1`.
- Offline teacher action steps: `50`.
- Action response steps per `(video, probe)`: `1`.
- Formal execution mode: `recompute_selected`.
- Execution-loss weight: `0.1`.
- Action-consistency weight: `1.0`.
- Action flow-matching weight: `0.01`.
- Video feature layer: post-block layer 19.
- Repulsion multiplier: `beta = 1`.
- Optimizer learning rate: `2e-6`.
- EMA decay: `0.995`.
- Formal profile: per-rank batch 1 and global batch 32; the script derives
  accumulation as `32 / NGPU` (default: 4 GPUs and accumulation 8).

The executable commands and shared-FS paths are recorded in `README.md`.

## Attention and persistent training precision

Packed teacher-forcing forwards carry explicit self/cross masks through every
block, including activation checkpoint recomputation. CUDA uses FlexAttention;
CPU uses a dense reference for small tests. Masked training is selected by the
forward call regardless of the checkpoint's inference attention backend.
Cached inference retains its existing key/value visibility.

Online and EMA students load FP32 master weights. FSDP uses the configured
forward dtype (BF16 by default), FP32 reduction, and FP32 timestep embedders.
Adam moments and EMA accumulation remain FP32. Full training checkpoints retain
FP32 online/EMA weights and optimizer states. Exact resume validates the new
numerics contract; older checkpoints can be used as explicit initialization,
not represented as exact continuations of the repaired training algorithm.
Checkpoint space is estimated from FP32 tensor sizes.

The teacher remains at its original precision. The exact model-source pair in
`INFERENCE_COMPATIBILITY.json` was checked for unchanged cached outputs and
video-input gradients with tiny native FP32/BF16 models. Only that source pair
shares the prior teacher-cache contract hash. Existing cache files/markers are
not rewritten; unknown source changes still invalidate the cache contract.

CPU regression checks cover forbidden labels, future/cross-sample isolation,
allowed current-video conditioning, padding, recomputation, cached equivalence,
and FP32 save/resume continuity. Actual CUDA FlexAttention, FSDP multi-GPU
execution, and full-model peak memory still require a CUDA environment.

## Packed action activation checkpointing

Online and EMA blocks use the same PyTorch checkpoint wrapper structure before
FSDP sharding. Only gradient-enabled calls with an explicit packed-training
attention mask recompute block activations during backward. Frozen EMA calls
and all cached video/action-response calls bypass recomputation. RNG state is
preserved. The explicit masks remain owned by each forward.

This changes activation storage and runtime, not loss definitions, batch size,
history length, or optimizer-step frequency. The existing adaptive gradient
synchronization threshold remains 24 latent frames pending GPU profiling.

## Short-context execution batching

Selected differentiable teacher responses use a direct batch when their
BF16-equivalent total context is at most 9792 tokens. With the formal B=1,
K=4 and 120 video + 16 action tokens/frame, this permits up to 16 history
frames. Larger contexts retain per-candidate checkpointing over the complete
cache lifecycle. The budget scales with candidate count, spatial layout and
input element size; it is a memory execution rule, not a loss hyperparameter.

The current formal seed42 schedule puts 1895 of 2000 microbatches (94.75%)
within this short-history range. Both execution paths use the same causal
history, selected teacher probes and loss denominator. The native CPU FP32
and CUDA BF16 tests force each path and compare outputs and input-video
gradients with the batched all-probe reference. Frozen teacher parameters
remain gradient-free and caches are cleared before and after backward.

## Sparse mask compiler repair (2026-09-13)

CUDA regression exposed corrupt sparse block counts after successive compiled
mask builds in PyTorch 2.9. This can corrupt key/value gradients while forward
outputs still match a dense causal reference. Copying forward indices and
rebuilding their transpose fixed the one-block case but failed a multi-block
shape transition, so the final implementation builds BlockMask eagerly.
The FlexAttention computation remains compiled; causal edges and cached
inference are unchanged. Checks cover block counts, Q/K/V gradients, two native
implementations, multiple blocks, padding, batch isolation, and shape changes.
The shared numerical contract is `training_attention=explicit_causal_eager_mask_v2`.

Full action sequence lengths differ across ranks. The long-sequence gradient
synchronization decision uses an all-rank reduction, so every rank follows the
same collectives even when its local video history is short.
