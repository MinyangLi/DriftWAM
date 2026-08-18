# driftwam

Diagnostics for drifting-based one-step distillation of **LingBot-VA** (RoboTwin).

Experiment 01 asks whether the teacher's conditional distribution `p(x | c)` is
spread out enough to support a drifting objective, or has collapsed to a Dirac
measure — in which case, by IDP Proposition 3.1, the repulsion term of explicit
drifting degenerates exactly into isotropic MSE. Video and action are judged
separately, because their conditional sharpness may differ by orders of magnitude.

Full protocol, hypotheses and decision table:
`/root/autodl-fs/exp01_teacher_conditional_variance.md`.

This package sits **beside** the Flash-WAM checkout and never writes into it.

## Read this first

Both published checkpoints ship `"attn_mode": "torch"` in
`transformer/config.json`, which selects `custom_sdpa` — plain
`scaled_dot_product_attention` with **no mask argument at all**. Only
`attn_mode="flex"` routes through `FlexAttnFunc`, the sole consumer of the
block-causal BlockMask that `init_mask` builds.

Loaded as shipped, `forward_train` therefore runs fully bidirectional attention:
noisy tokens attend to the clean ground truth of their own chunk, chunks are no
longer independent, and batch slots leak into each other. Always load through
`driftwam.models.load_transformer`, which forces `flex` and asserts the attention
op really is `FlexAttnFunc`.

## Layout

```
driftwam/
  paths.py      filesystem locations, every one overridable by env var
  bootstrap.py  puts Flash-WAM's wan_va/ and distillation/ on sys.path, in the
                one order that works, and returns its patched cfg
  shapes.py     item layout: T-shape camera crops, action channel semantics,
                denormalisation, chunk indexing
  models.py     checkpoint loading with the attn_mode override
  sampling.py   teacher-forced reverse-ODE sampler, one-step student, controls
  metrics.py    M1-M8 dispersion statistics and the V1-V3 validity controls
scripts/
  smoke_teacher.py   invariants + noise floor + throughput on the real teacher
tests/
  test_metrics.py      21 self-tests on data with known answers (CPU, seconds)
  test_forward_tiny.py all sampler invariants on a 2-layer random model
  diag_*.py            bisection tools kept from debugging the attn_mode issue
```

## Quick start

```bash
conda activate lingbot-distill
cd /root/autodl-tmp/wam/code/driftwam

python -m driftwam.paths                    # check every path resolves
python tests/test_metrics.py                # CPU only
python tests/test_forward_tiny.py           # GPU, tiny model, ~30 s
python scripts/smoke_teacher.py --full-ode  # GPU, real 5B teacher, ~2 min
```

`OMP_NUM_THREADS=0` in this container makes libgomp abort; `bootstrap` repairs it
before torch is imported, so import `driftwam.bootstrap` first in any script.

## Why sampling is built the way it is

Three properties of the block-causal mask, each verified bitwise on the real
teacher by `scripts/smoke_teacher.py`:

1. **Chunk independence** — each chunk denoises against its own true history, so
   one forward evaluates all `ceil(F/2)` conditions of an item at once.
2. **Modality decoupling** — noisy video and noisy action tokens cannot see each
   other, so the 25-step video grid and the 50-step action grid share 50
   conditional forwards instead of needing 75. The interleaved result is
   bit-identical to running the two branches separately.
3. **Batch isolation** — `seq_ids` separates batch elements in self- *and*
   cross-attention.

Cost: 50 conditional + 25 unconditional forwards = **75 per condition set**
(CFG applies to video only, since `action_guidance_scale = 1`).

**Batch ordering matters.** The same epsilon placed at a different batch position
produces a different result (~0.35% in bf16) because the block-sparse reduction
order changes, and that inflates measured variance — the direction that could make
a degenerate branch look healthy. So each item keeps a fixed batch slot and
epsilon varies across outer iterations. Same total FLOPs, no layout-induced floor.

## Two data-layout traps

* Action latent frame **0 is not data**: `_action_post_process` front-pads the raw
  stream by one full latent frame of zeros before normalising, so frame 0 holds a
  per-channel constant that is identical in every episode, yet `actions_mask`
  marks it valid. Use `shapes.action_frame_slice`.
* RoboTwin has many single-arm tasks where the idle arm's seven pose channels are
  bit-identical all episode, again flagged valid. Use
  `shapes.active_action_channels`, and rescale per channel with
  `ActionNorm.scale` before comparing across channels (one normalised unit is
  ~0.20 m of arm translation but 0.5 of gripper travel).

## Environment

See `requirements.txt`. `/root/autodl-tmp` survives shutdown but **not** an
instance change, so results are written to `/root/autodl-fs/wam/exp/exp01`.
