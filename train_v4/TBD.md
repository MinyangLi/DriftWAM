# Train v4 Follow-ups

- Run a one-update GPU smoke test and record peak memory and step time.
- Confirm finite action-consistency and action flow-matching losses, plus a
  nonzero online-student gradient.

The notes below describe the inherited v3 baseline only; v4 adds action
consistency and native action flow matching as documented in `DECISIONS.md`.

## Inherited Train v3 Follow-ups

The default method is fixed in `DECISIONS.md`. Only evidence-driven follow-ups
remain:

- Measure kernel-weight entropy and teacher coverage during the one-step GPU
  smoke run; do not change bandwidths before observing a failure.
- If action-response cost remains excessive, batch multiple teacher probes in
  one frozen-teacher call without changing the distance definition.
- Evaluate task success only after the kernel statistics and training loss are
  numerically stable.

Compare `reuse_all` and `recompute_selected` peak memory and step time on the
same one-update GPU smoke. Physical-action decoding and the student-signature path are not part of
the default experiment; the v4 action objective is documented above.
