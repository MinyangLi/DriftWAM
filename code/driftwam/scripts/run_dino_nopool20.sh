#!/usr/bin/env bash
# The paired check showed temporal pooling halves R, and the unpooled form is what
# a per-frame feature loss actually sees, so the headline number has to come from a
# full unpooled pass rather than a 6-item extrapolation.
set -u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lingbot-distill
cd /root/autodl-tmp/wam/code/driftwam
export HF_HUB_OFFLINE=1
PREV=${1:-}
if [ -n "$PREV" ]; then
  echo "waiting for pid $PREV"
  while kill -0 "$PREV" 2>/dev/null; do sleep 20; done
fi
echo "=== unpooled, all 20 items ==="
python -u scripts/extract_dino.py --items 20 --pool none \
  --out /root/autodl-fs/wam/exp/exp01/dino/teacher_cfg5_F8_K8_nopool \
  > /root/autodl-fs/wam/exp/exp01/logs/dino_nopool20.log 2>&1
echo "exit $?"
