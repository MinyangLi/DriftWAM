#!/usr/bin/env bash
# H7 in DINOv3 space, on video_x0 (the raw prediction) and unpooled, matching the
# configuration the teacher's headline number uses.
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
python -u scripts/extract_dino.py --items 20 --pool none --field video_x0 \
  --samples /root/autodl-fs/wam/exp/exp01/samples/student_cfg5_F8_K8 \
  --out /root/autodl-fs/wam/exp/exp01/dino/student_x0_nopool \
  > /root/autodl-fs/wam/exp/exp01/logs/dino_student_x0.log 2>&1
echo "exit $?"
