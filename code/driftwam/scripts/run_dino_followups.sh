#!/usr/bin/env bash
# Queue the follow-up passes behind the main DINOv3 extraction so the GPU never
# idles. Stage 1 answers whether averaging the four pixel frames of each latent
# frame is what limits the measured dispersion; stage 2 re-tests H7 (does the
# distilled student keep its sensitivity to epsilon) in the same feature space.
set -u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate lingbot-distill
cd /root/autodl-tmp/wam/code/driftwam
export HF_HUB_OFFLINE=1
LOG=/root/autodl-fs/wam/exp/exp01/logs

MAIN_PID=${1:-}
if [ -n "$MAIN_PID" ]; then
  echo "waiting for main extraction (pid $MAIN_PID)"
  while kill -0 "$MAIN_PID" 2>/dev/null; do sleep 20; done
  echo "main extraction finished"
fi

echo "=== STAGE 1/2: unpooled features, same 6 items (paired pooling check) ==="
python -u scripts/extract_dino.py --items 6 --pool none \
  --out /root/autodl-fs/wam/exp/exp01/dino/teacher_cfg5_F8_K8_nopool \
  > "$LOG/dino_nopool.log" 2>&1
echo "stage 1 exit $?"

echo "=== STAGE 2/2: Flash-WAM student in DINOv3 space (H7 re-test) ==="
python -u scripts/extract_dino.py --items 20 \
  --samples /root/autodl-fs/wam/exp/exp01/samples/student_cfg5_F8_K8 \
  > "$LOG/dino_student.log" 2>&1
echo "stage 2 exit $?"

echo "ALL FOLLOW-UPS COMPLETE"
