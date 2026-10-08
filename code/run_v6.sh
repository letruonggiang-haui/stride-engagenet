#!/bin/bash
# Confirmation round of v5, MODEL_V6_DESIGN_2026-10-06 (sha256 0b65307dc848ebe704000662c1081ebf9f8d74df1e0412a16d8788115899307c).
# Per fold: a new continued pretraining (seed 1), then fine-tuning seeds 3-5 for the control C and the proposed model P.
# Code unchanged since v5, whose smoke test passed, so no smoke test here. No official split is opened.
#
# Survives a reboot: pretraining and fine-tuning checkpoint every epoch and continue; finished steps are skipped; an @reboot
# cron entry restarts this script; a lock prevents a second copy; the entry removes itself at the end.
# usage: nohup setsid bash code/run_v6.sh > /dev/null 2>&1 &
set -u
R=~/engagenet_p2_crossdomain; cd $R
mkdir -p state logs
exec 9> state/v6.lock
flock -n 9 || { echo "[$(date -u '+%F %T') UTC] another v6 queue holds the lock; exiting" >> logs/v6.log; exit 0; }
PY=${PY:-$HOME/venvs/engagenet/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LOG=logs/v6.log
log(){ echo "[$(date -u '+%F %T') UTC] $*" | tee -a $LOG; }
read PB PA PEAK NEED_MB < state/v5_batch   # measured by the v5 smoke test: batch 32 x 1, peak 8.75 GB

wait_gpu(){
  local free said=0
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge "$NEED_MB" ] && { [ $said = 1 ] && log "GPU free: $free MiB"; return 0; }
    [ $said = 0 ] && log "waiting for GPU memory: $free MiB free, need $NEED_MB"; said=1
    sleep 300
  done
}

FT="--backbone svfap --dataset engagenet --train-splits train --cv-folds 3 --epochs 12 --warmup 2 --lr 1e-3 --layer-decay 0.75 --lora 16 --frames 16 --stride 20 --segments 0 --batch 16 --accum 2 --ema 0.999 --tta-windows 5 --resume"

pretrain(){   # fold
  local f=$1 tag=v6_pre_f$1
  [ -f results/PRE_${tag}.json ] && return 0
  wait_gpu
  log "start $tag$([ -f ckpt/pre_${tag}.pt ] && echo ' (resuming from checkpoint)')"
  $PY code/pretrain_v5.py --cv-folds 3 --cv-fold $f --batch $PB --accum $PA --seed 1 --tag $tag >> $LOG 2>&1
  log "EXIT $tag $?"
}

finetune(){   # arm fold seed
  local arm=$1 f=$2 s=$3 tag=v6_$1_f$2_s$3 init=""
  [ -f results/FT_${tag}_train.json ] && return 0
  if [ $arm = P ]; then
    [ -f results/PRE_v6_pre_f${f}.json ] || { log "skip $tag: the fold-$f pretraining has not finished"; return 1; }
    init="--init-ckpt $R/weights/svfap_v5/v6_pre_f$f.pth"
  fi
  wait_gpu
  log "start $tag$([ -f ckpt/ft_${tag}_train.pt ] && echo ' (resuming from checkpoint)')"
  $PY code/ft_video.py $FT --cv-fold $f --seed $s $init --tag $tag >> $LOG 2>&1
  log "EXIT $tag $?"
}

log "=== v6 queue (re)start; python $($PY -c 'import sys,torch;print(sys.version.split()[0], torch.__version__)')"
log "design sha256 $(sha256sum MODEL_V6_DESIGN_2026-10-06.md | cut -c1-64)  pretrain_v5 $(sha256sum code/pretrain_v5.py | cut -c1-64)  ft_video $(sha256sum code/ft_video.py | cut -c1-64)"
for f in 0 1 2; do
  pretrain $f
  for s in 3 4 5; do finetune C $f $s; finetune P $f $s; done
done
$PY code/gate_v6.py >> $LOG 2>&1
crontab -l 2>/dev/null | grep -v "code/run_v6.sh" | crontab -
log "=== v6 phase done; @reboot entry removed"
