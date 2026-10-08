#!/bin/bash
# Inner-fold phase of MODEL_V5_DESIGN_2026-10-05 (sha256 c13e32416f4df296273040b4692d73d1f91695761bf3c942a2a804de90103994).
# Per fold: continued MAE pretraining at the CMS window on that fold's training clips (pretrain_v5.py), then the arm-C
# fine-tuning recipe of v3 from the adapted encoder, seeds 0-2. The control is v3 arm C, not rerun. No official split opened.
#
# Survives a reboot: pretraining and fine-tuning checkpoint every epoch and continue; finished steps are skipped; an @reboot
# cron entry restarts this script; a lock prevents a second copy; the entry removes itself at the end.
# Waits for free GPU memory before every step.
# usage: nohup setsid bash code/run_v5.sh > /dev/null 2>&1 &
set -u
R=~/engagenet_p2_crossdomain; cd $R
mkdir -p state logs
exec 9> state/v5.lock
flock -n 9 || { echo "[$(date -u '+%F %T') UTC] another v5 queue holds the lock; exiting" >> logs/v5.log; exit 0; }
PY=${PY:-$HOME/venvs/engagenet/bin/python}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LOG=logs/v5.log
log(){ echo "[$(date -u '+%F %T') UTC] $*" | tee -a $LOG; }
NEED_MB=16000; PB=32; PA=1
[ -f state/v5_batch ] && read PB PA PEAK NEED_MB < state/v5_batch

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

smoke(){   # pretraining: 640 clips, 2 epochs, killed after the first checkpoint and resumed; then 1 fine-tuning epoch from it
  # (64 clips finished both epochs in 5 s, before the first kill could land: addendum MODEL_V5_ADDENDUM_2026-10-05.md)
  [ -f state/v5_smoke.done ] && return 0
  local cfg b acc tag sl pid peak
  for cfg in "32 1" "16 2"; do
    set -- $cfg; b=$1; acc=$2; tag=v5smoke2_pre_b$b; sl=logs/v5_smoke_$tag.log
    wait_gpu
    if [ -f ckpt/pre_${tag}.pt ] || [ -f results/PRE_${tag}.json ]; then
      log "smoke $tag: first leg already done before a restart"
    else
      log "smoke $tag, first leg (batch $b x accum $acc)"
      setsid $PY code/pretrain_v5.py --cv-folds 3 --cv-fold 0 --epochs 2 --warmup 1 --batch $b --accum $acc --max-train 640 --tag $tag >> $sl 2>&1 &
      pid=$!
      while kill -0 $pid 2>/dev/null && ! grep -q "ckpt saved epoch 0" $sl; do sleep 2; done
      if kill -0 $pid 2>/dev/null; then
        kill -TERM -- -$pid; wait $pid 2>/dev/null; log "smoke $tag: killed after the epoch-0 checkpoint"
      else
        wait $pid
        if grep -qi "out of memory" $sl; then log "smoke $tag: out of memory, trying a smaller batch"; continue; fi
        log "smoke $tag: first leg ended before a checkpoint (see $sl)"; return 1
      fi
    fi
    [ -f results/PRE_${tag}.json ] || $PY code/pretrain_v5.py --cv-folds 3 --cv-fold 0 --epochs 2 --warmup 1 --batch $b --accum $acc --max-train 640 --tag $tag >> $sl 2>&1
    if ! { grep -q "continuing at epoch 1 of 2" $sl && [ -f results/PRE_${tag}.json ]; }; then log "smoke $tag: the resume check failed (see $sl)"; return 1; fi
    log "smoke $tag: one fine-tuning epoch from the adapted encoder"
    $PY code/ft_video.py ${FT/--epochs 12 --warmup 2/--epochs 1 --warmup 0} --cv-fold 0 --seed 0 --max-train 64 \
        --init-ckpt $R/weights/svfap_v5/$tag.pth --tag v5smoke2_ft_b$b >> $sl 2>&1
    if ! grep -q "loaded svfap: missing 4" $sl || ! grep -q "FT_DONE v5smoke2_ft_b$b" $sl; then log "smoke: fine-tuning from the adapted encoder failed (see $sl)"; return 1; fi
    peak=$(grep -o "peak_mem_gb [0-9.]*" $sl | awk '{print $2}' | sort -g | tail -1)
    NEED_MB=$(awk -v p=$peak 'BEGIN{n=int(p*1024)+3000; print (n<10000?10000:n)}')
    PB=$b; PA=$acc; echo "$PB $PA $peak $NEED_MB" > state/v5_batch; touch state/v5_smoke.done
    log "smoke passed: pretraining batch $PB x accum $PA, peak $peak GB, steps now wait for $NEED_MB MiB free"
    return 0
  done
  log "smoke: out of memory at every batch tried"; return 1
}

pretrain(){   # fold
  local f=$1 tag=v5_pre_f$1
  [ -f results/PRE_${tag}.json ] && return 0
  wait_gpu
  log "start $tag$([ -f ckpt/pre_${tag}.pt ] && echo ' (resuming from checkpoint)')"
  $PY code/pretrain_v5.py --cv-folds 3 --cv-fold $f --batch $PB --accum $PA --tag $tag >> $LOG 2>&1
  log "EXIT $tag $?"
}

finetune(){   # fold seed
  local f=$1 s=$2 tag=v5_P_f$1_s$2
  [ -f results/FT_${tag}_train.json ] && return 0
  [ -f results/PRE_v5_pre_f${f}.json ] || { log "skip $tag: the fold-$f pretraining has not finished"; return 1; }
  wait_gpu
  log "start $tag$([ -f ckpt/ft_${tag}_train.pt ] && echo ' (resuming from checkpoint)')"
  $PY code/ft_video.py $FT --cv-fold $f --seed $s --init-ckpt $R/weights/svfap_v5/v5_pre_f$f.pth --tag $tag >> $LOG 2>&1
  log "EXIT $tag $?"
}

log "=== v5 queue (re)start; python $($PY -c 'import sys,torch;print(sys.version.split()[0], torch.__version__)')"
log "design sha256 $(sha256sum MODEL_V5_DESIGN_2026-10-05.md | cut -c1-64)  pretrain_v5 $(sha256sum code/pretrain_v5.py | cut -c1-64)  ft_video $(sha256sum code/ft_video.py | cut -c1-64)"
smoke || { log "=== v5 queue stopped: smoke test failed"; exit 1; }
for f in 0 1 2; do
  pretrain $f
  for s in 0 1 2; do finetune $f $s; done
done
$PY code/gate_v5.py >> $LOG 2>&1
crontab -l 2>/dev/null | grep -v "code/run_v5.sh" | crontab -
log "=== v5 inner-fold phase done; @reboot entry removed"
