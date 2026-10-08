#!/bin/bash
# Inner-fold phase of MODEL_V3_DESIGN_2026-10-02 (sha256 6843bb9613b99ebb7310657e95cd46faf290c6ce69939241651aa12e2705ec70),
# with the operational additions of MODEL_V3_ADDENDUM_2026-10-02.md.
# 27 runs: arms R, C, S x folds 0-2 x seeds 0-2, all on the EngageNet training split. No official split is opened.
#
# Survives a reboot: every run checkpoints after each epoch (--resume) and continues from its last completed epoch;
# a finished run (result file present) is skipped; an @reboot cron entry restarts this script, and a lock keeps a
# second copy from running. The cron entry removes itself when the phase is done.
# Waits for free GPU memory before each run, so it never squeezes a job that is already on the card.
#
# Order: smoke test (S on 64 clips, killed after its first checkpoint and resumed), then S fold 0 seed 0, then C,
# then R, then the remaining S runs, then the gate, which is computed once.
# usage: nohup setsid bash code/run_v3.sh > /dev/null 2>&1 &
set -u
R=~/engagenet_p2_crossdomain; cd $R
mkdir -p state logs
exec 9> state/v3.lock
flock -n 9 || { echo "[$(date -u '+%F %T') UTC] another v3 queue holds the lock; exiting" >> logs/v3.log; exit 0; }
PY=${PY:-$HOME/venvs/engagenet/bin/python}   # the interpreter of the v1 and v2 campaigns (addendum, item 1)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
LOG=logs/v3.log
log(){ echo "[$(date -u '+%F %T') UTC] $*" | tee -a $LOG; }
NEED_MB=16000
[ -f state/v3_batch ] && read SB SA PEAK NEED_MB < state/v3_batch

wait_gpu(){   # until the card has NEED_MB free
  local free said=0
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    [ "$free" -ge "$NEED_MB" ] && { [ $said = 1 ] && log "GPU free: $free MiB"; return 0; }
    [ $said = 0 ] && log "waiting for GPU memory: $free MiB free, need $NEED_MB"; said=1
    sleep 300
  done
}

COMMON="--backbone svfap --dataset engagenet --train-splits train --cv-folds 3 --epochs 12 --warmup 2 --lr 1e-3 --layer-decay 0.75 --lora 16 --frames 16 --ema 0.999 --tta-windows 5 --resume"

smoke(){   # checks memory, the segment path, and kill-then-resume, on 64 training clips of fold 0
  [ -f state/v3_smoke.done ] && return 0
  local cfg b acc tag sl pid peak
  for cfg in "4 8" "2 16"; do
    set -- $cfg; b=$1; acc=$2; tag=v3smoke_S_b$b; sl=logs/v3_smoke_$tag.log
    wait_gpu
    if [ -f ckpt/ft_${tag}_train.pt ] || [ -f results/FT_${tag}_train.json ]; then
      log "smoke $tag: first leg already done before a restart"
    else
      log "smoke $tag, first leg (batch $b x accum $acc)"
      setsid $PY code/ft_video.py ${COMMON/--epochs 12 --warmup 2/--epochs 2 --warmup 1} --cv-fold 0 --stride 4 --segments 5 \
          --batch $b --accum $acc --seed 0 --max-train 64 --tag $tag >> $sl 2>&1 &
      pid=$!
      while kill -0 $pid 2>/dev/null && ! grep -q "ckpt saved epoch 0" $sl; do sleep 15; done
      if kill -0 $pid 2>/dev/null; then
        kill -TERM -- -$pid; wait $pid 2>/dev/null; log "smoke $tag: killed after the epoch-0 checkpoint"
      else
        wait $pid
        if grep -qi "out of memory" $sl; then log "smoke $tag: out of memory, trying a smaller batch"; continue; fi
        log "smoke $tag: first leg ended before a checkpoint (see $sl)"; return 1
      fi
    fi
    log "smoke $tag, second leg (resume)"
    $PY code/ft_video.py ${COMMON/--epochs 12 --warmup 2/--epochs 2 --warmup 1} --cv-fold 0 --stride 4 --segments 5 \
        --batch $b --accum $acc --seed 0 --max-train 64 --tag $tag >> $sl 2>&1
    if grep -q "continuing at epoch 1 of 2" $sl && [ -f results/FT_${tag}_train.json ]; then
      peak=$(grep -o "peak_mem_gb [0-9.]*" $sl | awk '{print $2}' | sort -g | tail -1)
      NEED_MB=$(awk -v p=$peak 'BEGIN{n=int(p*1024)+3000; print (n<10000?10000:n)}')
      SB=$b; SA=$acc; echo "$SB $SA $peak $NEED_MB" > state/v3_batch; touch state/v3_smoke.done
      log "smoke passed: S batch $SB x accum $SA, peak $peak GB, runs now wait for $NEED_MB MiB free"
      return 0
    fi
    log "smoke $tag: the resume check failed (see $sl)"; return 1
  done
  log "smoke: out of memory at every batch tried"; return 1
}

run(){   # arm fold seed
  local arm=$1 f=$2 s=$3 stride seg b acc
  case $arm in
    R) stride=4;  seg=0; b=16;  acc=2 ;;
    C) stride=20; seg=0; b=16;  acc=2 ;;
    S) stride=4;  seg=5; b=$SB; acc=$SA ;;
  esac
  local tag=v3_${arm}_f${f}_s${s}
  [ -f results/FT_${tag}_train.json ] && return 0
  wait_gpu
  log "start $tag stride=$stride segments=$seg batch=$b accum=$acc$([ -f ckpt/ft_${tag}_train.pt ] && echo ' (resuming from checkpoint)')"
  $PY code/ft_video.py $COMMON --cv-fold $f --stride $stride --segments $seg --batch $b --accum $acc --seed $s --tag $tag >> $LOG 2>&1
  log "EXIT $tag $?"
}

log "=== v3 queue (re)start; python $($PY -c 'import sys,torch;print(sys.version.split()[0], torch.__version__)')"
log "design sha256 $(sha256sum MODEL_V3_DESIGN_2026-10-02.md | cut -c1-64)  ft_video sha256 $(sha256sum code/ft_video.py | cut -c1-64)"
smoke || { log "=== v3 queue stopped: smoke test failed"; exit 1; }
run S 0 0
for f in 0 1 2; do for s in 0 1 2; do run C $f $s; done; done
for f in 0 1 2; do for s in 0 1 2; do run R $f $s; done; done
for f in 0 1 2; do for s in 0 1 2; do run S $f $s; done; done
$PY code/gate_v3.py >> $LOG 2>&1
crontab -l 2>/dev/null | grep -v "code/run_v3.sh" | crontab -
log "=== v3 inner-fold phase done; @reboot entry removed"
