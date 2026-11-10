#!/bin/bash
# A1 final-turn on Core1 GPU 5 ONLY (deviation D-5). PHASE=dev runs the two dev-family checks; PHASE=main runs the
# three locked runs in the protocol order (requires protocols/EXPA_FINALTURN.lock.json committed). REL/GIT/SUMS/STAGER
# must be exported for the GPU5-guard release.
# Each run: prepare -> stage -> serve -> publisher -> wait ready -> runner(s) -> wait end -> stop-serve -> final-publish.
# Refuses to start unless both locks are committed. Stops the chain on any failure (no retries, no other GPU).
L="C:/Users/Hp/AppData/Local/Temp/session/C--Users-Hp/8f7e53e4-ce23-4904-aa51-7d9cd5eff1f9/scratchpad/L.sh"
REPO="D:/C_moved/Documents-work/2026-10-17/yaml-google-the-gemma-4-developer/outputs/absent-edges"
X=/mnt/kldata/absent-edges-20261018-core2/expA
G=GPU-7b8d9f89-0a8f-2d07-4833-bee70a0eeed2; PORT=8145; INST=g5; IDX=5
: "${REL:?}" "${GIT:?}" "${SUMS:?}" "${STAGER:?}" "${PHASE:?}"
CK="--checkouts-manifest /mnt/kldata/absent-edges-expA-public-20261101/repos-manifests/clean-checkouts.json --checkouts-manifest-sha256 58363dca69f434d14f6da8ed898fd46d367776bf2f183c80aa90610231163035"
MAIN_D="--dispatch-dir $X/prod-items-0b8ee0b7c/dispatch-main --dispatch-manifest-sha256 f21129cb717120ca352674840e93b400b51ef29429b965f2a9a36f7819b7b6b1"
DEV_D="--dispatch-dir $X/prod-items-0b8ee0b7c/dispatch-dev --dispatch-manifest-sha256 c8bac8a1dcb872c4736ce0701d50dbf5e6e9931cc5189e7342c690dcffa4e9be"
HARD_UNUSED="--dispatch-dir $X/hard-panel-20261102/dispatch-hard --dispatch-manifest-sha256 df4b54eab1635542e83391f1d5e50dec10202c656ec00294ec511b3abfcced37"
log() { echo "[$(date -u +%FT%TZ)] $*"; }

for f in $( [ "$PHASE" = main ] && echo protocols/EXPA_FINALTURN.lock.json ); do
  git -C "$REPO" ls-files --error-unmatch "$f" > /dev/null 2>&1 || { log "REFUSE: $f is not committed"; exit 2; }
done

wait_ready() {  # $1 run
  for i in $(seq 1 60); do
    r=$(timeout 60 ssh -o BatchMode=yes ai-core-team-1 "curl -s --max-time 3 http://127.0.0.1:$PORT/v1/models | grep -c absent-edges-gemma4-31b-bf16; ls /dev/shm/ae-expA-$1/serve/*/records/ 2>/dev/null | grep -c -E 'exited|collision|refus'")
    set -- $1 $r
    [ "${3:-0}" != 0 ] && { log "SERVE FAILURE records for $1"; return 1; }
    [ "${2:-0}" -ge 1 ] && { log "ready $1"; return 0; }
    sleep 60
  done
  log "NOT READY $1"; return 1
}
wait_runner_end() {  # $1 run
  for i in $(seq 1 1000); do
    sleep 120
    s=$(timeout 60 ssh -o BatchMode=yes ai-core-team-1 "systemctl is-active ae-expA-runner-$1 2>/dev/null; find /dev/shm/ae-expA-$1/run/cells -name result.json 2>/dev/null | wc -l")
    set -- $1 $s
    [ $((i % 15)) = 1 ] && log "progress $1 state=$2 results=$3"
    if [ "$2" != active ] && [ "$2" != activating ]; then log "runner ended $1 results=$3"; return 0; fi
  done
  log "TIMEOUT waiting for $1"; return 1
}
one_run() {  # $1 run id, $2 dispatch args, then runner arg sets separated by ";"
  run=$1; disp=$2; shift 2
  log "=== $run"
  timeout 120 ssh -o BatchMode=yes ai-core-team-1 "nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader | grep "^$IDX,""
  step() {  # run one launch step, show its last lines, stop on a non-zero exit
    out=$(bash "$L" "$@" 2>&1); rc=$?
    echo "$out" | tail -2
    [ $rc = 0 ] || { log "STEP FAILED rc=$rc: $1 $run"; return 1; }
  }
  step prepare --run $run --run-id $run --git-sha $GIT --release-dir $X/releases/$REL --release-sums-sha256 $SUMS     --stager-sha256 $STAGER $CK --concurrency 8 $disp --instance $INST=$G=$PORT || return 1
  step stage --run $run || return 1
  step serve --run $run || return 1
  step publisher --run $run || return 1
  wait_ready $run || return 1
  IFS=';' read -ra SETS <<< "$*"
  for sel in "${SETS[@]}"; do
    log "runner $run $sel"
    step runner --run $run $sel || return 1
    wait_runner_end $run || return 1
  done
  step stop-serve --run $run || return 1
  step final-publish --run $run || return 1
  log "=== done $run"
}

if [ "$PHASE" = dev ]; then
  one_run finalturn-dev-neutral-20261104a "$DEV_D" "--blocks P --seeds 11 --final-turn-policy neutral" || { log A1DEV_STOPPED; exit 1; }
  one_run finalturn-dev-deadline-20261104a "$DEV_D" "--blocks P --seeds 11 --final-turn-policy deadline" || { log A1DEV_STOPPED; exit 1; }
  log A1DEV_COMPLETE
elif [ "$PHASE" = main ]; then
  one_run finalturn-neutral-20261102a "$MAIN_D" "--blocks P --seeds 11 --final-turn-policy neutral" || { log A1_STOPPED; exit 1; }
  one_run finalturn-deadline-20261102a "$MAIN_D" "--blocks P --seeds 11 --final-turn-policy deadline" || { log A1_STOPPED; exit 1; }
  one_run finalturn-none-20261102a "$MAIN_D" "--blocks P --seeds 11" || { log A1_STOPPED; exit 1; }
  log A1_COMPLETE
fi
