#!/bin/bash
# Exp A serving smoke/calibration on Core1 (synthetic prompts only; not research data).
# Usage: bash serve_smoke.sh <GPU-UUID> <port> start|probe|stop <run-name>
set -euo pipefail
Q=/mnt/kldata/absent-edges-expA-public-20261101
GPU=$1; PORT=$2; ACTION=$3; RUN=$4
W=/dev/shm/$RUN
GPU5=GPU-7b8d9f89-0a8f-2d07-4833-bee70a0eeed2
case "$ACTION" in
start)
  [ "$GPU" != "$GPU5" ] || { echo "REFUSE_GPU5"; exit 2; }
  used=$(nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits | awk -F', ' -v g="$GPU" '$1==g{print $2}')
  procs=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$GPU" || true)
  echo "preflight gpu=$GPU used_mib=$used procs=$procs"
  [ -n "$used" ] && [ "$used" -le 128 ] && [ "$procs" -eq 0 ] || { echo "PREFLIGHT_FAIL"; exit 3; }
  mkdir -p "$W/cache" "$W/logs" "$W/tmp"; chmod 0700 "$W"
  nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader > "$W/logs/preflight_gpus.csv"
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader > "$W/logs/preflight_apps.csv"
  sudo systemd-run --unit="$RUN" --property=User=thilak.arjun --property=Group=thilak.arjun \
    --property=KillMode=control-group --property=RuntimeMaxSec=7200 --property=WorkingDirectory="$W" \
    --property=NoNewPrivileges=yes --property=ProtectSystem=strict --property=ProtectHome=yes \
    --property=ReadWritePaths="$W" --property="InaccessiblePaths=-/data5t -/workspace -/mnt/eu-ai-datasets" \
    --setenv=PATH="$Q/envs/serve/bin:/usr/bin:/bin" --setenv=LANG=C.UTF-8 --setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID \
    --setenv=TMPDIR="$W/tmp" --setenv=PYTHONNOUSERSITE=1 --setenv=FLASHINFER_WORKSPACE_BASE="$W/cache/flashinfer" \
    --property=StandardOutput=append:"$W/logs/vllm.out" --property=StandardError=append:"$W/logs/vllm.err" \
    --setenv=CUDA_VISIBLE_DEVICES="$GPU" --setenv=HOME="$W" --setenv=HF_HOME="$W/cache/hf" \
    --setenv=VLLM_CACHE_ROOT="$W/cache/vllm" --setenv=TRITON_CACHE_DIR="$W/cache/triton" \
    --setenv=TORCHINDUCTOR_CACHE_DIR="$W/cache/inductor" --setenv=XDG_CACHE_HOME="$W/cache/xdg" \
    --setenv=HF_HUB_OFFLINE=1 --setenv=TRANSFORMERS_OFFLINE=1 --setenv=VLLM_NO_USAGE_STATS=1 --setenv=DO_NOT_TRACK=1 \
    --setenv=PYTHONDONTWRITEBYTECODE=1 \
    "$Q/envs/serve/bin/python" -m vllm.entrypoints.openai.api_server \
      --model "$Q/models/gemma4-31b-bf16-518276fb" --served-model-name absent-edges-gemma4-31b-bf16 \
      --dtype bfloat16 --host 127.0.0.1 --port "$PORT" --max-model-len 32768 \
      --gpu-memory-utilization ${GPU_MEM_UTIL:-0.92} --max-num-seqs 16 --language-model-only \
      --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
      --generation-config vllm --enable-prefix-caching
  systemctl show "$RUN" -p MainPID,ActiveState,InvocationID
  ;;
probe)
  systemctl show "$RUN" -p MainPID,ActiveState,SubState
  curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" | head -c 600; echo
  grep -h -E "KV cache|Maximum concurrency|model weights|Loading weights took|Application startup complete|ERROR|Error" "$W/logs/vllm.err" "$W/logs/vllm.out" 2>/dev/null | tail -12
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
  ;;
stop)
  sudo systemctl stop "$RUN" || true
  sleep 5
  systemctl show "$RUN" -p ActiveState,SubState,Result 2>/dev/null || true
  nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader
  ;;
esac
