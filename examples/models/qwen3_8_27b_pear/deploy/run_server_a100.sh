#!/usr/bin/env bash
# Serve one PEAR station session on two A100s through a local SSH forward.
set -euo pipefail

RECIPE_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$RECIPE_DIR/../../../.." && pwd)
WORK_DIR=${WORK_DIR:?the PEAR work directory}
GPUS=${GPUS:-2,1}
PORT=${PORT:-8000}
NAME=${NAME:-pear}
MODEL=${MODEL:-$WORK_DIR/models/Qwen3.8-27B-PEAR-NVFP4}
DRAFT=${DRAFT:-$WORK_DIR/models/Qwen3.8-27B-DFlash2}
RUN=${RUN:?the directory for logs and compilation caches}
SERVER_ENV=$WORK_DIR/envs/verl

# 24 rounds of three 640x480 frames, without task reference images.
# Each frame uses 300 vision tokens and two markers. Budget 256 tokens for
# state, feedback and formatting, and 4096 for reasoning and the action per
# round. With 2048 for the system prompt: 24 * (906 + 256 + 4096) + 2048 = 128240.
MAX_IMAGES=${MAX_IMAGES:-72}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-131072}
GPU_MEMORY_PERCENT=75
# The previous TP2 run used about 33 GiB/card with a 30 GiB engine budget.
GRAPH_MARGIN_MIB=4096

if [[ ! $GPUS =~ ^[0-9]+,[0-9]+$ ]]; then
  printf 'GPUS must be two nvidia-smi indices, for example 2,1.\n' >&2
  exit 1
fi
IFS=, read -r -a gpu_indices <<< "$GPUS"
gpu_uuids=()
can_start=1
for gpu in "${gpu_indices[@]}"; do
  memory=$(nvidia-smi -i "$gpu" --query-gpu=uuid,memory.total,memory.used,memory.free \
    --format=csv,noheader,nounits)
  IFS=, read -r uuid total_mib used_mib free_mib <<< "${memory// /}"
  gpu_uuids+=("$uuid")
  required_mib=$(((total_mib * GPU_MEMORY_PERCENT + 99) / 100 + GRAPH_MARGIN_MIB))
  printf 'GPU %s: used %s MiB / %s MiB, free %s MiB, requires %s MiB free\n' \
    "$gpu" "$used_mib" "$total_mib" "$free_mib" "$required_mib"
  if ((free_mib < required_mib)); then
    can_start=0
    continue
  fi

  pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
  while read -r pid; do
    if [[ $pid =~ ^[0-9]+$ && ! -O /proc/$pid ]]; then
      printf 'GPU %s has a compute process outside this user (PID %s).\n' "$gpu" "$pid" >&2
      can_start=0
    fi
  done <<< "$pids"
done
if [[ ${gpu_uuids[0]} == "${gpu_uuids[1]}" ]]; then
  printf 'GPUS must select two distinct cards.\n' >&2
  exit 1
fi
if ((can_start == 0)); then
  printf 'Selected GPUs are unavailable. vLLM was not started.\n' >&2
  exit 1
fi

if [[ ! -d $MODEL || ! -d $DRAFT ]]; then
  printf 'MODEL and DRAFT must be existing local directories: %s, %s\n' "$MODEL" "$DRAFT" >&2
  exit 1
fi

# nvidia-smi indices can be enumerated differently by CUDA. Keep rank order by UUID.
export CUDA_VISIBLE_DEVICES="${gpu_uuids[0]},${gpu_uuids[1]}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
LOCAL_DIR=$RUN/cache
# shellcheck source=../../../../scripts/gpu_process_variables.sh
source "$ROOT/scripts/gpu_process_variables.sh" "$SERVER_ENV"

speculative_config=$("$SERVER_ENV/bin/python" -c '
import json
import sys

print(json.dumps({
    "method": "dflash",
    "model": sys.argv[1],
    "num_speculative_tokens": 7,
    "draft_sample_method": "probabilistic",
}))' "$DRAFT")
# The dimensions size the vision profiling input. The count limits all images
# in a request, including older frames referenced by UUID.
printf -v mm_limit '{"image":{"count":%d,"width":640,"height":480},"video":0}' "$MAX_IMAGES"

mkdir -p "$RUN"
printf 'Serving %s on 127.0.0.1:%s, context %s tokens, up to %s images per request\n' \
  "$NAME" "$PORT" "$MAX_MODEL_LEN" "$MAX_IMAGES"
"$SERVER_ENV/bin/vllm" serve "$MODEL" \
  --host 127.0.0.1 --port "$PORT" --served-model-name "$NAME" \
  --tensor-parallel-size 2 --distributed-executor-backend mp \
  --linear-backend marlin --dtype bfloat16 --reasoning-parser qwen3 \
  --gpu-memory-utilization "0.$GPU_MEMORY_PERCENT" \
  --max-num-seqs 1 --max-num-batched-tokens 8192 --enable-prefix-caching \
  --max-model-len "$MAX_MODEL_LEN" --limit-mm-per-prompt "$mm_limit" \
  --speculative-config "$speculative_config" \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,8]}' \
  2>&1 | tee -a "$RUN/server.log"
