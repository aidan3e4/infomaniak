#!/bin/bash
set -e

source "$HOME/.local/bin/env"
cd ~/infomaniak
source .venv/bin/activate

vllm serve "Qwen/Qwen3-4B-Instruct-2507" \
  --max-model-len 131072 \
  --kv-cache-dtype fp8_e4m3 \
  --gpu-memory-utilization 0.92
