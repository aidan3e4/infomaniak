#!/bin/bash
set -e

source "$HOME/.local/bin/env"
source .venv/bin/activate
set -a; source .env; set +a

# Options:
#   No adapter (base model):  ./serve.sh
#   HF adapter (latest):      ./serve.sh aidan3e4/receipt-lora-qwen3-4b
#   HF adapter at revision:   ./serve.sh aidan3e4/receipt-lora-qwen3-4b abc123f
#   Local checkpoint:         ./serve.sh ./receipt-lora-output/checkpoint-100

LORA_ADAPTER="${1:-}"
REVISION="${2:-}"

if [ -n "$LORA_ADAPTER" ]; then
  LORA_SPEC="receipt-lora=$LORA_ADAPTER"
  if [ -n "$REVISION" ]; then
    echo "Serving with LoRA adapter: $LORA_ADAPTER (revision: $REVISION)"
    LORA_SPEC="receipt-lora=$LORA_ADAPTER@$REVISION"
  else
    echo "Serving with LoRA adapter: $LORA_ADAPTER"
  fi
  vllm serve "Qwen/Qwen3-4B-Instruct-2507" \
    --max-model-len 131072 \
    --kv-cache-dtype fp8_e4m3 \
    --gpu-memory-utilization 0.92 \
    --enable-lora \
    --lora-modules "$LORA_SPEC"
else
  echo "Serving base model (no adapter)"
  vllm serve "Qwen/Qwen3-4B-Instruct-2507" \
    --max-model-len 131072 \
    --kv-cache-dtype fp8_e4m3 \
    --gpu-memory-utilization 0.92
fi
