#!/usr/bin/env bash
set -euo pipefail

# Utility to run integration tests sequentially with varying TP configurations.
# Usage: ./tp_config_sweep_accuracy_test.sh [standard|wide-ep|all]
#   standard  - Run PD TP test cases (default)
#   wide-ep   - Run wide EP test cases
#   all       - Run all test cases
SCRIPT="v1/kv_connector/nixl_integration/run_accuracy_test.sh"

# Define test configurations
standard_configs=(
  "GPU_MEMORY_UTILIZATION=0.6 PREFILLER_TP_SIZE=2 DECODER_TP_SIZE=2"
  "GPU_MEMORY_UTILIZATION=0.6 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2"
  "GPU_MEMORY_UTILIZATION=0.8 MODEL_NAMES=deepseek-ai/deepseek-vl2-tiny" # MLA case
  "GPU_MEMORY_UTILIZATION=0.8 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2 MODEL_NAMES=deepseek-ai/deepseek-vl2-tiny"
)

wide_ep_configs=(
  # MLA+P-TP1, D-DPEP=2 (TP=1) 
  "DP_EP=1 GPU_MEMORY_UTILIZATION=0.8 PREFILLER_TP_SIZE=1 DECODER_TP_SIZE=2 EXTRA_ARGS='--max-model-len 8192 --max-num-seqs 8' MODEL_NAMES=RedHatAI/DeepSeek-V2.5-1210-FP8"
)

# Select configs based on argument
TEST_MODE="${1:-standard}"
case "$TEST_MODE" in
  standard)
    configs=("${standard_configs[@]}")
    ;;
  wide-ep)
    configs=("${wide_ep_configs[@]}")
    ;;
  all)
    configs=("${standard_configs[@]}" "${wide_ep_configs[@]}")
    ;;
  *)
    echo "Unknown test mode: $TEST_MODE"
    echo "Usage: $0 [standard|wide-ep|all]"
    exit 1
    ;;
esac

echo "Running tests in '$TEST_MODE' mode (${#configs[@]} configs)"

run_tests() {
  local label=$1
  local extra_env=$2

  echo "=== Running tests (${label}) ==="
  for cfg in "${configs[@]}"; do
    echo "-> Running with ${cfg} ${extra_env:+and ${extra_env}}"
    # Use 'env' to safely set variables without eval
    if ! env ${extra_env} ${cfg} bash "${SCRIPT}"; then
      echo "❌ Test failed for config: ${cfg} ${extra_env:+(${extra_env})}"
      exit 1
    fi
  done
  echo "✅ All ${label} tests passed!"
}

# Run tests
run_tests "default backend" ""

# Check if FLASHINFER is set (non-empty)
if [[ -n "${FLASHINFER:-}" ]]; then
  echo "FLASHINFER is set, rerunning with VLLM_ATTENTION_BACKEND=FLASHINFER"
  run_tests "FLASHINFER backend" "VLLM_ATTENTION_BACKEND=FLASHINFER"
else
  echo "FLASHINFER not set, skipping FLASHINFER runs."
fi
