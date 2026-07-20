#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN=/home/heyang/miniconda3/bin/conda
ENV_NAME=smartguard_env
DATASET=${1:-fr}
SOURCE_CONTEXT=${2:-winter}
TARGET_CONTEXT=${3:-spring}
CONFIG=${4:-configs/gcad_source/fr.yaml}
COMPRESSION_THRESHOLD=${5:-0.918}
GENERATION_COUNT=${6:-137}
OUT=outputs/gcad_source/${DATASET}/${TARGET_CONTEXT}
STABLE=${OUT}/relations/stable/stable_relation.json

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m SmartGen.gcad_source.cli tensorize \
  --input "SmartGen/IoT_data/${DATASET}/${SOURCE_CONTEXT}/split_trn.pkl" \
  --output "${OUT}/tensor" --dataset "${DATASET}" --config "${CONFIG}"

for SEED in 2024 2025 2026; do
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m SmartGen.gcad_source.cli train \
    --tensor-dir "${OUT}/tensor" --output "${OUT}/seed_${SEED}" \
    --config "${CONFIG}" --seed "${SEED}" --device cpu
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m SmartGen.gcad_source.cli extract-relations \
    --tensor-dir "${OUT}/tensor" --checkpoint "${OUT}/seed_${SEED}/best.pt" \
    --output "${OUT}/relations/seed_${SEED}" --device cpu --edge-threshold 0.01 --top-k 5
done

for PARTITION in 0 1; do
  SEED=$((3101 + PARTITION))
  RUN=${OUT}/bootstrap_partition_${PARTITION}
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m SmartGen.gcad_source.cli train \
    --tensor-dir "${OUT}/tensor" --output "${RUN}" --config "${CONFIG}" --seed "${SEED}" \
    --device cpu --partition-index "${PARTITION}" --partition-count 2 --bootstrap-fraction 0.8
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m SmartGen.gcad_source.cli extract-relations \
    --tensor-dir "${OUT}/tensor" --checkpoint "${RUN}/best.pt" \
    --selection "${RUN}/replicate_selection.json" --output "${OUT}/relations/bootstrap_partition_${PARTITION}" \
    --device cpu --edge-threshold 0.01 --top-k 5
done

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m SmartGen.gcad_source.cli build-stable-relation \
  --matrices \
  "${OUT}/relations/seed_2024/asymmetric_relation_matrix.npy" \
  "${OUT}/relations/seed_2025/asymmetric_relation_matrix.npy" \
  "${OUT}/relations/seed_2026/asymmetric_relation_matrix.npy" \
  "${OUT}/relations/bootstrap_partition_0/asymmetric_relation_matrix.npy" \
  "${OUT}/relations/bootstrap_partition_1/asymmetric_relation_matrix.npy" \
  --primary-lags \
  "${OUT}/relations/seed_2024/primary_lag_matrix.npy" \
  "${OUT}/relations/seed_2025/primary_lag_matrix.npy" \
  "${OUT}/relations/seed_2026/primary_lag_matrix.npy" \
  "${OUT}/relations/bootstrap_partition_0/primary_lag_matrix.npy" \
  "${OUT}/relations/bootstrap_partition_1/primary_lag_matrix.npy" \
  --vocabulary "${OUT}/tensor/channel_vocabulary.json" --output "${OUT}/relations/stable" \
  --occurrence-threshold 0.6 --direction-consistency-threshold 0.6 --stability-threshold 0.08 \
  --seeds 2024 2025 2026 3101 3102 --split-ids full full full partition_0 partition_1

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m SmartGen.gcad_source.cli target-metadata \
  --dataset "${DATASET}" --output "${OUT}/target_static_metadata.json"
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m SmartGen.gcad_source.cli fuse-gss \
  --original-gss "SmartGen/IoT_data/${DATASET}/${SOURCE_CONTEXT}/action_transitions.json" \
  --stable-relation "${STABLE}" --target-metadata "${OUT}/target_static_metadata.json" \
  --output "${OUT}/fusion" --alpha 0.2 --mode rerank_existing
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m SmartGen.gcad_source.cli build-prompts \
  --dataset "${DATASET}" --source-context "${SOURCE_CONTEXT}" --target-context "${TARGET_CONTEXT}" \
  --threshold "${COMPRESSION_THRESHOLD}" --stable-relation "${STABLE}" \
  --fused-gss "${OUT}/fusion/fused_gss.json" --output "${OUT}/prompts"

SOURCE_COUNT=$(jq -r .sequence_count "${OUT}/tensor/tensor_metadata.json")
CONTEXT_DESCRIPTION="{\"change\":\"${SOURCE_CONTEXT} to ${TARGET_CONTEXT}\",\"static_only\":true}"
for METHOD in baseline gcad_gss; do
  GENERATION_OUT=outputs/codex_generation/${DATASET}/${TARGET_CONTEXT}/${METHOD}/replicate_1
  PROMPT=${OUT}/prompts/baseline_prompt.txt
  EXTRA_ARGS=()
  if [[ "${METHOD}" == "gcad_gss" ]]; then
    PROMPT=${OUT}/prompts/gcad_gss_prompt.txt
    EXTRA_ARGS=(--fused-gss "${OUT}/fusion/fused_gss.json" --stable-relation "${STABLE}")
  fi
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m SmartGen.gcad_source.cli export \
    --output "${GENERATION_OUT}" --experiment-id "${DATASET}-${TARGET_CONTEXT}-r1" \
    --dataset "${DATASET}" --context "${TARGET_CONTEXT}" --method "${METHOD}" --replicate 1 \
    --prompt "${PROMPT}" --count "${GENERATION_COUNT}" --batch-size 20 \
    --source-sequence-count "${SOURCE_COUNT}" --context-description "${CONTEXT_DESCRIPTION}" \
    --target-metadata "${OUT}/target_static_metadata.json" \
    --original-gss "SmartGen/IoT_data/${DATASET}/${SOURCE_CONTEXT}/action_transitions.json" "${EXTRA_ARGS[@]}"
done

echo "Requests exported. Codex must now author each generation_responses_raw.jsonl; no API is called."
echo "After responses exist, run validate, convert, continue-pipeline, then the frozen final evaluation commands in docs/gcad_integration_runbook.md."
