#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN=/home/heyang/miniconda3/bin/conda
ENV_NAME=smartguard_env
DATASET=${1:-fr}
SOURCE_CONTEXT=${2:-winter}
TARGET_CONTEXT=${3:-spring}
CONFIG=${4:-configs/gcad_source/fr.yaml}
OUT=outputs/gcad_source/${DATASET}/${TARGET_CONTEXT}

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

echo "Source-only training and relation extraction complete at ${OUT}."
echo "No external API is called. Export/validate/convert Codex file requests with the CLI."

