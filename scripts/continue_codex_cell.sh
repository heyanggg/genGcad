#!/usr/bin/env bash
set -euo pipefail

CONDA_BIN=/home/heyang/miniconda3/bin/conda
ENV_NAME=smartguard_env
DATASET=${1:-fr}
TARGET_CONTEXT=${2:-spring}
PERCENTILE=${3:-95.5}
STABLE=outputs/gcad_source/${DATASET}/${TARGET_CONTEXT}/relations/stable/stable_relation.json

for METHOD in baseline gcad_gss; do
  DIRECTORY=outputs/codex_generation/${DATASET}/${TARGET_CONTEXT}/${METHOD}/replicate_1
  if [[ ! -f "${DIRECTORY}/generation_responses_raw.jsonl" ]]; then
    echo "Missing Codex-authored response: ${DIRECTORY}/generation_responses_raw.jsonl" >&2
    exit 1
  fi
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m SmartGen.gcad_source.cli validate --directory "${DIRECTORY}"
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m SmartGen.gcad_source.cli convert \
    --directory "${DIRECTORY}" --dataset "${DATASET}"
  "${CONDA_BIN}" run -n "${ENV_NAME}" python -m SmartGen.gcad_source.cli continue-pipeline \
    --directory "${DIRECTORY}" --dataset "${DATASET}" --context "${TARGET_CONTEXT}" \
    --tof-epochs 10 --stable-relation "${STABLE}" --ranking-weight 1.0
done

if [[ "${RUN_FINAL_EVALUATION:-0}" != "1" ]]; then
  echo "Pre-evaluation pipeline complete. Set RUN_FINAL_EVALUATION=1 only after all settings are frozen."
  exit 0
fi

BASE=outputs/codex_generation/${DATASET}/${TARGET_CONTEXT}/baseline/replicate_1/tof
GCAD=outputs/codex_generation/${DATASET}/${TARGET_CONTEXT}/gcad_gss/replicate_1/tof
DOWNSTREAM=outputs/downstream/${DATASET}/${TARGET_CONTEXT}

"${CONDA_BIN}" run -n "${ENV_NAME}" python -m SmartGen.gcad_source.cli evaluate-generated \
  --generated "${BASE}/tof_sequences.pkl" --dataset "${DATASET}" --context "${TARGET_CONTEXT}" \
  --output "${DOWNSTREAM}/A_baseline" --percentile "${PERCENTILE}" --epochs 15
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m SmartGen.gcad_source.cli evaluate-generated \
  --generated "${GCAD}/tof_sequences.pkl" --dataset "${DATASET}" --context "${TARGET_CONTEXT}" \
  --output "${DOWNSTREAM}/B_gcad_gss" --percentile "${PERCENTILE}" --epochs 15
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m SmartGen.gcad_source.cli evaluate-generated \
  --generated "${BASE}/tof_sequences.pkl" --ranking "${BASE}/sequence_ranking.json" \
  --dataset "${DATASET}" --context "${TARGET_CONTEXT}" --output "${DOWNSTREAM}/C_ranking" \
  --percentile "${PERCENTILE}" --epochs 15
"${CONDA_BIN}" run -n "${ENV_NAME}" python -m SmartGen.gcad_source.cli evaluate-generated \
  --generated "${GCAD}/tof_sequences.pkl" --ranking "${GCAD}/sequence_ranking.json" \
  --dataset "${DATASET}" --context "${TARGET_CONTEXT}" --output "${DOWNSTREAM}/D_both" \
  --percentile "${PERCENTILE}" --epochs 15
