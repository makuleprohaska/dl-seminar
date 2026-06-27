#!/usr/bin/env bash
# 5-fold cross-validation for one model on one task, using eva's OWN code.
#
# Flow (frozen backbone -> embed once, CV the head):
#   1. eva predict  : embed ALL task samples once -> data/.../manifest.csv
#   2. make_fold_manifest.py : add fold0..fold4 columns -> manifest_cv.csv
#   3. eva fit x5   : one run per fold (FOLD_COLUMN=foldK); eva re-inits the head
#                     each run, trains from scratch, evaluates that fold's val set
#   4. collect_cv.py: pool the 5 fold accuracies -> mean +/- std
#
# eva does all the training/eval; we only assign folds and average. Run on the VM
# (eva present, Python 3.10+, repo root on PYTHONPATH so the registry resolves).
#
# Usage:
#   rootdir=~/patho-distill \
#   MODEL_NAME=pathology/patho_distill IN_FEATURES=768 NUM_CLASSES=4 \
#   TASK=bach DATA_ROOT=/path/to/BACH CHECKPOINT_PATH=$rootdir/checkpoints/patho-099613.ckpt \
#   bash scripts/run_cv.sh
#
# Teachers: set MODEL_NAME to their built-in registry name (e.g. pathology/uni2_h),
# IN_FEATURES accordingly (1536/1280/1536); no CHECKPOINT_PATH needed.
set -euo pipefail

: "${rootdir:?set rootdir to the repo root}"
: "${MODEL_NAME:?e.g. pathology/patho_distill}"
: "${TASK:?bach or mhist}"
: "${DATA_ROOT:?path to the raw task dataset (for the embed step)}"
export IN_FEATURES="${IN_FEATURES:?student 768 / uni2 1536 / virchow2 1280 / hoptimus1 1536}"
export NUM_CLASSES="${NUM_CLASSES:?BACH=4 MHIST=2}"
FOLDS="${FOLDS:-5}"

# Slugify the registry name for paths (pathology/patho_distill -> pathology_patho_distill).
slug="${MODEL_NAME//\//_}"
export EMBEDDINGS_ROOT="${EMBEDDINGS_ROOT:-$rootdir/data/embeddings_cv/$slug/$TASK}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$rootdir/logs/cv/$slug/$TASK}"
export MANIFEST_FILE="manifest_cv.csv"
STOCK_CONFIG="eva/configs/vision/pathology/offline/classification/${TASK}.yaml"

cd "$rootdir/eva/src"

# Per-model normalization for the embed step. Student/UNI2/Virchow2 use ImageNet
# stats (eva's defaults); H-optimus uses its OWN stats -- embedding it with
# ImageNet norm would silently degrade its features.
if [ "$MODEL_NAME" = "pathology/bioptimus_h_optimus_1" ]; then
  export NORMALIZE_MEAN="[0.707223, 0.578729, 0.703617]"
  export NORMALIZE_STD="[0.211883, 0.230117, 0.177517]"
  echo ">> using H-optimus normalization stats for embedding"
fi

# 1. Embed once (eva predict). Skips if the manifest already exists (overwrite:false).
if [ ! -f "$EMBEDDINGS_ROOT/manifest.csv" ]; then
  echo ">> [1/4] embedding all $TASK samples with $MODEL_NAME"
  EMBEDDINGS_ROOT="$(dirname "$EMBEDDINGS_ROOT")" DATA_ROOT="$DATA_ROOT" \
    python3.12 -m eva predict --config "$rootdir/$STOCK_CONFIG"
else
  echo ">> [1/4] embeddings present: $EMBEDDINGS_ROOT/manifest.csv"
fi

# 2. Add stratified fold columns.
echo ">> [2/4] adding $FOLDS fold columns"
python3.12 "$rootdir/scripts/make_fold_manifest.py" \
  --manifest "$EMBEDDINGS_ROOT/manifest.csv" \
  --out "$EMBEDDINGS_ROOT/$MANIFEST_FILE" --folds "$FOLDS"

# 3. Fit eva's head once per fold (head re-initialized each run).
for k in $(seq 0 $((FOLDS - 1))); do
  echo ">> [3/4] fold $k"
  FOLD_COLUMN="fold$k" python3.12 -m eva fit --config "$rootdir/configs/eva/cv_fit.yaml"
done

# 4. Aggregate the 5 fold accuracies.
echo ">> [4/4] $MODEL_NAME on $TASK -- 5-fold CV:"
python3.12 "$rootdir/scripts/collect_cv.py" "$OUTPUT_ROOT"/fold*/results.json
