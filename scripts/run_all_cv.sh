#!/usr/bin/env bash
# Full 5-fold CV sweep on cached embeddings: 4 models x {BACH, MHIST}.
# Embeddings are already cached (frozen backbone), so this only re-fits eva's
# linear head per fold -- eva re-initializes the head each run (genuine CV).
# Run from the repo root on the VM:  nohup bash scripts/run_all_cv.sh > logs/cv/all.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FOLDS=5

# label | embeddings dir | in_features | num_classes
COMBOS=(
  "student_bach|data/embeddings_morning_1414/patho-distill/bach|768|4"
  "uni2_bach|data/embeddings_teacher_uni2/teacher_uni2/bach|1536|4"
  "virchow2_bach|data/embeddings_teacher_virchow2/teacher_virchow2/bach|1280|4"
  "hoptimus1_bach|data/embeddings_teacher_hoptimus1/teacher_hoptimus1/bach|1536|4"
  "student_mhist|data/embeddings_mhist_student/mhist_student/mhist|768|2"
  "uni2_mhist|data/embeddings_mhist_uni2/mhist_uni2/mhist|1536|2"
  "virchow2_mhist|data/embeddings_mhist_virchow2/mhist_virchow2/mhist|1280|2"
  "hoptimus1_mhist|data/embeddings_mhist_hoptimus1/mhist_hoptimus1/mhist|1536|2"
)

# Clean stratified 5-fold partition (the prof's scheme) + a separate single fit on
# the designated split (FOLD_COLUMN=split) for Table-1 comparability.
MANIFEST="manifest_cv.csv"
ROOT="logs/cv"

run_fit() {  # $1=label $2=emb $3=infeat $4=nclass $5=fold_column
  MODEL_NAME="$1" IN_FEATURES="$3" NUM_CLASSES="$4" \
  EMBEDDINGS_ROOT="$2" MANIFEST_FILE="$MANIFEST" FOLD_COLUMN="$5" \
  OUTPUT_ROOT="$ROOT/$1" TQDM_REFRESH_RATE=0 \
    $PY -m eva fit --config configs/eva/cv_fit.yaml > "$ROOT/$1/$5.log" 2>&1 \
    && echo "  $5 done" || echo "  $5 FAILED (see $ROOT/$1/$5.log)"
}

for c in "${COMBOS[@]}"; do
  IFS='|' read -r label emb infeat nclass <<< "$c"
  echo "######## $label (dim=$infeat classes=$nclass) ########"
  mkdir -p "$ROOT/$label"
  $PY scripts/make_fold_manifest.py --manifest "$emb/manifest.csv" --out "$emb/$MANIFEST" | tail -1
  # 5 CV folds + the designated split, all independent -> run in parallel, then wait.
  for k in $(seq 0 $((FOLDS - 1))); do
    ( run_fit "$label" "$emb" "$infeat" "$nclass" "fold$k" ) &
  done
  ( run_fit "$label" "$emb" "$infeat" "$nclass" "split" ) &   # designated split
  wait
  echo "== $label 5-fold CV (eva nests results under a timestamped subdir) =="
  $PY scripts/collect_cv.py "$ROOT/$label"/fold*/*/results.json || echo "collect failed for $label"
  echo "== $label designated split (single fit) =="
  $PY scripts/collect_cv.py "$ROOT/$label"/split/*/results.json || echo "collect failed for $label"
done
echo "ALL DONE"
