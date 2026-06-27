#!/usr/bin/env bash
# eva's PUBLISHED fixed-split protocol, the way the paper's Table 1 was produced:
# the designated train/val split (FOLD_COLUMN=split = eva's hardcoded 268/132 BACH,
# 2175/977 MHIST) with n_runs=5 -> eva re-seeds + re-inits the linear head each run
# and reports the seed-variance mean(std). Cached embeddings, fit-only.
#
#   nohup bash scripts/run_designated.sh > logs/designated/all.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
MANIFEST="manifest_cv.csv"
ROOT="logs/designated"
mkdir -p "$ROOT"

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

for c in "${COMBOS[@]}"; do
  IFS='|' read -r label emb infeat nclass <<< "$c"
  echo "######## $label (dim=$infeat classes=$nclass) ########"
  out="$ROOT/$label"
  rm -rf "$out"; mkdir -p "$out"           # fresh -> no stale results pooled
  $PY scripts/make_fold_manifest.py --manifest "$emb/manifest.csv" --out "$emb/$MANIFEST" | tail -1
  MODEL_NAME="$label" IN_FEATURES="$infeat" NUM_CLASSES="$nclass" \
  EMBEDDINGS_ROOT="$emb" MANIFEST_FILE="$MANIFEST" FOLD_COLUMN="split" \
  N_RUNS=5 OUTPUT_ROOT="$out" TQDM_REFRESH_RATE=0 \
    $PY -m eva fit --config configs/eva/cv_fit.yaml > "$out/run.log" 2>&1 \
    && echo "  fit done" || echo "  FAILED (see $out/run.log)"
  # Pool every run value eva wrote (robust to a single aggregated json or 5 dirs).
  $PY - "$out" "$label" <<'PY'
import glob, json, statistics, sys
out, label = sys.argv[1], sys.argv[2]
vals = []
for f in glob.glob(f"{out}/split/*/results.json"):
    m = json.load(open(f))["metrics"]["val"][0]["val/MulticlassAccuracy"]
    vals += m["values"]
if vals:
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"  {label} fixed-split n_runs=5: {statistics.mean(vals)*100:.2f} +- {sd*100:.2f}  (n={len(vals)})")
else:
    print(f"  {label}: no results.json found under {out}/split")
PY
done
echo "ALL DONE"
