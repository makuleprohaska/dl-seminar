#!/usr/bin/env bash
# GOLD STANDARD: every model (student + 3 teachers) through eva's UNTOUCHED stock
# offline-classification configs, end-to-end via `eva predict_fit`:
#   embed the raw BACH/MHIST datasets through the registry backbone (eva's own
#   fixed split + ImageNet normalization for ALL models) -> fixed-split n_runs=5
#   linear probe. Directly comparable across models and to the paper's Table 1.
#
#   nohup bash scripts/run_eva_native.sh > logs/eva_native/all.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"   # eva imports distill.eval.eva_registry
CFG=eva/configs/vision/pathology/offline/classification
ROOT=logs/eva_native
EMB=data/embeddings_eva_native
# eva's predict embeddings-writer REFUSES to overwrite an existing dir (it raises
# FileExistsError, it does not reuse). Clear any stale cache so every model does a
# clean fresh embed -> uniform, directly-comparable runs.
rm -rf "$EMB"
mkdir -p "$ROOT"

# label | MODEL_NAME (registry key) | IN_FEATURES | teacher?(1/0)
COMBOS=(
  "student|pathology/patho_distill|768|0"
  "uni2|pathology/mahmood_uni2_h|1536|1"
  "virchow2|pathology/paige_virchow2|1280|1"
  "hoptimus1|pathology/bioptimus_h_optimus_1|1536|1"
)
DATASETS=("bach" "mhist")

for c in "${COMBOS[@]}"; do
  IFS='|' read -r label model infeat is_teacher <<< "$c"
  for ds in "${DATASETS[@]}"; do
    echo "######## $label / $ds (model=$model dim=$infeat) ########"
    out="$ROOT/$label/$ds"; mkdir -p "$out"
    extra=()
    if [ "$label" = "student" ]; then
      extra=(CHECKPOINT_PATH="$PWD/checkpoints/patho-099613.ckpt")
    else
      # gated teacher weights are cached -> load offline, no HF auth
      extra=(HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1)
    fi
    env "${extra[@]}" \
      MODEL_NAME="$model" IN_FEATURES="$infeat" \
      EMBEDDINGS_ROOT="$EMB" OUTPUT_ROOT="$out" TQDM_REFRESH_RATE=0 \
      $PY -m eva predict_fit --config "$CFG/$ds.yaml" > "$out/run.log" 2>&1 \
      && echo "  $label/$ds done" || echo "  $label/$ds FAILED (see $out/run.log)"
    res=$(find "$out" -name results.json 2>/dev/null | tail -1)
    $PY - "$res" "$label" "$ds" <<'PY'
import json, sys
res, label, ds = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    m = json.load(open(res))["metrics"]["val"][0]
    want = ["MulticlassAccuracy"] if ds == "bach" else ["BinaryBalancedAccuracy", "BinaryAccuracy"]
    for w in want:
        k = [x for x in m if x.endswith(w)]
        if k:
            d = m[k[0]]
            print("  RESULT %s/%s %s: %.2f +- %.2f (n=%d)"
                  % (label, ds, w, d["mean"]*100, d["stdev"]*100, len(d["values"])))
except Exception as e:
    print("  RESULT %s/%s: could not read %r (%s)" % (label, ds, res, e))
PY
  done
done
echo "ALL DONE"
