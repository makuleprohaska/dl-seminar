#!/usr/bin/env bash
# Re-run the two GATED teachers (UNI2-h, Virchow2) through eva's UNTOUCHED stock
# configs, end-to-end (predict_fit, n_runs=5). eva's gated-model loader requires
# HF_TOKEN (sourced from ~/.profile); weights are already cached so nothing big
# re-downloads. Student + H-optimus-1 already completed under logs/eva_native.
#   nohup bash scripts/run_eva_teachers.sh > logs/eva_native/teachers.log 2>&1 &
set -o pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1090
[ -f "$HOME/.profile" ] && source "$HOME/.profile" || true   # load HF_TOKEN
PY=.venv/bin/python
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CFG=eva/configs/vision/pathology/offline/classification
ROOT=logs/eva_native
EMB=data/embeddings_eva_native

COMBOS=(
  "uni2|pathology/mahmood_uni2_h|1536"
  "virchow2|pathology/paige_virchow2|1280"
)
for c in "${COMBOS[@]}"; do
  IFS='|' read -r label model infeat <<< "$c"
  for ds in bach mhist; do
    echo "######## $label / $ds (model=$model dim=$infeat) ########"
    out="$ROOT/$label/$ds"
    rm -rf "$out" "$EMB/$model/$ds"; mkdir -p "$out"   # predict writer refuses existing dirs
    MODEL_NAME="$model" IN_FEATURES="$infeat" EMBEDDINGS_ROOT="$EMB" \
    OUTPUT_ROOT="$out" TQDM_REFRESH_RATE=0 \
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
echo "TEACHERS DONE"
