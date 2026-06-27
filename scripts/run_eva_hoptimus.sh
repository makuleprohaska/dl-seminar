#!/usr/bin/env bash
# Re-run H-optimus-1 through eva's stock config but with H-optimus's OWN image
# normalization. H-optimus does NOT use ImageNet stats; eva's stock config
# defaults every model to ImageNet, which under-normalizes H-optimus and cost it
# ~2-3pp vs the paper. eva exposes NORMALIZE_MEAN/STD as env vars, so this is the
# untouched config with the correct per-model transform. Stats from
# distill/eval/eva_registry.py (H_OPTIMUS_1_MEAN/STD).
#   nohup bash scripts/run_eva_hoptimus.sh > logs/eva_native_hopt/all.log 2>&1 &
set -o pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CFG=eva/configs/vision/pathology/offline/classification
ROOT=logs/eva_native_hopt
EMB=data/embeddings_eva_native_hopt
model=pathology/bioptimus_h_optimus_1
mkdir -p "$ROOT"

export NORMALIZE_MEAN='[0.707223, 0.578729, 0.703617]'
export NORMALIZE_STD='[0.211883, 0.230117, 0.177517]'

for ds in bach mhist; do
  echo "######## hoptimus1 / $ds (H-optimus normalization) ########"
  out="$ROOT/$ds"
  rm -rf "$out" "$EMB/$model/$ds"; mkdir -p "$out"
  MODEL_NAME="$model" IN_FEATURES=1536 EMBEDDINGS_ROOT="$EMB" \
  OUTPUT_ROOT="$out" TQDM_REFRESH_RATE=0 \
    $PY -m eva predict_fit --config "$CFG/$ds.yaml" > "$out/run.log" 2>&1 \
    && echo "  hoptimus1/$ds done" || echo "  hoptimus1/$ds FAILED (see $out/run.log)"
  res=$(find "$out" -name results.json 2>/dev/null | tail -1)
  $PY - "$res" "$ds" <<'PY'
import json, sys
res, ds = sys.argv[1], sys.argv[2]
try:
    m = json.load(open(res))["metrics"]["val"][0]
    want = ["MulticlassAccuracy"] if ds == "bach" else ["BinaryBalancedAccuracy", "BinaryAccuracy"]
    for w in want:
        k = [x for x in m if x.endswith(w)]
        if k:
            d = m[k[0]]
            print("  RESULT hoptimus1/%s %s: %.2f +- %.2f (n=%d)"
                  % (ds, w, d["mean"]*100, d["stdev"]*100, len(d["values"])))
except Exception as e:
    print("  RESULT hoptimus1/%s: could not read %r (%s)" % (ds, res, e))
PY
done
echo "HOPT DONE"
