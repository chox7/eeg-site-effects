#!/usr/bin/env bash
# Compute the feature-level baseline_2layer DANN-probe results.
#
# Adds the missing `baseline_2layer` tag to the two master result CSVs so the
# 04_dann notebook can show baseline at BOTH extractor depths (1L vs 2L),
# matching the dann_2layer / mtl_2layer variants already present.
#
# Four jobs:  site-clf × {catboost, logreg}  +  patho-clf × {catboost, logreg}
# all on the baseline-2-layer features (extracted_features_baseline_2l_final.npz,
# already converted to ELM19_features_dann_baseline_2layer*_newlabels.csv).
#
# CatBoost uses task_type: GPU -> run on a GPU node (e.g. wigner). LogReg is CPU.
#
# Run with:
#   tmux new -s b2l
#   bash scripts/run_baseline_2layer.sh
#
# Logs:  results/logs/*_baseline_2layer.out
# Temp:  results/tables/04_dann/_rerun/   (merged into masters at the end)
# Idempotent: re-running drops any prior baseline_2layer rows before merging.

set -u  # NOT -e: a failing job should not abort the others.

cd /dmj/fizmed/kchorzela/licencjat/eeg-site-effects
source .venv/bin/activate

DANN_DIR=data/ELM19/dann/newlabels
RERUN=results/tables/04_dann/_rerun
SITE_MASTER=results/tables/04_dann/site_clf_results.csv
PATHO_MASTER=results/tables/04_dann/patho_clf_results.csv
TAG=baseline_2layer
mkdir -p results/logs "$RERUN"

# Safety backup of the two masters before we touch them.
BK=.csv_backup_$(date +%Y%m%d_%H%M%S)
mkdir -p "$BK"
cp "$SITE_MASTER" "$PATHO_MASTER" "$BK"/ 2>/dev/null || true
echo "Backed up masters to $BK/"

run_job () {
    local label="$1"; shift
    echo
    echo "================================================================"
    echo "[$(date '+%F %T')] >>> START  $label"
    echo "================================================================"
    "$@"
    echo "[$(date '+%F %T')] <<< DONE   $label  (exit=$?)"
}

# ---------------------------------------------------------------------------
# A. site_clf — residual site MCC on baseline-2L features (normals only)
# ---------------------------------------------------------------------------
for MODEL in catboost logreg; do
    run_job "dann_site $MODEL / $TAG" \
        bash -c "python experiments/ml/site_classification.py \
            -c experiments/configs/dann_site_classification_newlabels.yaml \
            --features ${DANN_DIR}/ELM19_features_dann_baseline_2layer_norm_newlabels.csv \
            --model ${MODEL} --tag ${TAG} \
            --results-file ${RERUN}/dann_site_${MODEL}_${TAG}.csv \
            2>&1 | tee results/logs/dann_site_${MODEL}_${TAG}.out"
done

# ---------------------------------------------------------------------------
# B. patho_clf — LOSO zero-shot pathology on baseline-2L features (k_calib=30)
# ---------------------------------------------------------------------------
for MODEL in catboost logreg; do
    run_job "dann_patho $MODEL / $TAG" \
        bash -c "python experiments/ml/pathology_classification.py \
            -c experiments/configs/dann_pathology_classification_newlabels.yaml \
            --features ${DANN_DIR}/ELM19_features_dann_baseline_2layer_newlabels.csv \
            --model ${MODEL} --tag ${TAG} \
            --results-file ${RERUN}/dann_patho_${MODEL}_${TAG}.csv \
            2>&1 | tee results/logs/dann_patho_${MODEL}_${TAG}.out"
done

# ---------------------------------------------------------------------------
# C. Merge temp results into the master CSVs (idempotent: drop old baseline_2layer)
# ---------------------------------------------------------------------------
echo
echo "[$(date '+%F %T')] Merging baseline_2layer rows into master CSVs..."
python - "$TAG" "$SITE_MASTER" "$PATHO_MASTER" "$RERUN" <<'PY'
import sys, glob, os
import pandas as pd

tag, site_master, patho_master, rerun = sys.argv[1:5]

def merge(master, temp_glob):
    temps = sorted(glob.glob(temp_glob))
    if not temps:
        print(f"  !! no temp files for {temp_glob} — skipping {master}")
        return
    new = pd.concat([pd.read_csv(t) for t in temps], ignore_index=True)
    if os.path.isfile(master):
        old = pd.read_csv(master)
        old = old[old.get('tag') != tag]          # idempotent: drop stale baseline_2layer
        out = pd.concat([old, new], ignore_index=True)
    else:
        out = new
    out.to_csv(master, index=False)
    print(f"  merged {len(new)} rows from {len(temps)} file(s) -> {master} "
          f"(tags now: {sorted(out['tag'].dropna().unique())})")

merge(site_master,  f"{rerun}/dann_site_*_{tag}.csv")
merge(patho_master, f"{rerun}/dann_patho_*_{tag}.csv")
PY

echo
echo "================================================================"
echo "ALL DONE at $(date '+%F %T')"
echo "Verify, then optionally:  rm -rf ${RERUN}"
echo "Next: re-run notebooks/04_dann/01_dann_evaluation.ipynb to regenerate figures/tables."
echo "================================================================"
