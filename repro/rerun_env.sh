source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export REPRO_DEVICE=${REPRO_DEVICE:-cuda} REPRO_BATCH=${REPRO_BATCH:-32} HF_HUB_OFFLINE=1 RL="$R/home/projects/smiles/logs/rerun"
mkdir -p "$RL"
