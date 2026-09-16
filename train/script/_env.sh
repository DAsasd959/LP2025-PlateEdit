# Shared environment for every training script. Sourced, not executed.
#
# Runs log to YOUR wandb account, not to anyone else's. No key is stored in this
# repository, and none is needed to train.
#
#   WANDB_API_KEY   your own key. Export it, or put it in ~/.netrc under
#                   `machine api.wandb.ai`. Leave it unset and training runs with
#                   logging skipped -- one printed line, nothing else changes.
#   WANDB_PROJECT   project name, overriding whatever the config says
#   WANDB_ENTITY    account or team to log under; defaults to the key's owner
#
# Never commit a key. It is an account credential, not a setting.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."          # repo root

if [ -z "${WANDB_API_KEY:-}" ] && [ -r "$HOME/.netrc" ]; then
  _k=$(awk '/machine[[:space:]]+api\.wandb\.ai/{f=1} f&&/password/{print $2; exit}' "$HOME/.netrc" 2>/dev/null || true)
  [ -n "${_k:-}" ] && export WANDB_API_KEY="$_k"
  unset _k
fi
if [ -n "${WANDB_API_KEY:-}" ]; then echo "wandb: key found (not printed)"; else echo "wandb: no key — logging disabled"; fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

run_train () {
  local cfg="$1"
  [ -f "$cfg" ] || { echo "config not found: $cfg" >&2; exit 1; }
  local flux; flux=$(python -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['flux_path'])" "$cfg")
  [ -d "$flux" ] || { echo "flux_path in $cfg points at '$flux', which does not exist." >&2
                      echo "Place FLUX.1-Fill-dev there, or edit the field." >&2; exit 1; }
  echo "config:     $cfg"
  echo "base model: $flux"
  export XFL_CONFIG="$cfg"
  torchrun --nproc_per_node="${NPROC:-1}" -m src.train.train
}
