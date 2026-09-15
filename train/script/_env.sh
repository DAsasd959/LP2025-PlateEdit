# Shared environment for every training script. Sourced, not executed.
#
# WANDB_API_KEY is never stored in this repository. Provide it one of three ways:
#   1. export WANDB_API_KEY=...        in your shell
#   2. a ~/.netrc entry for api.wandb.ai   (machine api.wandb.ai / login user / password <key>)
#   3. leave it unset — training still runs, wandb logging is skipped
# src/train/train.py asserts the variable is set only when the config has a
# `wandb:` block, so option 3 means commenting that block out of the YAML.
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
