# Publishing this repository

The weights are staged under `weights/` and verified against
`weights/checksums.txt`, but git ignores them — they go out as release assets,
not as tracked files. Confirm that before pushing:

```bash
git status --short          # weights/ must not appear
git ls-files weights/       # only README.md and checksums.txt
```

## 1. Create the repository

The GitHub CLI is not installed on this machine. Either install it:

```bash
sudo apt install gh && gh auth login
gh repo create <user>/LP2025-PlateEdit --public --source=. --push
```

or create the repository through the web interface and push by hand:

```bash
git remote add origin https://github.com/<user>/LP2025-PlateEdit.git
git branch -M main
git push -u origin main
```

## 2. Upload the weights as a release

```bash
gh release create v1.0 \
    weights/lp2025_27606/adapter_model.safetensors \
    weights/lp2025_27606/adapter_config.json \
    weights/trba_lp2025/best_accuracy.pth \
    weights/epoch_100.pt \
    --title "LP2025 checkpoints" \
    --notes "Diffusion adapter at step 27606, TRBA recogniser, ODM loss encoder. Checksums in weights/checksums.txt."
```

Without `gh`, the same thing through the web interface: Releases → Draft a new
release → tag `v1.0` → drag the four files in. Each is under the 2 GB per-asset
limit; the largest is `epoch_100.pt` at 710 MB, so expect the upload to take a
while.

## 3. Fill in the download URLs

`weights/README.md` carries a placeholder host. Replace it once the release
exists:

```bash
sed -i 's|<RELEASE_URL>|https://github.com/<user>/LP2025-PlateEdit/releases/download/v1.0|' \
    weights/README.md
git commit -am "Point the weight table at the v1.0 release" && git push
```

## Before making it public

- The paper's review status is your call, but a public repository is indexed and
  cached quickly; treat publication as difficult to undo.
- `FLUX.1 [dev]` weights are non-commercial. This repository does not ship them,
  but anyone following the README will download them, so the constraint travels
  with the instructions.
- The repository declares no license for its own code, which by default means
  others have no permission to reuse it. That may be what you want; it is worth
  being a deliberate choice rather than an oversight.
