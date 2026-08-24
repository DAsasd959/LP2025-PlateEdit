# Weights

None of these files are in the repository — they are too large for git. Download
them and place them here so the paths below resolve.

| File | Size | SHA-256 | Needed for |
|---|---:|---|---|
| `lp2025_27606/adapter_model.safetensors` | 222 MB | `61d52f45f3e220c396c398a19232ced1d65602b4201327a399fe9b0a1c690e96` | inference |
| `lp2025_27606/adapter_config.json` | 2 KB | — | inference |
| `trba_lp2025/best_accuracy.pth` | 191 MB | `c101257845ca6002ce7c9803387a4facc6b8b8173f9a44ae4df21c0287fa0eea` | `eval_ocr.py` |
| `epoch_100.pt` | 677 MB | `a7e329c97cae19e4fd3ad1b5867036952477dd09e53b90f39b2c68b100060156` | training only |

Verify after downloading:

```bash
sha256sum -c checksums.txt
```

The base diffusion model is separate: download **FLUX.1 Fill [dev]** from
Hugging Face in its ordinary full-precision form and point `FLUX_DIR` at it.
`src/train/model.py` quantises it to NF4 on load. Those weights are governed by
the FLUX.1 [dev] Non-Commercial License, which forbids commercial use.
