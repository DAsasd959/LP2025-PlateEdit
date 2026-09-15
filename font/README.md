# Fonts

Two typefaces drive glyph rendering. Neither is redistributed here — `.gitignore`
excludes them — because both are third-party and their licences were not written
with redistribution in mind. Obtain your own copies and drop them in this folder,
or point the scripts elsewhere with `--cjk_font` / `--latin_font`, or the
`CJK_FONT` / `LATIN_FONT` environment variables.

| File | Used for | Needed by |
|---|---|---|
| `TWGen7_V1.ttf` | the plate face: digits and Latin letters | every LP and CCPD script |
| `正黑體.ttf` (any CJK sans will do) | the CCPD province character | `cn_preprocess.py`, `build_ccpd_conditions.py`, `glyph_mixed.py` |

`TWGen7_V1.ttf` carries 254 glyphs and no CJK at all, which is why the province
cell needs a second font. Any CJK sans-serif substitutes acceptably — the model
learns to blend with the surrounding style rather than to copy the glyph exactly —
but the released checkpoints were trained with the two above, so a substitute will
shift results slightly.
