# Recognisers

Text accuracy (ACC / NED) is scored with TRBA — TPS-ResNet-BiLSTM-Attn from
[deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark).
Three trained recognisers are released; the benchmark code itself is not vendored
here, clone it from upstream.

| Release asset | Dataset | Character set | Iterations |
|---|---|---|---|
| `trba_lp2025` (in `v1.0`) | LP-2025 plates | `A–Z 0–9 ·` (37) | 300,000 |
| `trba_ccpd_final` (in `v2.0`) | CCPD | `0–9 A–Z` + 31 province characters + 警 学 (69) | 20,000 |

A third, `ccpd_prov_balanced`, was trained on the province-balanced split and is
not released; it exists to show that balancing helps the recogniser too.

## Retraining

```bash
git clone https://github.com/clovaai/deep-text-recognition-benchmark
cd deep-text-recognition-benchmark

# 1. plate crops + transcriptions -> lmdb
python create_lmdb_dataset.py --inputPath <crops> --gtFile <gt.txt> --outputPath <split>/lmdb

# 2. train -- these are the options the released recognisers used
python train.py \
    --train_data <train>/lmdb --valid_data <val>/lmdb \
    --select_data / --batch_ratio 1 \
    --Transformation TPS --FeatureExtraction ResNet \
    --SequenceModeling BiLSTM --Prediction Attn \
    --imgH 128 --imgW 128 --batch_size 32 --batch_max_length 25 \
    --num_iter 300000 \
    --character 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789·'
```

For CCPD swap the character set for the 69-symbol one above and use
`--num_iter 20000`. Note `imgH 128 --imgW 128`: square, not the benchmark's
default 32×100, because plate crops are close to square once cropped.

`gt.txt` is one `<filename>\t<transcription>` per line.

## Why the reported accuracy is a lower bound

Manual review of 409 CCPD items found the recogniser missed 129 correct edits and
credited none wrongly. On LP it scores 0.9996 on the same real photographs used as
edit sources, so its own ceiling is not the limiting factor there.
