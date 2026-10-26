# Experimental RF-DETR detection

Standalone RF-DETR runner that writes OTVision `.otdet` 1.4 files, which
`track.py` then tracks as usual. It has its own environment because RF-DETR
1.7.1 needs Python 3.11 and a newer torch than OTVision.

It uses RF-DETR's TorchScript path (`optimize_for_inference`). FP16 with
batch 8 was the fastest setting on an RTX 3090 without losing accuracy.

RF-DETR ranks every query/class pair, so one query can emit the same box under
several classes. Those copies become parallel tracks and overcount road users.
The runner keeps only the best class per box.

```sh
cd experimental/rfdetr
uv sync --frozen
uv run --frozen rfdetr_detect.py \
  -p path/to/video.mp4 \
  -w path/to/checkpoint_best_ema.pth --weights-sha256 <sha256> \
  --start-time 2026-06-16_16-43-40
```

The start time is read from a `YYYY-mm-dd_HH-MM-SS` part of the file name
unless `--start-time` is given. Output goes next to the video; `-o` sets the
path for a single video. Existing outputs are skipped.

Tests run without a GPU or model: `uv run --frozen pytest`.
