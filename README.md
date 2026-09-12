# Ultralytics-Usage

A small demo of **Ultralytics YOLO's custom-callback hooks** — the
extension point YOLO exposes for running your own code at specific points
in the training/prediction lifecycle, without forking the library.

## What this shows

- Registering callbacks on a `YOLO` model instance via `model.add_callback(event, fn)`.
- `on_train_epoch_end` / `on_train_batch_end` — logging custom
  per-epoch/per-batch training statistics (loss, learning rate, timing)
  as training runs.
- `on_predict_batch_end` — pairing each inference result with its source
  frame during prediction.

This is the pattern you'd reach for to stream training metrics to an
experiment tracker, save custom checkpoints, or post-process predictions,
without modifying Ultralytics' own training loop.

## Docs

- Callback events reference: https://docs.ultralytics.com/usage/callbacks/
- `YOLO.add_callback`: https://docs.ultralytics.com/reference/engine/model/

See [Ultralutics_Yolo_Custom_Callbacks.ipynb](Ultralutics_Yolo_Custom_Callbacks.ipynb)
for the full notebook (requires a GPU/CPU-heavy `model.train()` call —
not run as part of this repo's test suite).
