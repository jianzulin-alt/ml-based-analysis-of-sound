### Launch the GUI


```bash
make run_gradio_gui
```

This launches the two-tab Gradio app (Model + Info) using the fine-tuned weights at `saved_weights/chinese_single_class/train_1/best_val_acc.pt` by default. Upload or record a ~3 second clip, inspect the generated mel spectrogram, and review the predicted class
probabilities in the browser.

### Evaluate A Checkpoint And Save Results

```bash
python -m src.test.test \
  --checkpoint src/models/saved_weights/MobileNetV3_v1/best_val.pt \
  --test_manifest data/test/a-touch-of-zen.csv \
  --auto_threshold
```

This writes:
- `classification_report.csv`
- `predictions.csv`
- `summary.json`

to a timestamped folder under the checkpoint directory, and appends a one-line summary to `src/models/test_results.csv`.
