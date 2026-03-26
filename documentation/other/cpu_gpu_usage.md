## CPU Usage Is High but GPU Usage Is Low

**Short answer**: this is usually a data pipeline bottleneck, not a weak model architecture.  
The GPU is waiting for data prepared on CPU.

Common reasons:
- Mel/CQT extraction is CPU-heavy (`librosa` + `numpy`) and does not use GPU.
- DataLoader reads `.npy` files and performs stacking/normalization on CPU.
- On Windows, `num_workers > 0` can sometimes be slower due to process startup/copy overhead.
- Batch size is too small to fully utilize GPU.

Ways to speed up training:
1) **Increase batch size**
   - Usually improves GPU utilization, but increases VRAM usage.
2) **Precompute features**
   - Generate Mel/CQT before training; avoid online feature extraction in the training loop.
3) **Optimize DataLoader**
   - In `utils_mel_cqt.py` / `utils.py`, try `pin_memory=True` and `persistent_workers=True` (validate carefully on Windows).
4) **Use mixed precision**
   - If supported, enable AMP (`torch.cuda.amp.autocast`) in training.
5) **Adjust model/input scale**
   - Larger models or longer sequences can raise GPU load, but fix I/O and batch settings first.

How to identify the bottleneck:
- GPU usage stays low (for example `<30%`) while CPU is saturated: CPU/I/O bottleneck.
- GPU usage is high and VRAM is near full: model compute bottleneck.
