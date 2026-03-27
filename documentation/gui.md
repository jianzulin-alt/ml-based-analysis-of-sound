# Instrument Classifier GUI



## Repository Structure

Ensure your files are organised as follows to allow the predictor to correctly resolve paths and configurations:

```text
.
├── src/
│   ├── gui/
│   │   ├── gradio_interface.py  # Gradio layout and event logic
│   │   └── predictor.py         # Config-driven inference wrapper
│   ├── models/
│   │   ├── builder.py           # Model factory (CNN, DenseNet, etc.)
│   │   └── saved_weights/       # Checkpoints and run_configs
│   └── preprocessing/           # Core DSP (Mel, CQT, Normalisation)
└── requirements.txt
```


## How to Run

The GUI must be executed as a module from the repository root to ensure all internal package imports function correctly.

### Launch Command


```bash
python -m src.gui.gradio_interface
```
