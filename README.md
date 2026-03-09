# ML_based_analysis_of_sound

## Machine Learning-Based Analysis of Music and Sound in Martial Arts Films

[Project tasks](https://github.com/users/hughmancoder/projects/4)

[Models](src/models/models.md)

## Setup

Install prequisites on your machine
`git, python3, pip, make`

```bash
# Create virtual environment
python -m venv .venv

# On Linux/Mac:
source .venv/bin/activate   

# On Windows (cmd.exe)
.venv\Scripts\activate.bat

# On Windows (PowerShell)
. .venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Install the local package for module-style imports in notebooks/scripts
pip install -e .

# If your environment blocks network during build isolation:
# pip install -e . --no-build-isolation
```

Activate environment (venv) on every terminal session

## Run the project

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

## Datasets

Refer to data README.md [here](data/README.md) for details on datasets
