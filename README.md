# ML_based_analysis_of_sound

## Machine Learning-Based Analysis of Music and Sound in Martial Arts Films

[Project tasks](https://github.com/users/hughmancoder/projects/4)



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

# Upgrade pip tooling (recommended)
python -m pip install --upgrade pip setuptools wheel

# Install base project dependencies
python -m pip install -r requirements.txt

# Install the local package for module-style imports in notebooks/scripts
pip install -e .

# If your environment blocks network during build isolation:
# pip install -e . --no-build-isolation
```
Activate environment (venv) on every terminal session

## Setup dataset


**Film Dataset**

Add the film test dataset here: `data/test/a-touch-of-zen`

**IRMAS dataset (Pretraining)**

Download IRMAS train and test datasets and add it here: `data/IRMAS`.

**Train Dataset**

Download from teams and place in `data/train`



## Setup Dataset


## Documentation

[Generate Features](documentation/makefile.md)

[Train Models](documentation/training.md)

 [Dataset](data/README.md) 

 [Training Log](src/train/training_log.md)


