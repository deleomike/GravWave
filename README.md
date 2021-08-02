# Gravitational Wave Detection - Machine Learning

## Requirements

### Make the environment

Python
```bash
python -m venv env
. ./env/bin/activate
```

Conda
```bash
conda create -n GravWave python=3.7
conda activate GravWave
```

### Install the requirements

For a system that has CUDA
```bash
pip install -r requirements-cuda.txt
```

For a cpu only system
```bash
pip install -r requirements-cpu.txt
```

## Data
Download the data (69GB)
```bash
kaggle competitions download -c g2net-gravitational-wave-detection
```

Visualize the data with the visualization jupyter notebook.

To do a full setup of the data on a remote device

```bash
. ./full-data-setup.sh
```

## Preprocess the data

```bash
python preprocess.py
```

This script processes the time series data and puts it into spectrogram jpg format in the ./data directory.

## Train

```bash
python main.py
```

