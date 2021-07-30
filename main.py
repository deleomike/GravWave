import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import pycbc.types.timeseries as TimeSeries

TimeSeries.TimeSeries()

import seaborn as sns
sns.set()
plt.rcParams["axes.grid"] = False

import matplotlib.mlab as mlab
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz

from IPython.display import HTML

from gwpy.timeseries import TimeSeries
from gwpy.table import Table

from tqdm.notebook import tqdm

train_labels = pd.read_csv("./g2net-gravitational-wave-detection/training_labels_with_paths.csv")
train_labels.target.value_counts()

def correct(row):
    return row["filepath"].replace("/kaggle/input", ".")

train_labels["filepath"] = train_labels.apply(correct, axis=1)

example = 1

example_strain = np.load(train_labels[train_labels.target==1].iloc[example].filepath)

white = TimeSeries(abs(example_strain[0,:]) + 1)

white.asd()

hp = white.hi