from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot
import numpy as np
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
from matplotlib import pyplot as plt

import threading

from tqdm import tqdm

import pandas as pd

Q_RANGE = (16,32)
F_RANGE = (35,350)
SAMPLE_RATE=2048


def read_file(fname):
    data_list = np.load(fname)

    return [TimeSeries(data, sample_rate=SAMPLE_RATE) for data in data_list]


def preprocess(d1, d2, d3, bandpass=False, lf=35, hf=350, crop=True):
    duration = 0.25

    white_d1 = d1.whiten(window=("tukey", 0.2), fduration=duration)
    white_d2 = d2.whiten(window=("tukey", 0.2), fduration=duration)
    white_d3 = d3.whiten(window=("tukey", 0.2), fduration=duration)

    if crop:
        white_d1 = white_d1.crop(duration, 2 - duration)
        white_d2 = white_d2.crop(duration, 2 - duration)
        white_d3 = white_d3.crop(duration, 2 - duration)

    # Shifting
    white_d1.shift('6.9ms')
    white_d1 *= -1

    white_d2.shift('12ms')

    if bandpass:  # bandpass filter
        bp_d1 = white_d1.bandpass(lf, hf)
        bp_d2 = white_d2.bandpass(lf, hf)
        bp_d3 = white_d3.bandpass(lf, hf)
        return bp_d1, bp_d2, bp_d3
    else:  # only whiten
        return white_d1, white_d2, white_d3


Q_RANGE = (16, 32)
F_RANGE = (35, 350)


def create_image(fname):
    strain = read_file(fname)

    processed_strain = preprocess(*strain, crop=True, bandpass=True)

    q_transforms = [strain.q_transform(qrange=Q_RANGE, frange=F_RANGE, logf=True, whiten=False)
                    for strain in processed_strain]

    img = np.zeros([q_transforms[0].shape[0], q_transforms[0].shape[1], 3], dtype=np.uint8)

    scaler = MinMaxScaler()

    for i in range(3):
        img[:, :, i] = 255 * scaler.fit_transform(q_transforms[i])

    return Image.fromarray(img).rotate(90, expand=1)


# class AsynPreprocess(threading.Thread):
#
#     def __init__(self, data):
#         self.data = data
#
#     def run(self):
#


if __name__=="__main__":
    train_labels = pd.read_csv("./g2net-gravitational-wave-detection/training_labels_with_paths.csv")


    def correct(row):
        return row["filepath"].replace("/kaggle/input", ".")

    train_labels["filepath"] = train_labels.apply(correct, axis=1)

    arr = train_labels.to_numpy()

    print(len(train_labels))
    for index in tqdm(range(1000)):
        img = create_image(arr[index][2])

        img.save("./data/train/{}/{}.jpg".format(arr[index][1], arr[index][0]))