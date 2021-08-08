from gwpy.timeseries import TimeSeries
from gwpy.plot import Plot
import numpy as np
from scipy import signal
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from PIL import Image
from multiprocessing import Pool, freeze_support, RLock
from concurrent.futures import ThreadPoolExecutor
import time

from tqdm_multi_thread import TqdmMultiThreadFactory
from matplotlib import pyplot as plt

import threading

from tqdm import tqdm

import os

import pandas as pd

from pathlib import Path


def read_file(fname):
    SAMPLE_RATE = 2048
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


def create_image(fname):
    Q_RANGE = (16, 32)
    F_RANGE = (35, 350)
    # print(fname)
    strain = read_file(fname)

    processed_strain = preprocess(*strain, crop=True, bandpass=True)

    q_transforms = [strain.q_transform(qrange=Q_RANGE, frange=F_RANGE, logf=True, whiten=False)
                    for strain in processed_strain]

    img = np.zeros([q_transforms[0].shape[0], q_transforms[0].shape[1], 3], dtype=np.uint8)

    scaler = MinMaxScaler()

    for i in range(3):
        img[:, :, i] = 255 * scaler.fit_transform(q_transforms[i])

    return Image.fromarray(img).rotate(90, expand=1).resize((768, 768))

def preprocess_group(data, target_directory, isTest, pid):

    tqdm_text = "#" + "{}".format(pid).zfill(3)

    with tqdm(total=len(data), desc=tqdm_text, position=pid + 1) as pbar:
        for i in tqdm(range(len(data))):
            # print(data)
            # time.sleep(0.05)
            if isTest:
                img = create_image(str(data[i]))
            else:
                img = create_image(data[i][0])

            if isTest:
                img.save("{}/{}.jpg".format(target_directory, data[i].name.split(".")[0]))
            else:
                img.save("{}/{}/{}.jpg".format(target_directory, data[i][1], data[i][2]))

            pbar.update()

def async_preprocess(ids, paths, targets, target_directory, num_threads):
    if ids is None and targets is None:
        isTest = True
    else:
        isTest = False

    if isTest:
        data = np.array(paths)
    else:
        data = np.array([paths, targets, ids]).transpose()

    print(len(data))
    leftover_length = len(data) % num_threads
    print(leftover_length)
    if leftover_length > 0:
        split_data = np.split(data[:-leftover_length], num_threads)
        split_data.append(data[-leftover_length:])
    else:
        split_data = np.split(data, num_threads)

    threads = []

    pool = Pool(processes=num_threads, initargs=(RLock(),), initializer=tqdm.set_lock)

    convertJobs = [pool.apply_async(preprocess_group, args=(selected_data, target_directory, isTest, i,))
                   for i, selected_data in enumerate(split_data)]
    pool.close()
    result_list = [job.get() for job in convertJobs]


if __name__=="__main__":
    train_labels = pd.read_csv("./g2net-gravitational-wave-detection/training_labels_with_paths.csv")


    def correct(row):
        return row["filepath"].replace("/kaggle/input", ".")

    train_labels["filepath"] = train_labels.apply(correct, axis=1)

    arr = train_labels.to_numpy()

    print(len(arr))

    t_arr = np.transpose(arr)

    id_train, id_val, target_train, target_val, path_train, path_val = \
        train_test_split(*t_arr, test_size=0.1, train_size=0.9)

    print(len(id_train), len(id_val))

    os.system("mkdir -p ./data")

    os.system("rm -r ./data/*")

    os.system("mkdir -p ./data/test")
    os.system("mkdir -p ./data/train/0")
    os.system("mkdir -p ./data/validation/0")
    os.system("mkdir -p ./data/train/1")
    os.system("mkdir -p ./data/validation/1")

    test_paths = list(Path("./g2net-gravitational-wave-detection/test").rglob("*.npy"))

    # Save training lists for easy access

    np.save("./data/train_info.npy", np.array([id_train, target_train]).transpose())

    # Save validation lists for easy access
    np.save("./data/validation_info.npy", np.array([id_val, target_val]).transpose())

    num_threads = 40

    # Train Images
    async_preprocess(ids=id_train, paths=path_train, targets=target_train, num_threads=num_threads, target_directory="./data/train")

    # Val Images
    async_preprocess(ids=id_val, paths=path_val, targets=target_val, num_threads=num_threads, target_directory="./data/validation")

    # Test Images
    async_preprocess(ids=None, paths=test_paths, targets=None, num_threads=num_threads, target_directory="./data/test")

    # Train Images
    # for index in tqdm(range(len(id_train))):
    #     img = create_image(path_train[index])
    #
    #     img.save("./data/train/{}/{}.jpg".format(target_train[index], id_train[index]))

    # Validation Images
    #for index in tqdm(range(len(id_val))):
    #    img = create_image(path_val[index])

    #    img.save("./data/validation/{}/{}.jpg".format(target_val[index], id_val[index]))

    # for test_path in tqdm(test_paths):
    #     img = create_image(str(test_path))
    #
    #     img.save("./data/test/{}.jpg".format(test_path.name.split(".")[0]))