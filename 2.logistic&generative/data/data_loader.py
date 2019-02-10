import pandas as pd
import numpy as np


class train_data_loader():
    def __init__(self, mean=0, std=1, normalize=True, bias=True):
        self.mean = mean
        self.std = std
        # load
        train_data = pd.read_csv("./data/X_train", sep=',', header=0)
        train_data = np.array(train_data, dtype=float)
        label = pd.read_csv("./data/Y_train", sep=',', header=0)
        label = np.array(label, dtype=float)
        # shuffle
        self.data_size = len(train_data)
        train_data, label = self.shuffle(train_data, label)
        train_data = train_data.reshape(self.data_size, -1)
        label = label.reshape(self.data_size, 1)
        # normalize
        if normalize:
            t_d = train_data.T
            mean = np.mean(t_d, axis=1)
            std = np.std(t_d, axis=1)
            self.mean = mean
            self.std = std
            for i in range(len(t_d)):
                t_d[i] = (t_d[i] - mean[i]) / std[i]
            train_data = t_d.T

        if bias:
            # add bias
            bias = np.ones((self.data_size, 1))
            train_data = np.concatenate((bias, train_data), axis=1)

        self.feature_num = train_data.shape[1]
        self.train_data = train_data
        self.label = label

    def get_all_data(self):
        return self.train_data, self.label

    def shuffle(self, x, y):
        id = np.arange(len(x))
        id = np.random.shuffle(id)
        return x[id], y[id]


class test_data_loader():
    def __init__(self, mean=0, std=1, normalize=True, bias=True):
        self.mean = mean
        self.std = std
        # load
        data = pd.read_csv("./data/X_test", sep=',', header=0)
        data = np.array(data, dtype=float)

        # normalize
        if normalize:
            t_d = data.T
            mean = np.mean(t_d, axis=1)
            std = np.std(t_d, axis=1)
            for i in range(len(t_d)):
                if std[i] == 0:
                    t_d[i] = (t_d[i] - mean[i])
                else:
                    t_d[i] = (t_d[i] - mean[i]) / std[i]
            data = t_d.T
        # add bias
        if bias:
            self.data_size = len(data)
            bias = np.ones((self.data_size, 1))
            data = np.concatenate((bias, data), axis=1)

        self.feature_num = data.shape[1]
        self.data = data

    def get_all_data(self):
        return self.data
