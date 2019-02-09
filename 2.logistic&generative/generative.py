import csv
import numpy as np

from data import train_data_loader, test_data_loader


def predict(av_0, av_1, sigma, p_0, p_1, test_x):
    # Naive bayes classifier
    sigma_inverse = np.linalg.inv(sigma)
    w = np.dot((av_1 - av_0).T, sigma_inverse)
    b = -0.5 * np.dot(np.dot(av_1, sigma_inverse), av_1)\
        + 0.5 * np.dot(np.dot(av_0, sigma_inverse), av_0)\
        + np.log(p_1/p_0)

    prob = sigmoid(np.dot(test_x, w) + b)

    # save file
    with open("./result/ge_predict.csv", "w") as csvfile:
        writer = csv.writer(csvfile)      
        writer.writerow(["id", "label"])
        for i in range(len(prob)):
            id_name = str(i + 1)
            answer = float(prob[i])
            if answer > 0.5:
                answer = 1
            else:
                answer = 0
            writer.writerow([id_name, answer])


def train(train_X, train_Y):
    train_Y = train_Y.reshape(-1)
    # mean
    av_0 = np.dot(train_X.T, train_Y - 1) * (-1) / len(train_X)
    av_1 = np.dot(train_X.T, train_Y) / len(train_X)

    # get sigma and probability
    c_0 = 0
    c_1 = 0
    sigma_0 = 0
    sigma_1 = 0

    for i in range(len(train_X)):
        if train_Y[i] == 0:
            # class 0
            temp = train_X[i] - av_0
            temp2 = temp[np.newaxis, :]
            temp = temp[:, np.newaxis]
            sigma_0 = sigma_0 + np.dot(temp, temp2)
            c_0 = c_0 + 1
        elif train_Y[i] == 1:
            # class 1
            temp = train_X[i] - av_1
            temp2 = temp[np.newaxis, :]
            temp = temp[:, np.newaxis]
            sigma_1 = sigma_1 + np.dot(temp, temp2)
            c_1 = c_1 + 1
        else:
            print("input data error:", i, " th Y is", train_Y[i])

        if i % 1000 == 0 and i != 0:
            print("process:", i, '/', len(train_x), " ", i/len(train_X), "%")

    sigma_0 = sigma_0 / len(train_X)
    sigma_1 = sigma_1 / len(train_X)

    p_0 = c_0 / (c_0 + c_1)
    p_1 = c_1 / (c_0 + c_1)

    # force two classes use same sigma to reduce parameters
    # which will get a better result
    sigma = sigma_0 * p_0 + sigma_1 * p_1

    return av_0, av_1, sigma, p_0, p_1


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    train_d = train_data_loader(normalize=False)
    train_x, train_y = train_d.get_all_data()
    av_0, av_1, sigma, p_0, p_1 = train(train_x, train_y)
    test_d = test_data_loader(normalize=False)
    test_x = test_d.get_all_data()
    predict(av_0, av_1, sigma, p_0, p_1, test_x)
