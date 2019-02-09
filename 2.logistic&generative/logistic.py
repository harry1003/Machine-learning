import numpy as np
import csv

from data import train_data_loader, test_data_loader
delta = 2.22044604925e-16


def train(epochs=1000, batch_size=100, lr=0.001):
    d_l = train_data_loader()
    data, label = d_l.get_all_data()
    # init weight
    weight = np.ones((d_l.feature_num, 1))
    # start training
    for i in range(epochs):
        for batch in range(d_l.data_size//batch_size):
            d = data[batch * batch_size:(batch + 1) * batch_size]
            l = label[batch * batch_size:(batch + 1) * batch_size]
            loss, weight = train_on_batch(d, l, weight, lr)
        print("epoch:", i)
        print("cross_entropy:", loss)
        print()
    np.save("./result/lg_weight.npy", weight)


def predict():
    d_l = test_data_loader()
    data = d_l.get_all_data()
    weight = np.load("./result/lg_weight.npy")
    pre = np.dot(data, weight)
    pre = sigmoid(pre)

    # save file
    with open("./result/lg_predict.csv", "w") as csvfile:
        writer = csv.writer(csvfile)      
        writer.writerow(["id", "label"])
        for i in range(len(pre)):
            id_name = str(i + 1)
            answer = float(pre[i])
            if answer > 0.5:
                answer = 1
            else:
                answer = 0
            writer.writerow([id_name, answer])


def train_on_batch(data, label, weight, lr, loss_f="cross_entropy"):
    pre = np.dot(data, weight)
    pre = sigmoid(pre)
    cross_entropy = -1 * (np.dot(label.T, np.log(pre + delta)) + \
                    np.dot((1 - label).T, np.log(1 - pre + delta)))
    cross_entropy = cross_entropy[0][0]
    if loss_f == "SGD":
        loss = pre - label
        grad = np.dot(data.T, loss)
        weight = weight - grad * lr
    if loss_f == "cross_entropy":
        d1 = pre * (1 - pre) * data
        d2 = -1 * pre * (1 - pre) * data
        a1 = label / (pre + delta)
        a2 = (1 - label) / (1 - pre + delta)
        grad = -1 * (np.dot(d1.T, a1) + np.dot(d2.T, a2))
        weight = weight - grad * lr

    return cross_entropy, weight


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    train()
    predict()
