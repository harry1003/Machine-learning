import numpy as np

from data import train_data_loader


def train(epochs=1000, batch_size=100, lr=0.0001):
    d_l = train_data_loader()
    data, label = d_l.get_all_data()
    # init weight
    weight = np.ones((d_l.feature_num, 1))
    # start training
    for i in range (epochs):
        for batch in range (d_l.data_size//batch_size):
            d = data[batch * batch_size:(batch + 1) * batch_size]
            l = label[batch * batch_size:(batch + 1) * batch_size]
            loss, weight = train_on_batch(d, l, weight, lr)
        print("epoch:", i)
        print("cross_entropy:", loss)
        print()
    np.save("lg_weight.npy", weight)
    
    
def train_on_batch(data, label, weight, lr, loss_f="SGD"):
    pre = np.dot(data, weight)
    pre = sigmoid(pre)
    delta = 2.22044604925e-16
    cross_entropy = -1 * (np.dot(label.T, np.log(pre + delta)) + np.dot((1 - label).T, np.log(1 - pre + delta)))
    cross_entropy = cross_entropy[0][0]
    if loss_f == "SGD":
        loss = pre - label
        grad = np.dot(data.T, loss)
        weight = weight - grad * lr    
    return cross_entropy, weight
    
    
def sigmoid(z):
    z=np.clip(z,-500,500)
    return 1 / (1 + np.exp(-z))

train()