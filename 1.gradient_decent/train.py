import numpy as np

from data_loader import train_data_loader


def main():
    t_data_loader = train_data_loader()
    question, answer = t_data_loader.load_all_data()
    train(question, answer, 300000, 1, True)


##################################
#        readfile                #
#  Input                         #
# qustion [ :, 14 * 9 + 1 ]      #
# answer  [ :, 1 ]               #
#  Var                           #
# weight  [ 14 * 9 + 1, 1 ]      # 
##################################
def train(question, answer, training_times=3000, lr=0.00001, adag=True):
    # init weight, move
    weight = np.zeros(question[0].shape).reshape(-1, 1)
    # start get gradient
    if adag:
        grad_a = np.zeros(question[0].shape).reshape(-1, 1)
    for t in range(training_times):
        # [:,14 * 9 + 1] * [14 * 9 + 1, 1] = [:, 1]
        pre = np.dot(question, weight)
        loss = pre - answer
        cost = np.sum(np.sqrt((loss ** 2) / len(pre)))
        grad = np.dot(question.T, loss)
        if adag:
            grad_a = grad_a + grad ** 2
            weight = weight - grad * lr / np.sqrt(grad_a)
        else:
            weight = weight - grad * lr
        if (t % 100 == 0):
            print("epochs:", t)
            print("loss:", cost)
    np.save("./result/weight.npy", weight)


if __name__ == "__main__":
    main()
