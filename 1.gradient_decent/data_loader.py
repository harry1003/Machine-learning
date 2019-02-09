import numpy as np
import csv


class train_data_loader():
    def __init__(self, normalize=True):
        self.normalize = normalize
        raw_data = readfile()
        self.mean = 0
        self.std = 1
        if normalize:
            mean, std = self._get_mean_and_std(raw_data)
            self.mean = mean
            self.std = std
        question, answer = self._turn_to_pair(raw_data)
        self.question = question
        self.answer = answer

    def load_all_data(self):
        return self.question, self.answer

    ##################################
    #       _turn_to_pair            #
    # input    [ 18, 5760 ]          #
    # question [ :, 14 * 9 + 1 ]     #
    # answer   [ :, 1 ]              #
    # turn sequence data to pairs    #
    ##################################
    def _turn_to_pair(self, raw_data):
        question = []
        answer = []
        for m in range(12):
            # our data only given 1-20 days in a month
            hour_per_month = 20 * 24
            data_per_month = \
                raw_data[:, m * hour_per_month:(m + 1) * hour_per_month]
            # cut data into (9, 1) pair
            for i in range(0, hour_per_month - 9):
                # we don't want the information of wind
                pair = data_per_month[0:14, i:i + 10]
                # check pair
                if(self.check_pair(pair)):
                    if(self.normalize):
                        pair = self.norm(pair)
                    question.append(pair[:, :9])
                    answer.append(pair[9, 9]) 
        data_size = len(question)
        question = np.array(question).reshape(data_size, -1)
        # add bias
        bias = np.ones((data_size, 1))
        question = np.concatenate((bias, question), axis=1)
        answer = np.array(answer).reshape(data_size, 1)
        return question, answer

    ##################################
    #        check_pair              #
    # input    [ 18, 10 ]            #
    # output   True or False         #
    # to clean the data              #
    ##################################
    def check_pair(self, pair):
        if np.any(pair > 300) or np.any(pair < 0):
            return False
        return True

    ##################################
    #       _get_mean_and_std        #
    # input  [ 18, 5760 ]            #
    # mean   [ 18 ]                  #
    # std    [ 18 ]                  #
    ##################################
    def _get_mean_and_std(self, raw_data):
        mean = np.mean(raw_data, axis=1)
        std = np.std(raw_data, axis=1)  
        return mean, std

    ##################################
    #        norm                    #
    # input    [ 18, 10 ]            #
    # output   [ 18, 10 ]            #
    ##################################
    def norm(self, pair):
        for i in range(len(pair)):
            pair[i] = (pair[i] - self.mean[i]) / self.std[i]
        return pair


##################################
#        readfile                #
# input  None                    #
# output [ 18, 5760 ]            #
# 5750 = 12(m) * 20(d) * 24(h)   #
##################################
def readfile(path="./data/train.csv", mode="tr"):
    pollutants = [] 
    # 18 kinds
    for i in range(0, 18):
        pollutants.append([])
    with open(path, newline='', encoding='big5') as file:
        reader = csv.reader(file)
        if mode == "tr":
            row_n = -2
        else:
            row_n = -1
        for row in reader:
            row_n = row_n + 1
            # row 1 isn't data
            if row_n >= 0:
                if mode == "tr":
                    pollutants[(row_n) % 18].extend(row[3:27])
                else:
                    pollutants[(row_n) % 18].extend(row[2:26])
    for i in range(len(pollutants)):
        pollutants[i] = turn_to_float(pollutants[i])
    pollutants = np.array(pollutants)
    return pollutants


##################################
#        turn_to_float           #
# input  [ C ]                   #
# output [ C ]                   #
# turn str to float              #
##################################
def turn_to_float(x):
    for i in range(0, len(x)):
        if x[i] == 'NR':
            x[i] = 0
        else:
            x[i] = float(x[i])
    return x


class test_data_loader():
    def __init__(self, mean, std):
        raw_data = readfile("./data/test.csv", "te")
        self.mean = mean
        self.std = std
        question = self._get_question(raw_data)
        self.question = question

    def get_data(self):
        return self.question

    def _get_question(self, raw_data):
        question = []
        for i in range(raw_data.shape[1] // 9):
            pair = raw_data[0:14, i * 9:(i + 1) * 9]
            pair = self.norm(pair)
            question.append(pair)

        data_size = len(question)
        question = np.array(question).reshape(data_size, -1)
        bias = np.ones((data_size, 1))
        question = np.concatenate((bias, question), axis=1)            
        return question

    def norm(self, pair):
        for i in range(len(pair)):
            pair[i] = (pair[i] - self.mean[i]) / self.std[i]
        return pair
