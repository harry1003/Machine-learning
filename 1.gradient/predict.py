import numpy as np
import csv

from data_loader import train_data_loader, test_data_loader

def predict():
    weight = np.load("weight.npy")
    tr_data_loader = train_data_loader()
    te_data_loader = test_data_loader(tr_data_loader.mean, tr_data_loader.std)
    question = te_data_loader.get_data()
    # predict
    pre = np.dot(question, weight)
    pre = (pre * te_data_loader.std[9]) + te_data_loader.mean[9]
    for i in range(len(pre)):
        print("id:", i, ":", pre[i])
    # save file
    with open("./result/predict.csv","w") as csvfile: 
        writer = csv.writer(csvfile)      
        writer.writerow(["id","value"])
        for i in range (len(pre)):
            id_name = 'id_'
            id_name = id_name+str(i) 
            answer = float(pre[i])
            if answer < 0:
                answer= 0
            writer.writerow([id_name, answer])   
    
predict()