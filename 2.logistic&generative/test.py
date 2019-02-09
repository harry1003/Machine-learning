import csv
import numpy as np

def sigmoid(z):
    z=np.clip(z,-500,500)
    return 1 / (1 + np.exp(-z))


def accuracy(train_X,train_Y,av_0,av_1,sigma,p_0,p_1):
    sigma_inverse = np.linalg.inv(sigma)
    w = np.dot( (av_1-av_0).T, sigma_inverse )    
    b = -0.5*np.dot(np.dot(av_1,sigma_inverse),av_1)+0.5*np.dot(np.dot(av_0,sigma_inverse),av_0)+np.log(p_1/p_0)
    
    prob = sigmoid(np.dot(train_X,w)+b)    
    prob = prob>0.5
    prob = (prob == train_Y)
    right = np.sum(prob)/len(train_X)    
    return right

def turn_to_float( x ):
    for i in range (0,len(x)):
        x[i]=float(x[i])

def readfile():
    train_X=[]
    train_Y=[]
    
    with open("./data/X_train", newline='',encoding='big5') as file:
        reader = csv.reader(file)
        row_n = -2;
        for row in reader:
            row_n=row_n+1;
            if row_n>=0:
               train_X.append([])
               train_X[row_n].append(1)      #add bias
               train_X[row_n].extend(row)    #add data
               turn_to_float(train_X[row_n])
               train_X[row_n]=np.array(train_X[row_n])  
    train_X=np.array(train_X)           
    with open("./data/Y_train", newline='',encoding='big5') as file:
        reader = csv.reader(file)
        row_n = -2;
        for row in reader:
            row_n=row_n+1;
            if row_n>=0:
                train_Y.append(float(row[0]))    
    train_Y = np.array(train_Y)            
    return train_X,train_Y

def train(train_X,train_Y):

    print(train_X)
    # print(train_Y.shape)
    #mean
    av_0 = np.dot( train_X.T , train_Y-1)*(-1)/len(train_X) 
    av_1 = np.dot( train_X.T , train_Y )/len(train_X) 
    
    
    #sigma
    c_0 = 0
    c_1 = 0    
    sigma_0 = 0    
    sigma_1 = 0
      
    
    for i in range (0,len(train_X)):
        if train_Y[i]==0: #class 0            
            temp = train_X[i]-av_0
            temp2 = temp[np.newaxis,:]
            temp = temp[:,np.newaxis]
            sigma_0 = sigma_0 + np.dot( temp,temp2 )
            c_0 = c_0+1
        elif train_Y[i]==1 : #class 1
            temp = train_X[i]-av_1
            temp2 = temp[np.newaxis,:]
            temp = temp[:,np.newaxis]
            sigma_1 = sigma_1 + np.dot( temp,temp2)
            c_1 = c_1+1
        else:
            print("input data error:",i," th Y is",train_Y[i])
           
    sigma_0 = sigma_0/len(train_X) 
    sigma_1 = sigma_1/len(train_X)
        
    p_0 = c_0/(c_0+c_1)
    p_1 = c_1/(c_0+c_1)    
    
    sigma = sigma_0*p_0 +sigma_1*p_1
        
    return av_0,av_1,sigma,p_0,p_1
    
def main():
    train_X,train_Y=readfile()    
    av_0,av_1,sigma,p_0,p_1 = train(train_X,train_Y)   
    acc = accuracy(train_X,train_Y,av_0,av_1,sigma,p_0,p_1)
    print("accurancy:",acc)
    
main()    