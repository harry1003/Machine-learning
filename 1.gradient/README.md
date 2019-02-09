## Prediction of PM2.5  
# How to use this code?
This exercise is to show how to use Gradient Descent to optimize the 
weight in the model.
* run below to see how we train weight

        python train.py 
* run below to predict test.csv, answer will save in ./result

        python predict.py

# Task description
This task is to predict PM2.5 value.   
We will give you the value of some features  
(eg. SO2, PM2.5, PM2...) in first 9 hrs,   
and you have to predict the value of   
PM2.5 in the 10th hr.  

# Gradient  

* What is gradient descent?  
    > Gradient descent is widely used in maching learning.  
    > The concept is find the gradient of loss, and then go backward.  
    > So, the loss can reach a local min, where may have a good result.  
    
    In this example, we just use one layer weight to try to predict PM2.5.  
    We define   
    
        loss = (question * w - answer) ** 2, 
    so the gradient of this loss is   
    
        2 * (question * w - answer) * question. 
    To find the min of the loss, we just upgrade weight  
    
        w = w - grad * lr.
* useful trick   
    Some tricks is used in this work.  
    1. We normalize the value of the input so that it can converge more faster.  
    2. Besides, we use adagrad, so that the value of lr can adjust itself to some extent.   
        However, if you didn't use this tech, you have to be careful to choose your lr.   
        If your lr is too big, loss will diverge.  
* future work
    close form  
    validation    
    The result of this is not good to reach the strong baseline since I didn't clean the data very well in this code.  
    To improve, you may need to clean the data more carefully. The data is quite dirty. I almost delete 30% of data to  
    reach strong baseline.
