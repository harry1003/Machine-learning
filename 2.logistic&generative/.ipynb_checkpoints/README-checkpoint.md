# Task description
> Given you people's data with features(127), and you have to predict   
> if their income is over 50k

# Logistic
This exercise is to show how use logistic model to predict
* run below to see how we train

        python logistic.py 

* What is logistic?  
    > Very similiar with task in (1.gradient).    
    > But, this time we are going to predict a probability(0~1).    
    > To transform the output from R->(0~1) we add a sigmoid function   
    > to it, just define at below.   
    
        sig(z) = 1 / (1 + np.exp(-z))
    > After the transform, we can use the tech in (1.gradient) to change   
    > the weight.(output and label's value are in range(0~1))
    
    In this example, we just use one layer weight to try to predict. We use   
    gradient desent again, so that   
    We define   
        
        pre = sig(data * w)
        loss = (pre - lable) ** 2, 
    so the gradient of this loss is   
    
        grad = 2 * (pre - lable) * data 
    To find the min of the loss, we just upgrade weight  
    
        w = w - grad * lr.
* useful trick   
    Some tricks is used in this work.  
    1. We normalize the value of the input so that it can converge more faster.  
    2. When applying sigmoid, you have to clip the value or it will diverge.  
* future work
    adagrad  
    cross_entropy_loss
    
