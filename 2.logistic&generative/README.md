# Income prediction
## How to use this code?
This exercise is to show how to use Logistic Regression and Probabilistic Generative  
Model to make a classifier.
* run below to see how logistic regression works

        python logistic.py
* run below to see how probabilistic generative model works

        python generative.py
* the result will be saved in ./result

## Task description
        given some features(eg. country, gender, month's expenditure ...)  
        of some people, and you need to figure out which of them have income  
        over 50k.

## Probabilistic Generative Model
        The idea of probabilistic generative model is that you assume each of 
        the class has a specific probability distribution. When you get a data,
        you can figure out the probability that the data belongs to different 
        classes. Then, you can choose the class with highest probability.
* how to get the distrubution?

        First, you need to choose a distrubution that you think the data will
        fit. For example, month's expenditure may fit a guassian distrubution.
        After you choose the distrubution, you use training data and training
        label to find out the exact distrubution of each class. The actual 
        calculation may depend on the distrubution you choose.
        
 * calculation of guassian
 
        In this exercise, I just assume all features are guassian, so I'm  
        goint to intruduce how to calculate this case.
        
        1. Find distrubution of each class:
           For guassian, you need to find mean and sigma of each class. Where
           sigma = np.dot((train_X[i] - av_0).T, train_X[i] - av_0)
           
        2. Getting probability of test data:
           we use the formula
           prob = sigmoid(np.dot(test_x, w) + b)
           w = np.dot((av_1 - av_0).T, sigma_inverse)
           b = -0.5 * np.dot(np.dot(av_1, sigma_inverse), av_1)\
                + 0.5 * np.dot(np.dot(av_0, sigma_inverse), av_0)\
                + np.log(p_1/p_0)
        For the detail of derivation, see https://www.youtube.com/watch?v=fZAZUYEeIMg.

## Logistic Regression
        What is logistic regression? You can see that at the end of generative
        model, we get 
                prob = sigmoid(np.dot(test_x, w) + b)
        so, we can just use gradient descent to get a better w and b in order to minumize 
        the loss. Which is very similiar with what we do in 1.gradient. But, this time we 
        are going to predict a probability(0-1) but not a real number.  
 
* loss function
   
        We define 2 kind of loss, MSE and Cross_entrophy.
        
    > for MSE:    
        
        loss = (pre - lable) ** 2, 
        grad = 2 * (pre - lable) * data 
        
    > for Cross_entrophy:         
        
        loss = -1 * (label * log(pre) + (1 - label) * np.log(1 - pre))
        d_pre = d(pre)/dw = d(sig(data * w))/dw = pre * (1 - pre) * data
        grad = -1 * (label / pre * d_pre - (1 - label) / (1 - pre) * d_pre)

    To find the min of the loss, we just upgrade weight  
    
        w = w - grad * lr.
        
## useful trick   
    Some tricks is used in this work.  
    1. We normalize the value of the input so that it can converge more faster.  
    2. When applying sigmoid, you have to clip the value or it will diverge.  
    3. In generative model, we share the simga of every classes which decreases the  
        parameter and improves the performance.

* future work    
    adagrad  
