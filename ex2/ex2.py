import numpy as np
from random import shuffle
from math import exp, sqrt, pi, pow
import matplotlib.pyplot as plt
nclass=3
samples_per_class=100
epocs=300
learning_rate=0.01

def f(x,y):
    '''
    natural distribution of x around expectation of 2*y with a variance of 1
    '''
    return exp(-pow(x-2*y,2)/2)/sqrt(2*pi)
def p_actual(y,x):
    '''
    the conditional probability of y given x, calculated using formula f
    '''
    return f(x,y=y)/sum([f(x,y=a) for a in range(1,nclass+1)])
def generate_training_set():
    '''
    generates 'samples_per_class'*'nclass' shuffled training examples naturally distributed around
    2 * class index with a variance of 1
    '''
    train=[]
    n=samples_per_class
    for a in range(1,nclass+1):
        train+=zip(np.random.normal(2*a, 1, n),[a]*n)
    shuffle(train)
    return train
def upadte(W,x,y):
    '''
    update weight matrix W given example x and y
    '''
    derivative = [.0]*nclass
    derivative_denominator=sum([exp(k) for k in np.asarray(np.dot(W, x)).reshape(-1)])
    for i in range(nclass):
        derivative_nominator=exp(np.dot(W[i, :], x))
        derivative[i] = ((derivative_nominator/derivative_denominator)-1)*x if i==y-1 \
            else (derivative_nominator/derivative_denominator)*x
    for i in range(nclass):
        W[i] -= learning_rate*derivative[i]
def train_logistic_regression(train):
    '''
    :param train:
    :return: trains a matrix of weights from one input neuron + constant bias neuron to 3 output neurons representing
    the classifications classes, for a number of 'epocs' times, per training input
    '''
    W=np.matrix([[0.5]*2]*nclass)
    for e in range(epocs):
        shuffle(train)
        for x,y in train:
            upadte(W, np.array([x,1]), y)
    def p(y, x):
        '''
        return the trained probability
        '''
        x = [x,1]
        return exp(np.dot(W[y-1,:],x)) / sum([ exp(k) for k in np.asarray(np.dot(W, x)).reshape(-1) ])
    return p

# logistic regression training
train=generate_training_set()
p_lr=train_logistic_regression(train)
# plot graph
X = np.linspace(0., 10., num=1000)
plt.plot(X, [ p_actual(1,x) for x in X ], 'bs', X, [ p_lr(1,x) for x in X ], 'rs')
plt.show()
