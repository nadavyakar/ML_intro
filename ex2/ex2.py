#import sys
import numpy as np
from random import shuffle
from math import exp, sqrt, pi, pow
nclass=3
nsamples_per_class=100
epocs=1
learning_rate=.1

def f(x,y):
    return exp(-pow(x-2*y,2)/2)/sqrt(2*pi)
    #todo per normal dist formula, if incorrect - check how to get it from numpy
def p_actual(y,x):
    return f(x,y=y)/sum([f(x,y=a) for a in range(1,nclass+1)])
def generate_training_set():
    training_set=[]
    n=nsamples_per_class
    for a in range(1,nclass+1):
        training_set+=zip([a]*n,np.random.normal(2a, 1, n))
    shuffle(training_set)
    return training_set
def generate_dummy_training():
    return [(1,2),(2,4),(3,6)]
def upadte(W,x,y):
    derivative = [.0]*nclass
    derivative_denominator=sum([exp(k) for k in np.asarray(np.dot(W, x)).reshape(-1)])
    for i in range(nclass):
        derivative_nominator=exp(np.dot(W[i, :], x))
        derivative[i] = ((derivative_nominator/derivative_denominator)-1)*x if i==y \
            else (derivative_nominator/derivative_denominator)*x
    for i in range(nclass):
        W[i] -= learning_rate*derivative[i]
def train_logistic_regression(train):
    W=np.matrix([[0]*2]*nclass)
    for e in range(epocs):
        for x,y in train:
            upadte(W, np.array([x,1]), y)
    def p(y, x):
        return exp(np.dot(W[y,:],x)) / sum([ exp(k) for k in np.asarray(np.dot(W, x)).reshape(-1) ])
    return p
# logistic regression training
train=generate_dummy_training()
#train=generate_training_set()
p_lr=train_logistic_regression(train)

# plot graph
plot 0..10 for p_lr & p_actual