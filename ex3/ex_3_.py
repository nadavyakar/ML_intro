import numpy as np
import random as rnd
from math import log, exp
import pickle
import sys
import matplotlib.pyplot as plt
import logging

pic_size=1
nclasses=3
batch_size=1
validation_ratio=0
epocs=[300]
learning_rates=[0.01]
architectures=[[pic_size,nclasses]]
weight_init_boundries=[0.5]
from math import exp, sqrt, pi, pow

Y=dict([(y,[ 1 if i==y else 0 for i in range(nclasses)]) for y in range(nclasses) ])
rnd.seed(1)

# logger=logging.getLogger(__name__)
logging.basicConfig(filename="/home/nadav/data/nn.log",level=logging.ERROR)

def init_model(params):
    layer_sizes, weight_init_boundry = params
    return [ np.matrix([[0.5] * (layer_sizes[l]+1)] * layer_sizes[l+1]) for l in range(len(layer_sizes)-1) ]
class ActivationSoftmax:
    def __call__(self, IN_VEC):
        denominator = sum([exp(v) for v in IN_VEC])
        return np.array([exp(v) / denominator for v in IN_VEC ])
    def derivative(self, out):
        raise Error("ERROR: you should not have gotten here ActivationSoftmax")
class ActivationInputIdentity:
    def __call__(self, IN_VEC):
        return IN_VEC
    def derivative(self, out):
        return np.array([.0,])
activation=[ActivationInputIdentity(), ActivationSoftmax()]
class LossNegLogLikelihood:
    def __call__(self, V, y):
        return -log(V[int(y)])
    def derivative_z(self, out, Y):
        return out-Y
loss=LossNegLogLikelihood()
def split_to_valid(train_x,train_y):
    data_set=zip(train_x, train_y)
    rnd.shuffle(data_set)
    train_size=len(data_set)-int(validation_ratio*len(data_set))
    return data_set[:train_size],data_set[:train_size]
def fw_prop(W,X):
    out_prev = X
    # print("fw prop layer 0: {}".format(X))
    out=[out_prev]
    for i in range(len(W)):
        x_in = np.dot(W[i],np.append(out_prev,1.))
        out_prev=activation[i+1](x_in.reshape(x_in.shape[1],-1))
        out.append(np.array(out_prev))
        # print("fw prop layer {}: {}".format(i+1,out_prev))
    return out
def bk_prop(W,X,Y,nlayers,learning_rate):
    out=fw_prop(W,X)
    err=loss.derivative_z(out[-1], Y)
    loss_derivative_in = err
    for l in list(reversed(range(nlayers)))[:-1]:
        out_prev=np.append(out[l-1],1.)
        loss_derivative_W_l = np.outer(loss_derivative_in, out_prev) # calculate a matrix of the same shape as W[l] used to update it
        err=np.squeeze(np.asarray(np.dot(np.transpose(W[l-1]), loss_derivative_in))) # pass the delta of layer l neurons down to layer l-1 neurons
        W[l-1]-=learning_rate*loss_derivative_W_l # update layer l-1 to layer l weights
        if l>1:
            loss_derivative_in = activation[l-1].derivative(out_prev)[:-1]*err[:-1] # G_l = loss_derivative_z
def validate(W,valid):
    sum_loss= 0.0
    correct=0.0
    for X, y in valid:
        out = fw_prop(W,X)
        sum_loss += loss(out[-1],y)
        if out[-1].argmax() == y:
            correct += 1
    return sum_loss/ len(valid),correct/ len(valid)
def train(W,train_x,train_y,params):
    starting_epoc, ending_epoc, learning_rate, layer_sizes, batch_size, weight_init_boundry, avg_loss_list, avg_acc_list = params
    train,valid=split_to_valid(train_x,train_y)
    for e in range(starting_epoc,ending_epoc):
        rnd.shuffle(train)
        for X,y in train:
            bk_prop(W,X,Y[y],len(layer_sizes),learning_rate)
        avg_loss,acc=validate(W, valid)
        avg_loss_list.append(avg_loss)
        avg_acc_list.append(acc)
        logging.error("epoc {} avg loss {} accuracy {}".format(e,avg_loss,acc))
    epocs_list=list(range(ending_epoc))
    plt.plot(epocs_list,avg_loss_list,'bs',epocs_list,avg_acc_list,'rs')
    plt.xlabel("epocs")
    plt.savefig("/home/nadav/data/perf.e_{}.lr_{}.hs_{}.bs_{}.w_{}.png".format(ending_epoc,learning_rate,layer_sizes[1],batch_size,weight_init_boundry))
    plt.clf()
    def p(y, x):
        '''
        return the trained probability
        '''
        X = [x,1]
        return exp(np.dot(W[-1][y-1,:],X)) / sum([ exp(k) for k in np.asarray(np.dot(W[-1], X)).reshape(-1) ])
    return W, avg_loss_list, avg_acc_list,p

def f(x,y):
    '''
    natural distribution of x around expectation of 2*y with a variance of 1
    '''
    return exp(-pow(x-2*y,2)/2)/sqrt(2*pi)
def p_actual(y,x):
    '''
    the conditional probability of y given x, calculated using formula f
    '''
    return f(x,y=y)/sum([f(x,y=a) for a in range(1,nclasses+1)])
def generate_training_set():
    '''
    generates 'samples_per_class'*'nclass' shuffled training examples naturally distributed around
    2 * class index with a variance of 1
    '''
    train=[]
    n=100
    for a in range(1,nclasses+1):
        train+=zip(np.random.normal(2*a, 1, n),[a]*n)
        rnd.shuffle(train)
    return train

train_x=[]
train_y=[]
for x,y in generate_training_set():
    train_x.append(x)
    train_y.append(y)

for weight_init_boundry in weight_init_boundries:
    for layer_sizes in architectures:
        for learning_rate in learning_rates:
            params=layer_sizes, weight_init_boundry
            W = init_model(params)
            avg_loss_list = []
            avg_acc_list = []
            starting_epoc=0
            for ending_epoc in epocs:
                params = starting_epoc, ending_epoc, learning_rate, layer_sizes, batch_size, weight_init_boundry, avg_loss_list, avg_acc_list
                logging.error("start training with params: e_{}.lr_{}.hs_{}.bs_{}.w_{}".format(ending_epoc,learning_rate,layer_sizes[1],batch_size,weight_init_boundry))
                W, avg_loss_list, avg_acc_list, p =train(W,train_x,[ y-1 for y in train_y],params)
                starting_epoc = ending_epoc
                X = np.linspace(0., 10., num=1000)
                plt.plot(X, [p_actual(1, x) for x in X], 'bs', X, [p(1, x) for x in X], 'rs')
                plt.show()