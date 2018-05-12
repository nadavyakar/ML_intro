import numpy as np
import random as rnd
from math import log, exp
import pickle
import sys

pic_size=784
nclasses=10
epocs=100
learning_rate=0.05
layer_sizes=[pic_size,100,nclasses]
batch_size=1
validation_ratio=.2
weight_init_boundry=0.08
Y=dict([(y,[ 1 if i==y else 0 for i in range(nclasses)]) for y in range(nclasses) ])
rnd.seed(1)
def init_model():
    return [ np.matrix([[rnd.uniform(-weight_init_boundry,weight_init_boundry)] * (layer_sizes[l]+1)] * layer_sizes[l+1]) for l in range(len(layer_sizes)-1) ]
class ActivationSigmoid:
    def __call__(self, IN_VEC):
        X=[]
        for val in IN_VEC:
            try:
                X.append(1. / (1. + exp(-val)))
            except Exception:
                X.append(0. if val<0 else 1.)
        return X
    def derivative(self, out):
        return (1.0 - out) * out
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
activation=[ActivationInputIdentity(), ActivationSigmoid(), ActivationSoftmax()]
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
    return data_set[:train_size],data_set[train_size:]
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
def bk_prop(W,X,Y):
    out=fw_prop(W,X)
    err=loss.derivative_z(out[-1], Y)
    loss_derivative_in = err
    for l in list(reversed(range(len(layer_sizes))))[:-1]:
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
def train(train_x,train_y):
    train,valid=split_to_valid(train_x,train_y)
    W=init_model()
    for e in range(epocs):
        rnd.shuffle(train)
        for X,y in train:
            bk_prop(W,X,Y[y])
        avg_loss,acc=validate(W, valid)
        print "epoc {} avg loss {} accuracy {}".format(e,avg_loss,acc)
    return W
def test_and_write(model,test_x):
    '''
    :param model: a learned weight matrices. the 1st layer W matrix width=|X|=pic_size & high=hidden_layer_size
                  the 2nd layer matrix width=hidden_layer_size & high=nclasses
    :param test_x:
    :return:
    '''
    print("test")
    with open('test.pred', 'w') as f:
        for X in test_x:
            f.write("{}\n".format(fw_prop(model,X)[-1].argmax()))
train_x = np.loadtxt("train_x")
train_y = np.loadtxt("train_y")
test_x = np.loadtxt("test_x")
# train_x = np.loadtxt("train_x_top_10")
# train_y = np.loadtxt("train_y_top_10")
# test_x = np.loadtxt("test_x_top_10")

#todo del:
with open(r"data.txt", "wb") as f:
    f.write(pickle.dumps((train_x, train_y, test_x)))
sys.exit(0)
data = open(r"data.txt", "rb").read()
train_x,train_y,test_x = pickle.loads(data)



W=train([[pixel/255 for pixel in pic] for pic in train_x],train_y)
test_and_write(W,test_x)