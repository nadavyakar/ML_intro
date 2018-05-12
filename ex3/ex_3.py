import numpy as np
from random import shuffle
from math import log, exp

pic_size=784
nclasses=10
epocs=1
learning_rate=0.01
layer_sizes=[pic_size,20,nclasses]
batch_size=1
validation_ratio=.2
Y=dict([(y,[ 1 if i==y else 0 for i in range(nclasses)]) for y in range(nclasses) ])

def init_model():
    return [ np.matrix([[0.5] * (layer_sizes[l]+1)] * layer_sizes[l+1]) for l in range(len(layer_sizes)-1) ]
class Sigmoid:
    def __call__(self, IN_VEC):
        X=[]
        for val in IN_VEC:
            try:
                X.append(1.0 / (1.0 + exp(-val)))
            except Exception:
                X.append(0 if val<0 else 1)
        return X
    def derivative(self, V):
        return (1.0 - V) * V
class Softmax:
    def __call__(self, IN_VEC):
        denominator = sum([exp(v) for v in IN_VEC])
        return [exp(v) / denominator for v in IN_VEC ]
    def derivative(self, V, Y):
        return [v-y for v,y in zip(V,Y)]
activation=[None,Sigmoid(), Softmax()]
def loss(V,y):
    return -log(V[y])
def split_to_valid(train_x,train_y):
    data_set=zip(train_x, train_y)
    #shuffle(data_set)
    train_size=len(data_set)-int(validation_ratio*len(data_set))
    return data_set[:train_size],data_set[train_size:]
def fw_prop(W,X):
    out_prev = X
    out=[out_prev]
    for i in range(len(W)):
        x_in = np.dot(W[i],np.append(out_prev,1))
        out_prev=activation[i+1](x_in.reshape(x_in.shape[1],-1))
        out.append(np.array(out_prev))
    return out
def bk_prop(W,X,Y):
    out=fw_prop(W,X)
    err=Y
    for l in list(reversed(range(len(layer_sizes))))[:-1]:
        loss_derivative_in = activation[l].derivative(out[l], err) # G_l = loss_derivative_z
        loss_derivative_W_l = np.outer(loss_derivative_in, np.append(out[l-1],1)) # calculate a matrix of the same shape as W[l] used to update it
        err=np.dot(np.transpose(W[l-1]), loss_derivative_in) # pass the delta of layer l neurons down to layer l-1 neurons
        W[l-1]-=learning_rate*loss_derivative_W_l # update layer l-1 to layer l weights
def validate(W,valid):
    sum_loss= 0.0
    correct=0.0
    for X, y in valid:
        out = fw_prop(W,X)
        sum_loss += loss(out[-1],y)
        if out[-1].index(max(out[-1])) == y:
            correct += 1
    return sum_loss/ len(valid),correct/ len(valid)
def train(train_x,train_y):
    train,valid=split_to_valid(train_x,train_y)
    W=init_model()
    for e in range(epocs):
        #shuffle(train)
        for X,y in train:
            bk_prop(W,X,Y[y])
        print "epoc {} avg loss,accuricy {}".format(e,validate(W,valid))
    return W
def test(model,test_x):
    '''
    :param model: a learned weight matrices. the 1st layer W matrix width=|X|=pic_size & high=hidden_layer_size
                  the 2nd layer matrix width=hidden_layer_size & high=nclasses
    :param test_x:
    :return:
    '''
    out=[]
    for X in test_x:
        out.append(fw_prop(model,X)[-1])
    return out
# train_x = np.loadtxt("train_x")
# train_y = np.loadtxt("train_y")
# test_x = np.loadtxt("test_x")
train_x = np.loadtxt("train_x_top_10")
train_y = np.loadtxt("train_y_top_10")
test_x = np.loadtxt("test_x_top_10")


W=train([[pixel/255 for pixel in pic] for pic in train_x],train_y)
out=test(W,test_x)
with open('test.pred','w') as f:
    for l in out:
        write(l)