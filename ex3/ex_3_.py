import numpy as np
import random as rnd
from math import log, exp
import pickle
import sys
import matplotlib.pyplot as plt
import logging

class ActivationSigmoid:
    def __call__(self, IN_VEC):
        X=[]
        for val in IN_VEC:
            try:
                X.append(1. / (1. + exp(-val)))
            except Exception:
                x = 0. if val<0 else 1.
                X.append(x)
                logging.debug("sigmoid for {} is {}".format(len(X)-1,x))
        return np.array(X)
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

pic_size=784
nclasses=10
train_x = np.loadtxt("train_x_top_100")
train_y = np.loadtxt("train_y_top_100")
test_x = np.loadtxt("train_x_top_10")
architectures=[[pic_size,50,nclasses]]
activation=[ActivationInputIdentity(), ActivationSigmoid(), ActivationSoftmax()]
tst_name="top_100"

# fails:
# pic_size=2
# nclasses=2
# train_x=[[0,0],
#          [0,1],
#          [1,0],
#          [1,1]]*10
# train_y=[0,1,1,0]*10
# architectures=[[pic_size,50,nclasses]]
# activation=[ActivationInputIdentity(), ActivationSigmoid(), ActivationSoftmax()]
# tst_name="xor"

# success:
# nclasses=4
# pic_size=2
# train_x=[[0,0],
#          [0,1],
#          [1,0],
#          [1,1]]
# train_y=[0,1,2,3]
# architectures=[[pic_size,4,nclasses]]
# activation=[ActivationInputIdentity(), ActivationSigmoid(), ActivationSoftmax()]
# tst_name="identity"

# success:
# nclasses=2
# pic_size=2
# train_x=[[0,1],
#          [1,0]]
# train_y=[0,1]
# architectures=[[pic_size,2,nclasses]]
# activation=[ActivationInputIdentity(), ActivationSigmoid(), ActivationSoftmax()]
# tst_name="identity"

# success:
# nclasses=2
# pic_size=2
# train_x=[[0,1],
#          [1,0]]
# train_y=[0,1]
# architectures=[[pic_size,nclasses]]
# activation=[ActivationInputIdentity(), ActivationSoftmax()]
# tst_name="identity"

# train_x=[np.array(X) for X in train_x]

batch_size=1
validation_ratio=.2
epocs=[100]
learning_rates=[0.01]
weight_init_boundries=[0.05]
from math import exp

Y=dict([(y,[ 1 if i==y else 0 for i in range(nclasses)]) for y in range(nclasses) ])
# rnd.seed(1)

# logger=logging.getLogger(__name__)
logging.basicConfig(filename="/home/nadav/data/nn.log",level=logging.ERROR)

def init_model(params):
    layer_sizes, weight_init_boundry = params
    return [ np.matrix([[rnd.uniform(-weight_init_boundry,weight_init_boundry) for i in range(layer_sizes[l]+1)] for j in range(layer_sizes[l+1])]) for l in range(len(layer_sizes)-1) ]
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
    IN = X
    layers_out=[IN]
    for i in range(len(W)):
        WdotIN = np.dot(W[i],np.append(IN,1.))
        OUT=activation[i+1](WdotIN.reshape(WdotIN.shape[1],-1))
        layers_out.append(np.array(OUT))
        logging.debug("fw prop layer {}:\nin\n{}\n\nW\n{}\n\nW*in\n{}\n\nout\n{}\n".format(i,IN,W[i],WdotIN,OUT))
        IN=OUT
    return layers_out
def bk_prop(W,X,Y,nlayers,learning_rate):
    logging.debug("back prop:\nX\n{}\n\nY\n{}\n\n".format(X, Y))
    out_list=fw_prop(W,X)
    err=loss.derivative_z(out_list[-1], Y) # Y_hat - Y
    dLdZ = err # Z is a vec of sums of multipications of weights with prev layer output to last l
    dldz_str="dLdZ{} = out_list{} - Y = \n{} - {} =\n{}\n\n".format(2,len(out_list)-1,out_list[-1],Y,dLdZ)
    # ayer neurons
    for l in list(reversed(range(nlayers)))[:-1]:
        logging.debug("bk prop layer {}:".format(l))
        out_prev=np.append(out_list[l-1],1.) # out_prev = V(l-1)
        logging.debug(dldz_str)
        logging.debug("V{}\n{}\n\n".format(l-1,out_prev))
        dLdW = np.outer(dLdZ, out_prev) # calculate a matrix of the same shape as W[l] used to update it
        logging.debug("dLdW{} = dLdZ{} outer V{}\n{}\n\n".format(l,l,l-1,dLdW))
        err=np.squeeze(np.asarray(np.dot(np.transpose(W[l-1]), dLdZ))) # pass the delta of layer l neurons down to layer l-1 neurons
        logging.debug("err{} = T(W{}) dot dLdZ{}\n{}\n\n".format(l-1,l-1,l,err))
        # err=np.asarray(np.dot(np.transpose(W[l-1]), dLdZ)) # pass the delta of layer l neurons down to layer l-1 neurons
        W[l-1]-=learning_rate*dLdW # update layer l-1 to layer l weights
        logging.debug("W{} -= {}*dLdW{}\n{}\n\n".format(l - 1, learning_rate, l, dLdW))
        if l>1:
            dLdZ = activation[l-1].derivative(out_prev[:-1])*err[:-1] # G_l
            dldz_str="dLdZ{} =\nactivation_{}'({})*err{}=\n( 1 - {} )* {} * {} =\n{}\n\n".format(l - 1, l-1,out_prev, l-1,out_prev,out_prev,err,activation[l-1].derivative(out_prev))


# def bprop(W,X,Y,nlayers,learning_rate):
#   out_list = fw_prop(W, X)
#   x=X
#   y=Y
#   h1=out_list[1]
#   h2=out_list[2]
#   # z1,
#   dz2 = (h2 - y)                                #  dL/dz2
#   dW2 = np.dot(dz2, h1.T)                       #  dL/dz2 * dz2/dw2
#   db2 = dz2                                     #  dL/dz2 * dz2/db2
#   sigmoid=ActivationSigmoid()
#   dz1 = np.dot(W[1].T,(h2 - y))
#   dz1=dz1* h1*(1-h1)#sigmoid(z1) * (1-sigmoid(z1))   #  dL/dz2 * dz2/dh1 * dh1/dz1
#   #np.dot(fprop_cache['W2'].T,(h2 - y)) * sigmoid(z1) * (1 - sigmoid(z1))
#   #-----err---------------------------     ----=h2?--    -----=1-h2?-----   - check cal of new dLdZ!
#   dW1 = np.dot(dz1, x.T)                        #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
#   db1 = dz1                                     #  dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
#   return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}

def validate(W,valid):
    sum_loss= 0.0
    correct=0.0
    for X, y in valid:
        logging.debug("validating:\nX {}\ny {}\n".format(X,y))
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
            # bprop(W,X,Y[y],len(layer_sizes),learning_rate)
        avg_loss,acc=validate(W, valid)
        avg_loss_list.append(avg_loss)
        avg_acc_list.append(acc)
        logging.error("epoc {} avg loss {} accuracy {}".format(e,avg_loss,acc))
    epocs_list=list(range(ending_epoc))
    plt.plot(epocs_list,avg_loss_list,'bs',epocs_list,avg_acc_list,'rs')
    plt.xlabel("epocs")
    plt.savefig("/home/nadav/data/perf.tst_{}.e_{}.lr_{}.hs_{}.bs_{}.w_{}.png".format(tst_name,ending_epoc,learning_rate,layer_sizes[1],batch_size,weight_init_boundry))
    plt.clf()
    return W, avg_loss_list, avg_acc_list
def test_and_write(model,test_x,params):
    '''
    :param model: a learned weight matrices. the 1st layer W matrix width=|X|=pic_size & high=hidden_layer_size
                  the 2nd layer matrix width=hidden_layer_size & high=nclasses
    :param test_x:
    :return:
    '''
    starting_epoc, epoc, learning_rate, layer_sizes, batch_size, weight_init_boundry, avg_loss_list, avg_acc_list = params
    with open("/home/nadav/data/test.tst_{}.e_{}.lr_{}.hs_{}.bs_{}.w_{}".format(tst_name,epoc,learning_rate,layer_sizes[1],batch_size,weight_init_boundry), 'w') as f:
        for X in test_x:
            p=fw_prop(model,X)[-1].argmax()
            print("{}:{}".format(X,p))
            f.write("{}".format(p))
#train_x = np.loadtxt("train_x")
#train_y = np.loadtxt("train_y")
#test_x = np.loadtxt("test_x")

#with open(r"data.txt", "wb") as f:
#    f.write(pickle.dumps((train_x, train_y, test_x)))
#sys.exit(0)
# data = open(r"/home/nadav/data/data.txt", "rb").read()
# train_x,train_y,test_x = pickle.loads(data)
#train_x=[[pixel/255 for pixel in pic] for pic in train_x]
test_x =train_x

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
                logging.error("start training with params: tst_{}.e_{}.lr_{}.hs_{}.bs_{}.w_{}".format(tst_name,ending_epoc,learning_rate,layer_sizes[1],batch_size,weight_init_boundry))
                W, avg_loss_list, avg_acc_list =train(W,train_x,train_y,params)
                starting_epoc = ending_epoc
                # with open(r"/home/nadav/data/W.e_{}.lr_{}.hs_{}.bs_{}.w_{}".format(ending_epoc,learning_rate,layer_sizes[1],batch_size,weight_init_boundry),"wb") as f:
                #     f.write(pickle.dumps((W,avg_loss_list, avg_acc_list)))
                #pW=open(r"/home/nadav/data/W.e_{}.lr_{}.hs_{}.bs_{}.w_{}".format(epocs,learning_rate,layer_sizes[1],batch_size,weight_init_boundry)).read()
                #W,avg_loss_list, avg_acc_list=pickle.loads(pW)
                test_and_write(W,test_x,params)
