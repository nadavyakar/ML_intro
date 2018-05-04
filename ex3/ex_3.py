import numpy as np

pic_size=784
nclasses=10
epocs=1
learning_rate=0.01
hidden_layer_size=pic_size
batch_size=1
validation_ratio=.2

def init_model():
    W1=np.matrix([[0.5] * pic_size+1] * hidden_layer_size)
    W2=np.matrix([[0.5] * hidden_layer_size+1] * nclasses)
    return (W1,W2)
class Activation:
    def __init__(self, c=1):
        self.c = c
    def __call__(self, val):
        try:
            x = 1.0 / (1.0 + exp(-self.c*val))
        except Exception:
            return 0 if val<0 else 1
        return x
    def derivative(self, output):
        return (1.0 - output) * output
activation=Activation()
def split_to_valid(train_x,train_y):
    data_set=zip(train_x, train_y)
    shuffle(data_set)
    train_size=len(data_set)-int(validation_ratio*len(data_set))
    return data_set[:train_size],data_set[train_size:]
def train(train_x,train_y):
    #todo split to validation set
    train,valid=split_to_valid(train_x,train_y)
    model=init_model()
    for e in range(epocs):
        #shuffle(train)
        for X,y in train:
            ...
    return model
def fw_prop(model,X):
    #in layer to hidden
    W1=model[0]
    l1_in_vector=np.dot(W1,X+[1])
    l1_out_vector=[activation(y) for y in l1_in_vector]
    #hidden to out layer
    W2=model[1]
    l2_in_vector=np.dot(W2,l1_out_vector+[1])
    #todo should the activation be softmax and backprop as in ex2 instead of reg activation?
    l2_out_vector=[activation(y) for y in l2_in_vector]
    return l2_out_vector
def test(model,test_x):
    '''
    :param model: a learned weight matrices. the 1st layer W matrix width=|X|=pic_size & high=hidden_layer_size
                  the 2nd layer matrix width=hidden_layer_size & high=nclasses
    :param test_x:
    :return:
    '''
    out=[]
    for X in test_x:
        out.append(fw_prop(model,X))
    return out
# train_x = np.loadtxt("train_x")
# train_y = np.loadtxt("train_y")
# test_x = np.loadtxt("test_x")
train_x = np.loadtxt("train_x_top_10")
train_y = np.loadtxt("train_y_top_10")
test_x = np.loadtxt("test_x_top_10")

model=train([[pixel/255 for pixel in pic] for pic in train_x],train_y)
out=test(model,test_x)
with open('test.pred','w') as f:
    for l in out:
        write(l)