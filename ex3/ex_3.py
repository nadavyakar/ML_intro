import numpy as np

pic_size=784
nclasses=10
epocs=1
learning_rate=0.01
hidden_layer_size=pic_size
batch_size=1
validation_ratio=.2
Y=dict([(y,[ 1 if i==y else 0 for i in range(nclasses)]) for y in range(nclasses) ])

def init_model():
    W1=np.matrix([[0.5] * pic_size+1] * hidden_layer_size)
    W2=np.matrix([[0.5] * hidden_layer_size+1] * nclasses)
    return (W1,W2)
class Sigmoid:
    def __call__(self, IN_VEC):
        X=[]
        for val in IN_VEC:
            try:
                X.append(1.0 / (1.0 + exp(-self.c*val)))
            except Exception:
                X.append(0 if val<0 else 1)
        return X
    def derivative(self, output):
        return (1.0 - output) * output
class Softmax:
    def __call__(self, V):
        denominator = sum([exp(v) for v in V])
        return [exp(v) / denominator for v in V ]
    def derivative(self, V, Y):
        return V-Y

activation=[Sigmoid(), Softmax()]
def split_to_valid(train_x,train_y):
    data_set=zip(train_x, train_y)
    shuffle(data_set)
    train_size=len(data_set)-int(validation_ratio*len(data_set))
    return data_set[:train_size],data_set[train_size:]
def bk_prop(model,X,Y,W):
    V=fw_prop(model,X)
    #update weights to last layer
    err = activation[1].derivative(V[1],Y)
    derivative = np.dot(err, V[0])
    W[1]-=learning_rate*derivative
    #update weights to hidden layer
    err=np.dot(W[1],err)
    g=np.dot(activation[0].derivative(V[0]),err)
    derivative = np.dot(g, X+[1])
    W[0]-=learning_rate*derivative

def train(train_x,train_y):
    train,valid=split_to_valid(train_x,train_y)
    model=init_model()
    for e in range(epocs):
        #shuffle(train)
        for X,y in train:
            bk_prop(model,X,Y[y],W)
    return W
def fw_prop(W,X):
    V={}
    #in layer to hidden
    l1_in_vector=np.dot(W[0],X+[1])
    V[0]=activation[0](l1_in_vector)
    #hidden to out layer
    l2_in_vector=np.dot(W[1],V[0]+[1])
    V[1]=activation[1](l2_in_vector)
    return V
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