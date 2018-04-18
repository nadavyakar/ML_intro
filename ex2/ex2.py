import sys
import numpy as np
from random import shuffle
from math import exp, sqrt, pi, pow
import matplotlib.pyplot as plt
nclass=3
nsamples_per_class=100
# epocs=300
epocs=0
learning_rate=0.01
gap=1
def f(x,y):
    return exp(-pow(x-2*y,2)/2)/sqrt(2*pi)
    #todo per normal dist formula, if incorrect - check how to get it from numpy
def p_actual(y,x):
    return f(x,y=y)/sum([f(x,y=a) for a in range(1,nclass+1)])
def generate_training_set():
    train=[]
    n=nsamples_per_class
    for a in range(1,nclass+1):
        train+=zip(np.random.normal(2*a, 1, n),[a]*n)
    #todo remove normalization
    l=[x for x,y in train]
    # gap=max(l)-min(l)
    train = [(x/gap,y) for x,y in train]
    shuffle(train)
    return train
def upadte(W,x,y):
    derivative = [.0]*nclass
    #derivative_denominator=sum([exp(k) for k in np.asarray(np.dot(W, x)).reshape(-1)])
    a = np.asarray(np.dot(W, x)).reshape(-1)
    derivative_denominator=sum([exp(k) for k in a ])
    print "x {}  y {} W*x {} denominator {}".format(x, y, a,derivative_denominator)
    for i in range(nclass):
        # derivative_nominator=exp(np.dot(W[i, :], x))
        derivative_nominator=exp(np.dot(W[i, :], x))
        derivative[i] = ((derivative_nominator/derivative_denominator)-1)*x if i==y-1 \
            else (derivative_nominator/derivative_denominator)*x
        print "i: {} W(i): {} W(i)*x: {} nominator: {} derivative: {}".format(i, W[i, :].reshape(-1), np.dot(W[i, :], x).reshape(-1), derivative_nominator, derivative[i])
    for i in range(nclass):
        W[i] -= learning_rate*derivative[i]
def train_logistic_regression(train):
    W=np.matrix([[0.5]*2]*nclass)
    #todo remove
    W=np.matrix([[-1.4203355,7.53263104],[ 0.56437152,1.35344333],[ 2.35596398,-7.38607437]])
    # W = np.matrix([[-4.54144343,2.49196813],[0.59017353,0.71689935],[5.4512699,-1.70886748]])
    print "{} W: {}".format(-1, W)
    for e in range(epocs):
        shuffle(train)
        #todo remove
        W_=np.matrix([[1.0]*2]*nclass)
        for q in [0,1,2]:
            for w in [0,1]:
                W_[q,w]=W[q,w]
        for x,y in train:
            upadte(W, np.array([x,1]), y)
            # todo remove
#        print "{} W: {}\ndW {}".format(e, W, W-W_)
            print "{} W: {}".format(e, W)

    def p(y, x):
        x = [x/gap,1]
#        p_x_given_y = exp(np.dot(W[y,:],x)) / sum([ exp(k) for k in np.asarray(np.dot(W, x)).reshape(-1) ])
        a = exp(np.dot(W[y-1,:],x))
        b = sum([ exp(k) for k in np.asarray(np.dot(W, x)).reshape(-1) ])
        p_x_given_y = a / b
        p_y = 0.3
        p_x = 0.1
        return p_x_given_y
        # print "x {} y {} nominator {} denominator {} p(x|y) {}".format(x[0],y,a,b,p_x_given_y)
        # return p_x_given_y * p_y / p_x
    return p
def generate_dummy_training_set():
    return [(.0,0)]*10+[(.5,1)]*10+[(1.,2)]*10#)+([(1,1)]*1)
# logistic regression training
train=generate_training_set()
# train=generate_dummy_training_set()

#todo delme: count training set
cells={}
count={}
#for a in [1, 2, 3]:
for a in []:
    X=sorted([x for x,y in train if y==a])
    mina=min(X)
    maxa=max(X)
    interval=(maxa-mina)/100
    cells[a]=np.linspace(mina,maxa,100)
    count[a]={}
    for i in range(len(cells[a])-1):
        count[a][i]=0
        for x in X:
            if x>cells[a][i]:
                break
            if x>cells[a][i-1]:
                count[a][i]+=1
    print "a {} min {} max {} interval {} len count {} len cells {}".format(a,mina,maxa,interval,len(count[a]),len(cells[1]))

# plt.plot(cells[1][:-1], [ count[1][x] for x in range(len(cells[1])-1) ], 'bs',
#          cells[2][:-1], [ count[2][x] for x in range(len(cells[2])-1) ], 'rs',
#          cells[3][:-1], [ count[3][x] for x in range(len(cells[3])-1) ], 'gs')
# plt.show()
# sys.exit(0)
p_lr=train_logistic_regression(train)
# for a in [0,1,2]:
#     for x in [.0,.5,1.]:
#         print "p(x={}|y={})={}".format(x,a,p_lr(a,x))
# sys.exit(0)

# plot graph
X = np.linspace(0., 10., num=100)
plt.plot(X, [ p_actual(1,x) for x in X ], 'bs', X, [ p_lr(1,x) for x in X ], 'rs')
plt.show()
