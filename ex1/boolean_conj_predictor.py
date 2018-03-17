import sys
import numpy as np

def apply(h,x):
	for i in range(len(x)):
		if (i in h and True in h[i] and x[i]==0) or (i in h and False in h[i] and x[i]==1):
			return 0
	return 1

training_examples = np.loadtxt(sys.argv[1])
X = training_examples[:,:-1]
Y = training_examples[:,-1]

#consistency algo
h=dict([ (x,{True,False}) for x in range(len(X[0])) ])
for t in range(len(X)):
	if Y[t]==1 and apply(h,X[t])==0:
		for i in range(len(X[t])):
			# if the literal Xi doesn't exists already in our hypothesis
			if X[t][i]==1:
				if False in h[i]:
					h[i].remove(False)
			else:
                               	if True in h[i]:
                                       	h[i].remove(True)

#print results to file
out_str=""
for x in sorted(h.keys()):
	if True in h[x]:
		out_str+="x{},".format(x+1)
	if False in h[x]:
                out_str+="not(x{}),".format(x+1)
if len(out_str)==0:
	out_str=" "
with open('output.txt','w') as f:
	f.write(out_str[:-1])
