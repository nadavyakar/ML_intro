import sys
import numpy as np
import random
import itertools
# <command> <conjuct len> <amount of examples>

def apply(h,x):
	for i in range(len(x)):
		if (i in h and True in h[i] and x[i]==0) or (i in h and False in h[i] and x[i]==1):
			return 0
	return 1
d = int(sys.argv[1])
h={}
empty=True
while empty:
	for state,x in zip([ random.randint(1,2) for i in range(d) ],range(d)):
		if state==2:
			h[x]={True}
			empty=False
		elif state==1:
			h[x]={False}
			empty=False
		else:
			h[x]={}
with open("data_{}.txt".format(d),'w') as f:
	for ones_amount in range(d+1):
		for permutation in list(itertools.permutations(([0]*(d-ones_amount))+([1]*ones_amount))):
			out_str=""
			for x in permutation:
				out_str+="{} ".format(x)
			out_str+=str(apply(h,permutation))
			f.write(out_str+"\n")
out_str=""
for x in sorted(h.keys()):
        if True in h[x]:
                out_str+="x{},".format(x+1)
        if False in h[x]:
                out_str+="not(x{}),".format(x+1)
print out_str[:-1]

