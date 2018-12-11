
import os
import re
import numpy as np


a=os.listdir("./data/mlp/")
b=filter(re.compile('mlp-mnist01s(.*)e').search, a)
lstart = []
for i in range(len(b)):
	lstart.append(int(re.split("[mlp,s,e]",b[i])[7]))# mlp 7, cnn 4	

g=open("mlp-mnist01.bin","w")
while(len(lstart)>0):
	tmp = np.argmin(lstart)
	#print b[tmp]
	f=open("./data/mlp/"+b[tmp],"r")
	TMP=f.read()
	f.close()
	g.write(TMP)
	b.remove(b[tmp])
	lstart.remove(lstart[tmp])
g.close()
