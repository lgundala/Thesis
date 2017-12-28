from __future__ import division
import pandas as pd
import numpy as np
import math
import sys
import random
mc = {}
for d in range(1,11):
    mc['mc'+str(d)] = 0

    
for i in range(0,10):
    for j in range(0,5):
        test = pd.read_csv('fb'+str(i)+'svrrbf'+str(j)+'.csv',delimiter='\t')
        test = test.sort(['svrrbfExp'], ascending=False)
        for d in range(1,11):
            newn = int(round(len(test)*d*0.1))
            cf = test.head(newn)
            mc['mc'+str(d)] = mc['mc'+str(d)] + (len(cf[cf['class']==1])/newn)
            


f = open('graphprecisionsvr.txt','w')
for d in mc:
    f.write(str(d)+': '+str(mc[d]/50)+'\n')


f.close()
