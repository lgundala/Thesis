from __future__ import division
import pandas as pd
import numpy as np
import math
import sys

i = int(sys.argv[0])
j = int(sys.argv[1])
test = pd.read_csv('fb'+str(i)+'test'+str(j)+'class_persistency.csv',delimiter='\t')
nrow = test[test['status']==1].index
tt = 0.0
nz = 0.0
for d in nrow:
    for b in range(d, len(test)):
        if d!=b and test.at[d,'time']!=test.at[b,'time']:
            h = test.at[d,'FineExp']-test.at[b,'FineExp']
            if h >0:
                tt = tt+1
                tt5 = tt5+1
            elif h<0:
                tt = tt+0
            nz = nz+1

ci = tt/nz
return ci
