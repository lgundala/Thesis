import pandas as pd
import numpy as np
import math

mci = 0
for i in range(0,10):
    for j in range(0,5):
        test = pd.read_csv('wiki'+str(i)+'test'+str(j)+'.csv',delimiter='\t')
        df = test.sort(['time'], ascending=False)
        Y_p = df['jaccard'].values
        nrow = len(test)
        tt = 0
        nz = 0
        for d in range(0,nrow):
            for b in range(d,nrow):
                if d!=b:
                    h = Y_p[d]-Y_p[b]
                    if h>0:
                        tt=tt+1
                    elif h==0:
                        tt = tt+0.5
                    else:
                        tt = tt+0
                    nz=nz+1
        ci = tt/nz
        mci=mci+ci
                
        

mci = mci/50;
f = open('cinexjaccard.csv','w')
f.write(str(mci))
f.close()
        

        
     






