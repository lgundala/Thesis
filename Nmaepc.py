from __future__ import division
from sklearn.metrics import *
from scipy.stats.stats import pearsonr
import pandas as pd
import numpy as np

mae = 0.0
pc = 0.0
for i in range(0,10):
    for j in range(0,5):
        df = pd.read_csv('wiki'+str(i)+'test'+str(j)+'nv.csv',delimiter='\t')
        y_true = df['time'].values
        df = df.replace(np.nan, 0)
        y_pred = df['newnv'].values         
        pc = pc + pearsonr(y_true, y_pred)[0]
        rgs = np.linspace(0,round(max(abs(df['newnv']))),5)
        for index, row in df.iterrows():
            d = abs(row['newnv'])
            for k in range(1,len(rgs)):
                if d>=rgs[k-1] and d<=rgs[k]:
                    df.at[index, 'newnv'] = ((d-rgs[k-1])/(rgs[k]-rgs[k-1]))*k
        mdf = df[df['status']==1]
        y_true = mdf['time'].values
        y_pred = mdf['newnv'].values
        y_pred = np.nan_to_num(y_pred)
        mae = mae + mean_absolute_error(y_true, y_pred)



f = open('maepcnv.csv','w')
f.write('mae: '+str(mae/200)+'\n')
f.write('pc: '+str(pc/50)+'\n')
f.close()
