from __future__ import division
from sklearn.metrics import *
from scipy.stats.stats import pearsonr
import pandas as pd

mae = 0.0
pc = 0.0
for i in range(0,10):
    for j in range(0,5):
        df = pd.read_csv('wiki'+str(i)+'lasso'+str(j)+'E.csv',delimiter='\t')
        mdf = df[df['status']==1]
        y_true = mdf['time'].values
        y_pred = mdf['lassoExp'].values 
        mae = mae + mean_absolute_error(y_true, y_pred)
        y_true = df['time'].values
        y_pred = df['lassoExp'].values         
        pc = pc + pearsonr(y_true, y_pred)[0]


f = open('maepcLasso.csv','w')
f.write('mae: '+str(mae/200)+'\n')
f.write('pc: '+str(pc/50)+'\n')
f.close()

