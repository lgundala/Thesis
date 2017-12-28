import pandas as pd
import numpy as np
import scipy.stats
import math
from sklearn.cross_validation import StratifiedKFold, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from itertools import cycle
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

mci = 0
for i in range(0,10):
    for j in range(0,5):
        test = pd.read_csv('wiki'+str(i)+'test'+str(j)+'.csv',delimiter='\t')
        df = test.sort(['time'], ascending=False)
        Y_p = df['pref'].values
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
f = open('cinexpref.csv','w')
f.write(str(mci))
f.close()
        

        
     






