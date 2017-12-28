from __future__ import division
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
mc = {}
for d in range(1,11):
    mc['mc'+str(d)] = 0


neigh = KNeighborsClassifier(n_neighbors=5,p = 1, n_jobs=30, leaf_size = 30)
for i in range(0,10):
    for j in range(0,5):
        train = pd.read_csv('wiki'+str(i)+'train'+str(j)+'.csv',delimiter='\t')
        test = pd.read_csv('wiki'+str(i)+'test'+str(j)+'.csv',delimiter='\t')
        tt = test.copy()
        Y_tr = train['class'].values
        del train['prev']
        del train['curr']
        del train['time']
        del train['status']
        del train['class']
        del train['status_class']
        del train['class_per']
        del train['prefAttachCategories']
        del train['jaccardCategories']
        del train['commonCategories']
        X_tr = train.values;
        Y_te_li = test['class'].values
        del test['prev']
        del test['curr']
        del test['time']
        del test['status']
        del test['class']
        del test['class_per']
        del test['status_class']
        del test['prefAttachCategories']
        del test['jaccardCategories']
        del test['commonCategories']
        X_te = test.values;
        probas_ = neigh.fit(X_tr,Y_tr).predict_proba(X_te)
        predicted_digits = neigh.predict(X_te)
        Y_p = predicted_digits
        Y_pp = probas_
        tt['pred'] = Y_pp[:,1]
        tt = tt.sort(['pred'], ascending=False)
        for d in range(1,11):
            newn = int(round(len(tt)*d*0.1))
            cf = tt.head(newn)
            mc['mc'+str(d)] = mc['mc'+str(d)] + (len(cf[cf['class']==1])/newn)
            

        

f = open('graphprecisionKN.txt','w')
for d in mc:
    f.write(str(d)+': '+str(mc[d]/50)+'\n')


f.close()
     


