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
accuracy_8 = 0;
apc_8 = 0
roc_accuracy_8 = 0;
p1_8 = 0
p0_8= 0
accuracy_9per = 0;
apc_9per = 0
roc_accuracy_9per = 0;
p1_9per = 0
p0_9per= 0
accuracy_9link = 0;
apc_9link = 0
roc_accuracy_9link = 0;
p1_9link = 0
p0_9link= 0
rf = RandomForestClassifier(n_estimators=100, criterion="entropy",random_state=42949694,class_weight="balanced")
mci = 0
for i in range(0,10):
    for j in range(0,5):
        train = pd.read_csv('wiki'+str(i)+'train'+str(j)+'.csv',delimiter='\t')
        test = pd.read_csv('wiki'+str(i)+'test'+str(j)+'.csv',delimiter='\t')
        Y_tr = train['status_class'].values
        del train['prev']
        del train['curr']
        del train['time']
        del train['status']
        del train['class']
        del train['status_class']
        del train['class_per']
        del train['commonCategories']
        del train['jaccardCategories']
        del train['prefAttachCategories']
        X_tr = train.values;
        Y_te_st = test['status_class'].values;
        del test['prev']
        del test['curr']
        del test['time']
        del test['status']
        del test['class']
        del test['class_per']
        del test['status_class']
        del test['commonCategories']
        del test['jaccardCategories']
        del test['prefAttachCategories']
        X_te = test.values;
        probas_ = rf.fit(X_tr,Y_tr).predict_proba(X_te)
        predicted_digits = rf.predict(X_te)
        Y_p = predicted_digits
        Y_pp = probas_
        nrow = len(test)
        tt = 0
        nz = 0
        for d in range(0,nrow):
            for b in range(d,nrow):
                if d!=b:
                    h = Y_pp[:,1][d]-Y_pp[:,1][b]
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
f = open('wikiresultscindexrandom.csv','w')
f.write(str(mci))
f.close()
        

        
     






