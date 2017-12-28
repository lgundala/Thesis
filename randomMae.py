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
from sklearn.model_selection import StratifiedKFold
from itertools import cycle
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import *
from sklearn.metrics import *
from scipy.stats.stats import pearsonr
import pandas as pd
pc = 0.0
rf = RandomForestClassifier(n_estimators=100, criterion="entropy",random_state=42949694,class_weight="balanced")
for i in range(0,10):
    for j in range(0,5):
        train = pd.read_csv('fb'+str(i)+'train'+str(j)+'class_persistency.csv',delimiter='\t')
        test = pd.read_csv('fb'+str(i)+'test'+str(j)+'class_persistency.csv',delimiter='\t')
        Y_tr = train['status_class'].values
        del train['user1']
        del train['user2']
        del train['time']
        del train['status']
##        del train['class']
        del train['status_class']
        X_tr = train.values;
        tt = test.copy()
        Y_te_st = test['status_class'].values;
        Y_te_per = test['class_per'].values
        Y_te_li = test['class'].values
        y_true = test['time'].values
        del test['class']
        del test['user1']
        del test['user2']
        del test['time']
        del test['status']
        del test['class_per']
        del test['status_class']
        for d in range(1,6):
            del test[str(d)]
            del test['ep'+str(d)]
        del test['FineExp']
        X_te = test.values;
        probas_ = rf.fit(X_tr,Y_tr).predict_proba(X_te)
        predicted_digits = rf.predict(X_te)
        Y_p = predicted_digits
        Y_pp = probas_
        tt['predict'] = Y_pp[:,1]
        tt.to_csv('fb'+str(i)+'test'+str(j)+'random.csv',sep='\t', index=None)
        y_true = tt['time'].values
        y_pred = Y_pp[: ,1]        
        pc = pc + pearsonr(y_true, y_pred)[0]

        
f = open('maepcRandom.csv','w')
f.write('pc: '+str(pc/50)+'\n')
f.close()
