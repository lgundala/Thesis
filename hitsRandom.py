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
for i in range(0,10):
    for j in range(0,5):
        train = pd.read_csv('fb'+str(i)+'train'+str(j)+'class_persistency.csv',delimiter='\t')
        test = pd.read_csv('fb'+str(i)+'test'+str(j)+'class_persistency.csv',delimiter='\t')
        Y_tr = train['status_class'].values

        X_tr = train['hits'].values;
        X_tr = X_tr.reshape(-1,1)
        Y_te_st = test['status_class'].values;
        Y_te_per = test['class_per'].values
        Y_te_li = test['class'].values

        X_te = test['hits'].values;
        X_te = X_te.reshape(-1,1)
        probas_ = rf.fit(X_tr,Y_tr).predict_proba(X_te)
        predicted_digits = rf.predict(X_te)
        Y_p = predicted_digits
        Y_pp = probas_
        
        accuracy_8 = accuracy_8+accuracy_score(Y_te_st, Y_p);
        fpr, tpr, thresholds = roc_curve(Y_te_st, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        roc_accuracy_8 = roc_accuracy_8 + roc_auc;
        apc_8 = apc_8 + average_precision_score(Y_te_st, Y_pp[:,1])
        p1_8 = p1_8+precision_score(Y_te_st, Y_p, average='binary', pos_label=1)
        p0_8 = p0_8+precision_score(Y_te_st, Y_p, average='binary', pos_label=0)
        
        accuracy_9per = accuracy_9per+accuracy_score(Y_te_per, Y_p);
        fpr, tpr, thresholds = roc_curve(Y_te_per, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        roc_accuracy_9per = roc_accuracy_9per + roc_auc;
        apc_9per = apc_9per + average_precision_score(Y_te_per, Y_pp[:,1])
        p1_9per = p1_9per+precision_score(Y_te_per, Y_p, average='binary', pos_label=1)
        p0_9per = p0_9per+precision_score(Y_te_per, Y_p, average='binary', pos_label=0)
        
        accuracy_9link = accuracy_9link+accuracy_score(Y_te_li, Y_p);
        fpr, tpr, thresholds = roc_curve(Y_te_li, probas_[:, 1])
        roc_auc = auc(fpr, tpr)
        roc_accuracy_9link = roc_accuracy_9link + roc_auc;
        apc_9link = apc_9link + average_precision_score(Y_te_li, Y_pp[:,1])
        p1_9link = p1_9link+precision_score(Y_te_li, Y_p, average='binary', pos_label=1)
        p0_9link = p0_9link+precision_score(Y_te_li, Y_p, average='binary', pos_label=0)


f = open('fbresultsExp1randomhits.csv','w')
f.write('using 2008\n')
f.write('accuracy: '+str(accuracy_8/50)+'\n')
f.write('roc: '+str(roc_accuracy_8/50)+'\n')
f.write('precision score class 1 : '+str(p1_8/50)+'\n')
f.write('precision score class 0 : '+str(p0_8/50)+'\n')
f.write('average precision score: '+str(apc_8/50)+'\n')

f.write('using 2009 persistency\n')
f.write('accuracy: '+str(accuracy_9per/50)+'\n')
f.write('roc: '+str(roc_accuracy_9per/50)+'\n')
f.write('precision score class 1 : '+str(p1_9per/50)+'\n')
f.write('precision score class 0 : '+str(p0_9per/50)+'\n')
f.write('average precision score: '+str(apc_9per/50)+'\n')

f.write('using 2009 link\n')
f.write('accuracy: '+str(accuracy_9link/50)+'\n')
f.write('roc: '+str(roc_accuracy_9link/50)+'\n')
f.write('precision score class 1 : '+str(p1_9link/50)+'\n')
f.write('precision score class 0 : '+str(p0_9link/50)+'\n')
f.write('average precision score: '+str(apc_9link/50)+'\n')
f.close()
        

        
     






