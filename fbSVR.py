from sklearn.svm import SVR
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
mci_rbf = 0
mci_lin = 0
mci_poly = 0
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=2)
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
        Y_te_st = test['status_class'].values;
        Y_te_per = test['class_per'].values
        Y_te_li = test['class'].values
        del test['user1']
        del test['user2']
        del test['time']
        del test['status']
        del test['class']
        del test['class_per']
        del test['status_class']
        for d in range(1,6):
            del test[str(d)]
            del test['ep'+str(d)]
        del test['FineExp']
        X_te = test.values
        y_rbf = svr_rbf.fit(X_tr, Y_tr).predict(X_te)
        y_lin = svr_lin.fit(X_tr, Y_tr).predict(X_te)
        y_poly = svr_poly.fit(X_tr, Y_tr).predict(X_te)
        nrow = len(test)
        Y_p = y_rbf
        tt = 0
        nz = 0
        for d in range(0,nrow):
            for b in range(d,nrow):
                if d!=b:
                    h = Y_p[d]-Y_p[b]
                    if h>0:
                        tt=tt+1
                    elif h ==0:
                        tt = tt+0.5
                    else:
                        tt = tt+0
                    nz=nz+1
        ci = tt/nz
        mci_rbf=mci_rbf+ci
        Y_p = y_lin
        tt = 0
        nz = 0
        for d in range(0,nrow):
            for b in range(d,nrow):
                if d!=b:
                    h = Y_p[d]-Y_p[b]
                    if h>0:
                        tt=tt+1
                    elif h ==0:
                        tt = tt+0.5
                    else:
                        tt = tt+0
                    nz=nz+1
        ci = tt/nz
        mci_lin=mci_lin+ci
        Y_p = y_poly
        tt = 0
        nz = 0
        for d in range(0,nrow):
            for b in range(d,nrow):
                if d!=b:
                    h = Y_p[d]-Y_p[b]
                    if h>0:
                        tt=tt+1
                    elif h ==0:
                        tt = tt+0.5
                    else:
                        tt = tt+0
                    nz=nz+1
        ci = tt/nz
        mci_poly=mci_poly+ci

f = open('fbSVR.csv','w')
f.write("svr rbf "+'\t'+str(mci_rbf/50)+'\n')
f.write("svr linear "+'\t'+str(mci_lin/50)+'\n')
f.write("svr poly "+'\t'+str(mci_poly/50)+'\n')
f.close()


        
        
        
        
