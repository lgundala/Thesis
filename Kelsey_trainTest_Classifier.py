import pandas as pd
import numpy as np
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score, accuracy_score,matthews_corrcoef
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import linear_model, svm, gaussian_process
import sklearn.metrics as met
import nltk
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import *
from sklearn.metrics import accuracy_score
import random

def robustSwap(x_tef, feature, rows1, rows0):
    j = 1;
    for i in range(0,len(rows1)):
        x_tef.loc[rows1[i],feature] = x_tef.loc[rows0[i],feature];
    j = feature + 8;
    while(j<609):
        for i in range(0,len(rows1)):
            x_tef.loc[rows1[i],j] = x_tef.loc[rows0[i],j];
        j = j+6;
    j = (feature-1)*5 + 610;
    for k in range(0,5):
        for i in range(0,len(rows1)):
            x_tef.loc[rows1[i],j] = x_tef.loc[rows0[i],j];
        j = j+1;
    return x_tef;
def randomSwap(x_tef,rows1, rows0):
    j = 1;
    feature = random.randint(1,6);
    for i in range(0,len(rows1)):
        x_tef.loc[rows1[i],feature] = x_tef.loc[rows0[i],feature];
    
    j = feature + 8;
    while(j<609):
        for i in range(0,len(rows1)):
            x_tef.loc[rows1[i],j] = x_tef.loc[rows0[i],j];
        j = j+random.randint(1,6);
    j = (random.randint(1,6)-1)*5 + 610;
    while(j<640):
        for i in range(0,len(rows1)):
            x_tef.loc[rows1[i],j] = x_tef.loc[rows0[i],j];
        j = j+random.randint(1,6);
    return x_tef;    

def main():
    accuracy1 = 0;
    accuracy2 = 0;
    accuracy3 = 0;
    accuracy4 = 0;
    roc_accuracy1 = 0;
    roc_accuracy2 = 0;
    roc_accuracy3 = 0;
    roc_accuracy4 = 0;
    kappa1 = 0;
    matthews1 = 0;
    precision1_0 = 0;
    precision2_0 = 0;
    precision3_0 = 0;
    precision4_0 = 0;
    precision1_1 = 0;
    precision2_1 = 0;
    precision3_1 = 0;
    precision4_1 = 0;
    recall1_0 = 0;
    recall2_0 = 0;
    recall3_0 = 0;
    recall4_0 = 0;
    recall1_1 = 0;
    recall2_1 = 0;
    recall3_1 = 0;
    recall4_1 = 0;
    f1measure1_0 = 0;
    f1measure2_0 = 0;
    f1measure3_0 = 0;
    f1measure4_0 = 0;
    f1measure1_1 = 0;
    f1measure2_1 = 0;
    f1measure3_1 = 0;
    f1measure4_1 = 0;
    K = 10
    for i in range(0,10):
##        print(i);
        x_tr = pd.read_csv("forAmulya/640/all_test_"+str(i)+"new.csv",delimiter=",", header=None, encoding="utf-8");
        x_te = pd.read_csv("forAmulya/640/all_train_"+str(i)+"new.csv",delimiter=",", header=None, encoding="utf-8");
        ####uncomment below region if you want to change size of test w.r.t train and change the lines to extract y
##        x_tr_1 = x_tr.loc[x_tr[8]==1]
##        x_tr_0 = x_tr.loc[x_tr[8]==0]
##        l = round((len(x_tr_1.index))*0.1)
##        rows = random.sample(list(x_tr_1),l )
##        x_tr_11 = x_tr_1.ix[rows];
##        frames = [x_tr_11, x_tr_0];
##        x_trf = pd.concat(frames);
        x_te_1 = x_te.loc[x_te[8]==1]
        x_te_0 = x_te.loc[x_te[8]==0]
        l = round((len(x_te_1.index)))
        print(l)
        rows = random.sample(list(x_te_1),l )
        x_te_11 = x_te_1.ix[rows];
        frames = [x_te_11, x_te_0];
        x_tef = pd.concat(frames);
        ###### to test system robustness
        x_ter = x_tef.loc[x_tef[8]==1];       
        l = round(len(x_ter)*0.01);
        rows1 = random.sample(list(x_ter.index),l);
        x_ter0 = x_tef.loc[x_tef[8]==0];
        rows0 = random.sample(list(x_ter0.index),l);
##        x_tef = x_te;
        for i in range(5):
            x_tef = robustSwap(x_tef, i,rows1, rows0);
##            x_tef = randomSwap(x_tef,rows1, rows0);
        y_te = x_tef[8].values;
        del x_tef[8]
        del x_tef[609];
        y_tr = x_tr[8].values;
        del x_tr[8];
        del x_tr[609];
        x = x_tr.values;
        X_tr=x[:,1:640];
        Y_tr = y_tr;
        x = x_tef.values;
        X_te = x[:,1:640];
        Y_te = y_te;
        rf = RandomForestClassifier(n_estimators=100, criterion="entropy",random_state=0)
        rf.fit(X_tr, Y_tr)
        predicted_digits = rf.predict(X_te)
        Y_p1 = rf.predict_proba(X_te)
        Y_p = predicted_digits
##        accuracy1 = accuracy1+accuracy_score(Y_te, Y_p);
##        print(accuracy_score(Y_te, Y_p));
##        precision1_0 = precision1_0 + precision_score(Y_te, Y_p, average='binary', pos_label=0);
##        precision1_1 = precision1_1 + precision_score(Y_te, Y_p, average='binary', pos_label=1);
##        recall1_0 = recall1_0 + recall_score(Y_te, Y_p, average='binary', pos_label=0);
##        recall1_1 = recall1_1 + recall_score(Y_te, Y_p, average='binary', pos_label=1);       
##        print(precision_score(Y_te, Y_p, average='binary', pos_label=1));
##        print(precision_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(recall_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(recall_score(Y_te, Y_p, average='binary', pos_label=1));
##        f1measure1_0 = f1measure1_0 + f1_score(Y_te, Y_p, average='binary', pos_label=0)
##        f1measure1_1 = f1measure1_1 + f1_score(Y_te, Y_p, average='binary', pos_label=1)
####        print(f1_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(f1_score(Y_te, Y_p, average='binary', pos_label=1));
        roc_accuracy1 = roc_accuracy1 + met.roc_auc_score(Y_te, Y_p1[:,1]);
##        print(confusion_matrix(Y_te, Y_p));
##        sv = svm.SVC(kernel='linear', random_state = 254348)
##        sv.fit(X_tr, Y_tr)
##        predicted_digits = sv.predict(X_te)
##        Y_p = predicted_digits
##        accuracy2 = accuracy2+accuracy_score(Y_te, Y_p);
##        print(accuracy_score(Y_te, Y_p));
##        precision2_0 = precision2_0 + precision_score(Y_te, Y_p, average='binary', pos_label=0);
##        precision2_1 = precision2_1 + precision_score(Y_te, Y_p, average='binary', pos_label=1);
##        recall2_0 = recall2_0 + recall_score(Y_te, Y_p, average='binary', pos_label=0);
##        recall2_1 = recall2_1 + recall_score(Y_te, Y_p, average='binary', pos_label=1);       
##        print(precision_score(Y_te, Y_p, average='binary', pos_label=1));
##        print(precision_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(recall_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(recall_score(Y_te, Y_p, average='binary', pos_label=1));
##        f1measure2_0 = f1measure2_0 + f1_score(Y_te, Y_p, average='binary', pos_label=0)
##        f1measure2_1 = f1measure2_1 + f1_score(Y_te, Y_p, average='binary', pos_label=1)
##        print(f1_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(f1_score(Y_te, Y_p, average='binary', pos_label=1));
##        print(confusion_matrix(Y_te, Y_p));
##        logreg = linear_model.LogisticRegression(C=1e5)
##        logreg.fit(X_tr, Y_tr)
##        predicted_digits = logreg.predict(X_te)
##        Y_p = predicted_digits
##        accuracy3= accuracy3+accuracy_score(Y_te, Y_p);
##        print(accuracy_score(Y_te, Y_p));
##        precision3_0 = precision3_0 + precision_score(Y_te, Y_p, average='binary', pos_label=0);
##        precision3_1 = precision3_1 + precision_score(Y_te, Y_p, average='binary', pos_label=1);
##        recall3_0 = recall3_0 + recall_score(Y_te, Y_p, average='binary', pos_label=0);
##        recall3_1 = recall3_1 + recall_score(Y_te, Y_p, average='binary', pos_label=1);       
##        print(precision_score(Y_te, Y_p, average='binary', pos_label=1));
##        print(precision_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(recall_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(recall_score(Y_te, Y_p, average='binary', pos_label=1));
##        f1measure3_0 = f1measure3_0 + f1_score(Y_te, Y_p, average='binary', pos_label=0)
##        f1measure3_1 = f1measure3_1 + f1_score(Y_te, Y_p, average='binary', pos_label=1)
##        print(f1_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(f1_score(Y_te, Y_p, average='binary', pos_label=1));
##        print(confusion_matrix(Y_te, Y_p));
##        neigh = KNeighborsClassifier(n_neighbors=3)
##        neigh.fit(X_tr, Y_tr)
##        predicted_digits = neigh.predict(X_te)
##        Y_p = predicted_digits
##        accuracy4 = accuracy4+accuracy_score(Y_te, Y_p);
##        print(accuracy_score(Y_te, Y_p));
##        precision4_0 = precision4_0 + precision_score(Y_te, Y_p, average='binary', pos_label=0);
##        precision4_1 = precision4_1 + precision_score(Y_te, Y_p, average='binary', pos_label=1);
##        recall4_0 = recall4_0 + recall_score(Y_te, Y_p, average='binary', pos_label=0);
##        recall4_1 = recall4_1 + recall_score(Y_te, Y_p, average='binary', pos_label=1);       
##        print(precision_score(Y_te, Y_p, average='binary', pos_label=1));
##        print(precision_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(recall_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(recall_score(Y_te, Y_p, average='binary', pos_label=1));
##        f1measure4_0 = f1measure4_0 + f1_score(Y_te, Y_p, average='binary', pos_label=0)
##        f1measure4_1 = f1measure4_1 + f1_score(Y_te, Y_p, average='binary', pos_label=1)
##        print(f1_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(f1_score(Y_te, Y_p, average='binary', pos_label=1));
##        print(confusion_matrix(Y_te, Y_p));
        
##    table = [
         ##        ["Random Forests","","","","","","","","","",""],
    print("Random Forests",str(roc_accuracy1/K))##precision1_0/K,precision1_1/K,recall1_0/K,recall1_1/K,f1measure1_0/K,f1measure1_1/K, kappa1/K, matthews1/K]]
    ##             ["Logistic regression","","","","","","","","","",""],
    ##             ["",accuracy3/K,roc_accuracy3/K,precision3_0/K,precision3_1/K,recall3_0/K,recall3_1/K,f1measure3_0/K,f1measure3_1/K, kappa3/K, matthews3/K],
    ##             ["k-nearest neighbor","","","","","","","","","",""],
    ##             ["",accuracy4/K,roc_accuracy4/K,precision4_0/K,precision4_1/K,recall4_0/K,recall4_1/K,f1measure4_0/K,f1measure4_1/K, kappa4/K, matthews4/K]]
##    print(tabulate(table, headers= ["Classifier","Accuracy","roc_auc_curve","precision_0","precision_1","recall_0","recall_1","f1 score_0","f1 score_1","kappa Score","Matthews score"]))
##    f = open('Results_e5_30.txt','w')
##    f.write(tabulate(table, headers= ["Classifier","Accuracy","roc_auc_curve","precision_0","precision_1","recall_0","recall_1","f1 score_0","f1 score_1","kappa Score","Matthews score"]))
##    f.close()






if __name__=="__main__":
    main()
