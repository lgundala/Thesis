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
def main():
    roc_accuracy1 = 0;
    K = 10
    for i in range(0,10):
        print(i);
        x_te = pd.read_csv("test_M"+str(i)+".csv",delimiter=",", header=None, encoding="utf-8");
        x_tr = pd.read_csv("train_M"+str(i)+".csv",delimiter=",", header=None, encoding="utf-8");
        ####uncomment below region if you want to change size of test w.r.t train and change the lines to extract y
##        x_tr_1 = x_tr.loc[x_tr[8]==1]
##        x_tr_0 = x_tr.loc[x_tr[8]==0]
##        l = round((len(x_tr_1.index))*0.1)
##        rows = random.sample(list(x_tr_1),l )
##        x_tr_11 = x_tr_1.ix[rows];
##        frames = [x_tr_11, x_tr_0];
##        x_trf = pd.concat(frames);
####        x_te_1 = x_te.loc[x_te[8]==1]
####        x_te_0 = x_te.loc[x_te[8]==0]
####        l = round((len(x_te_1.index))*0.01)
####        rows = random.sample(list(x_te_1),l )
####        x_te_11 = x_te_1.ix[rows];
####        frames = [x_te_11, x_te_0];
####        x_tef = pd.concat(frames);
        y_te = x_te[2].values;
        del x_te[2]
        y_tr = x_tr[2].values;
        del x_tr[2];
        x = x_tr.values;
        X_tr=x[:,1:640];
        Y_tr = y_tr;
        x = x_te.values;
        X_te = x[:,1:640];
        Y_te = y_te;
        rf = RandomForestClassifier(n_estimators=100, criterion="entropy",random_state=0, class_weight = "balanced")
        rf.fit(X_tr, Y_tr)
        predicted_digits = rf.predict(X_te)
        Y_p1 = rf.predict_proba(X_te)
        Y_p = predicted_digits
##        accuracy1 = accuracy1+accuracy_score(Y_te, Y_p);
##	print("Random Forests")
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
##        print(f1_score(Y_te, Y_p, average='binary', pos_label=0));
##        print(f1_score(Y_te, Y_p, average='binary', pos_label=1));
	roc_accuracy1= roc_accuracy1 +met.roc_auc_score(Y_te,Y_p);
##        print(met.roc_auc_score(Y_te,Y_p));
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
##        logreg = linear_model.LogisticRegression(C=1e5, class_weight="balanced")
##        logreg.fit(X_tr, Y_tr)
 ##       predicted_digits = logreg.predict(X_te)
 ##       Y_p = predicted_digits
 ##       accuracy3= accuracy3+accuracy_score(Y_te, Y_p);
##	print("Logistic Regression")
  ##      print(accuracy_score(Y_te, Y_p));
   ##     precision3_0 = precision3_0 + precision_score(Y_te, Y_p, average='binary', pos_label=0);
   ##     precision3_1 = precision3_1 + precision_score(Y_te, Y_p, average='binary', pos_label=1);
    ##    recall3_0 = recall3_0 + recall_score(Y_te, Y_p, average='binary', pos_label=0);
    ##    recall3_1 = recall3_1 + recall_score(Y_te, Y_p, average='binary', pos_label=1);       
   ##     print(precision_score(Y_te, Y_p, average='binary', pos_label=1));
   ##     print(precision_score(Y_te, Y_p, average='binary', pos_label=0));
   ##     print(recall_score(Y_te, Y_p, average='binary', pos_label=0));
   ##     print(recall_score(Y_te, Y_p, average='binary', pos_label=1));
  ##      f1measure3_0 = f1measure3_0 + f1_score(Y_te, Y_p, average='binary', pos_label=0)
    ##    f1measure3_1 = f1measure3_1 + f1_score(Y_te, Y_p, average='binary', pos_label=1)
   ##     print(f1_score(Y_te, Y_p, average='binary', pos_label=0));
  ## ##     print(f1_score(Y_te, Y_p, average='binary', pos_label=1));
   ##     print(confusion_matrix(Y_te, Y_p));
   ##     neigh = KNeighborsClassifier(n_neighbors=3)
   ##     neigh.fit(X_tr, Y_tr)
    ##    predicted_digits = neigh.predict(X_te)
    ##    Y_p = predicted_digits
    ##    accuracy4 = accuracy4+accuracy_score(Y_te, Y_p);
	##print("K-nearest neighbor")
   ##     print(accuracy_score(Y_te, Y_p));
     ##   precision4_0 = precision4_0 + precision_score(Y_te, Y_p, average='binary', pos_label=0);
     ##   precision4_1 = precision4_1 + precision_score(Y_te, Y_p, average='binary', pos_label=1);
     ##   recall4_0 = recall4_0 + recall_score(Y_te, Y_p, average='binary', pos_label=0);
     ##   recall4_1 = recall4_1 + recall_score(Y_te, Y_p, average='binary', pos_label=1);       
     ##   print(precision_score(Y_te, Y_p, average='binary', pos_label=1));
      ##  print(precision_score(Y_te, Y_p, average='binary', pos_label=0));
      ##  print(recall_score(Y_te, Y_p, average='binary', pos_label=0));
      ##  print(recall_score(Y_te, Y_p, average='binary', pos_label=1));
     ##   f1measure4_0 = f1measure4_0 + f1_score(Y_te, Y_p, average='binary', pos_label=0)
     ##   f1measure4_1 = f1measure4_1 + f1_score(Y_te, Y_p, average='binary', pos_label=1)
       ## print(f1_score(Y_te, Y_p, average='binary', pos_label=0));
     ##   print(f1_score(Y_te, Y_p, average='binary', pos_label=1));
      ##  print(confusion_matrix(Y_te, Y_p));
   

    print("Random Forests",str(roc_accuracy1/K))##,str(precision1_0/K),str(precision1_1/K),str(recall1_0/K),str(recall1_1/K),str(f1measure1_0/K),str(f1measure1_1/K))
 ##   print("Logistic Regression",str(accuracy3/K),str(roc_accuracy3/K),str(precision3_0/K),str(precision3_1/K),str(recall3_0/K),str(recall3_1/K),str(f1measure3_0/K),str(f1measure3_1/K))
 ##   print("K-newarest neighbor",str(accuracy4/K),str(roc_accuracy4/K),str(precision4_0/K),str(precision4_1/K),str(recall4_0/K),str(recall4_1/K),str(f1measure4_0/K),str(f1measure4_1/K))
##
if __name__=="__main__":
    main()
