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
    accuracy1 = 0;
    roc_accuracy1 = 0;
    precision1_0 = 0;
    precision1_1 = 0;
    recall1_0 = 0;
    recall1_1 = 0;
    f1measure1_0 = 0;
    f1measure1_1 = 0;
    K = 10
    for i in range(0,10):
        x_te = pd.read_csv("test_"+str(i),delimiter=",", header=None, encoding="utf-8");
        x_tr = pd.read_csv("train_"+str(i),delimiter=",", header=None, encoding="utf-8");
        y_te = x_te[9].values;
        y_tr = x_tr[9].values;
        del x_tr[9];
        del x_te[9]
        x = x_tr[[1,2,7]].values;
        X_tr=x[:,0:10];
        Y_tr = y_tr;
        x = x_te[[1,2,7]].values;
        X_te = x[:,0:10];
        Y_te = y_te;
        rf = RandomForestClassifier(n_estimators=100, criterion="entropy",random_state=0, class_weight="balanced")
        rf.fit(X_tr, Y_tr)
        predicted_digits = rf.predict(X_te)
        Y_p = predicted_digits
        accuracy1 = accuracy1+accuracy_score(Y_te, Y_p);
        precision1_0 = precision1_0 + precision_score(Y_te, Y_p, average='binary', pos_label=0);
        precision1_1 = precision1_1 + precision_score(Y_te, Y_p, average='binary', pos_label=1);
        recall1_0 = recall1_0 + recall_score(Y_te, Y_p, average='binary', pos_label=0);
        recall1_1 = recall1_1 + recall_score(Y_te, Y_p, average='binary', pos_label=1);
        f1measure1_0 = f1measure1_0 + f1_score(Y_te, Y_p, average='binary', pos_label=0)
        f1measure1_1 = f1measure1_1 + f1_score(Y_te, Y_p, average='binary', pos_label=1)
        roc_accuracy1 = roc_accuracy1 + met.roc_auc_score(Y_te, Y_p);
    print("Random Forests"+'\t'+str(roc_accuracy1/K)+'\t'+str(accuracy1/K)+'\t'+str(precision1_0/K)+'\t'+str(precision1_1/K)+'\t'+str(f1measure1_0/K)+'\t'+str(f1measure1_1/K)+'\t'+str(recall1_0/K)+'\t'+str(recall1_1/K))




if __name__=="__main__":
    main()
