from __future__ import division
import numpy as np
pqueuelist = {}
import math
from collections import OrderedDict
feblinks_directed = {}
febtotal = {}

def createGraphs():
    with open('/Users/Amulya/Documents/FebruaryLinkswithTabs.csv','r') as f:
        lines = f.readlines()[1:];
        for line in lines:
            line = line.strip('\n');
            s = line.split('\t');
            if s[0] in feblinks_directed:
                if not s[1] in feblinks_directed[s[0]]:
                    feblinks_directed[s[0]][s[1]] = s[3];
            else:
                feblinks_directed[s[0]] = {}
                feblinks_directed[s[0]][s[1]] = s[3];
                
            if s[1] in febtotal:
                if not s[0] in febtotal[s[1]]:
                    febtotal[s[1]][s[0]] = s[3];
            else:
                febtotal[s[0]] = {}
                febtotal[s[0]][s[1]] = s[3];
                
            if s[1] in febtotal:
                if not s[0] in febtotal[s[1]]:
                    febtotal[s[1]][s[0]] = s[3];
            else:
                febtotal[s[1]] = {}
                febtotal[s[1]][s[0]] = s[3];
                
    with open('/Users/Amulya/Documents/Febother.csv','r') as f:
        lines = f.readlines()[1:];
        for line in lines:
            line = line.strip('\n');
            s = line.split('\t');
            if s[0] in febtotal:
                if not s[1] in febtotal[s[0]]:
                    febtotal[s[0]][s[1]] = s[3];
            else:
                febtotal[s[0]] = {}
                febtotal[s[0]][s[1]] = s[3];
                
            if s[1] in febtotal:
                if not s[0] in febtotal[s[1]]:
                    febtotal[s[1]][s[0]] = s[3];
            else:
                febtotal[s[1]] = {}
                febtotal[s[1]][s[0]] = s[3];
                
                
##            if s[0] in febother:
##                if not s[1] in febother[s[0]]:
##                    febother[s[0]][s[1]] = s[3];
##            else:
##                febother[s[0]] = {}
##                febother[s[0]][s[1]] = s[3];
##    return febother
            
def getDict(test):
    febother = {}
    s = []
    for i in range(0,len(test)):
        s[0] = test[i][0]
        s[1] = test[i][1]
        s[2] = test[i][2]
        if s[0] in febother:
            if not s[1] in febother[s[0]]:
                febother[s[0]][s[1]] = s[2];
        else:
            febother[s[0]] = {}
            febother[s[0]][s[1]] = s[2];
    return febother

from Queue import PriorityQueue

class MyPriorityQueue(PriorityQueue):
    def __init__(self):
        PriorityQueue.__init__(self)
        self.counter = 0
    def put(self, item, priority):
        PriorityQueue.put(self, (priority, self.counter, item))
        self.counter += 1
    def get(self, *args, **kwargs):
        _, _, item = PriorityQueue.get(self, *args, **kwargs)
        return item;
 
def getsigmaE(prev):
    sigmaE = 0;
    if prev in feblinks_directed:
        for curr in feblinks_directed[prev]:
            sigmaE = sigmaE + int(feblinks_directed[prev][curr]);
    return sigmaE

def predictClass(pqueue,febother,K):
    mg = 0
    pp = febother
    for page in pqueuelist.iteritems():
        mg = mg+ float(pqueuelist[page])*(-1);
    for i in range(0,K):
        page = pqueue.get()
        marginGainp = float(pqueuelist[page])*(-1)/mg
        prev, curr = page.split('|');
        febother[prev][curr] = 1
        pp[prev][curr] = marginGainp
    for i in range(k,len(pqueuelist)):
        page = pqueue.get()
        marginGainp = float(pqueuelist[page])*(-1)/mg
        prev, curr = page.split('|');
        febother[prev][curr] = 0
        pp[prev][curr] = 1- marginGainp
    return np.array(febother.values()), np.array(pp.values())


def Rw_algorithm(f,testset,K):
    pqueue = MyPriorityQueue()
    febother = testset
    for prev in febother:
        sigma = 0;
        sigmaE = getsigmaE(prev);
##	print(sigmaE);
        c = 0;
        pi = 1;
        pst = {}
        ws = {}
        for curr in febother[prev]:
            nst = int(febother[prev][curr])
            n = 0
            for curr in febtotal[prev]:
                n = n + int(febtotal[prev][curr]);
            ns = n;
            pst[curr] = nst/ns;
            ws[curr] = math.log(ns);
            pst = OrderedDict(sorted(pst.items(),key = lambda kv:kv[1], reverse = True))       
        for curr,value in pst.iteritems():          
            sigma = sigma + value 
            pi = pi*(1-value);
            if f == 'f1':
                c_cap = ws[curr]*sigma;
            if f == 'f2':
                c_cap = ws[curr]*(1-pi);
            if f == 'f3':
		try:
                	c_cap = ws[curr]*sigma/(sigma+sigmaE);
		except:
			continue
            pair = str(prev)+'|'+str(curr);
            c_star = c-c_cap;
            pqueue.put(pair,c_star);
            pqueuelist[pair] = c_star;
            c = c_cap;
    predictClass(pqueue,febother,K)

        
            

            


