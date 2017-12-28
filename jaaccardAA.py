import pandas as pd
import networkx as nx
import json
import math
from networkx.readwrite import json_graph
p = pd.read_csv('2006graph.csv',sep='\t')
p['jaccard']='0'
p['adamicadar']='0'


dg = nx.Graph();
for index,row in p.iterrows():
    user1 = row['user1']
    user2 = row['user2'];
    dg.add_node(user2);
    dg.add_node(user1);
    dg.add_edge(user1,user2);

jc = {}
aa = {}
for index,row in p.iterrows():
    user1 = row['user1']
    user2 = row['user2'];
    u1 = dg.neighbors(user1)
    u2 = dg.neighbors(user2)
    inter =  [val for val in u1 if val in u2]
    un = list(set().union(u1,u2))
    jc = len(inter)/len(un)
    aa = 0
    for n in inter:
        d = math.log(1/dg.degree(n))
        aa = aa+d
    p.at[index,'jaccard'] = jc
    p.at[index,'adamicadar']=aa
    
	
p.to_csv('2006cja.csv',sep='\t',index=None);




	
	
