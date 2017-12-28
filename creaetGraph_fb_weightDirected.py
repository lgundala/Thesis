import pandas as pd
import networkx as nx
import json
from networkx.readwrite import json_graph

fl = pd.read_csv('fb2006.csv',delimiter='\t');
p = fl[fl['class']=='y']
del p['class']
p['user1_pagerank']=0
p['user1_successors'] = 0
p['user1_predecessors']=0
p['user1_indegree']=0
p['user1_outdegree']=0


p['user2_pagerank']=0
p['user2_successors'] = 0
p['user2_predecessors']=0
p['user2_indegree']=0
p['user2_outdegree']=0

dg = nx.DiGraph();
for index,row in p.iterrows():
	user1 = str(row['user1']);
	user2 = str(row['user2']);
	n = row['hits']
	dg.add_node(user1);
	dg.add_node(user2);
	dg.add_edge(user1,user2, weight = n);

pageranks = nx.pagerank(dg);
for index,row in p.iterrows():
	user1 = str(row['user1']);
	user2 = str(row['user2']);
	p.at[index,'user1_successors'] = len(dg.successors(user1))
	p.at[index,'user1_predecessors'] = len(dg.predecessors(user1))
	p.at[index,'user1_indegree'] = dg.in_degree(user1)
	p.at[index,'user1_outdegree'] = dg.out_degree(user1)
	p.at[index,'user1_pagerank']=pageranks[user1];
	p.at[index,'user2_successors'] = len(dg.successors(user2))
	p.at[index,'user2_predecessors'] = len(dg.predecessors(user2))
	p.at[index,'user2_indegree'] = dg.in_degree(user2)
	p.at[index,'user2_outdegree'] = dg.out_degree(user2)
	p.at[index,'user2_pagerank']=pageranks[user2];


	

p.to_csv('2006Digraph.csv',sep='\t',index=None);




	
	
