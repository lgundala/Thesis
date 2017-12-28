import pandas as pd
import networkx as nx
import json
from networkx.readwrite import json_graph
import math
p = pd.read_csv('upto2006-u1.csv',delimiter='\t')




dg = nx.Graph();
for index,row in p.iterrows():
	user1 = str(row['user1']);
	user2 = str(row['user2']);
	dg.add_node(user2);
	dg.add_node(user1);
	dg.add_edge(user1,user2);

p = pd.read_csv('fb_dt_new.csv',delimiter='\t')
p['common_neighbors'] = '0'
for index,row in p.iterrows():
	user1 = str(row['user1']);
	user2 = str(row['user2']);
	try:
		p.at[index,'common_neighbors'] = len(sorted(nx.common_neighbors(dg,user1,user2)))
	except:
		p.at[index,'common_neighbors'] = '0'
	
	


p.to_csv('2006graphCN.csv',sep='\t',index=None);




	
	
