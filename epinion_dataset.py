import pandas as pd
import networkx as nx
import json
from networkx.readwrite import json_graph
import math
p = pd.read_csv('/Users/Amulya/Downloads/Epinions/dataset/trust2001.csv',delimiter='\t')
dg = nx.Graph();
for index,row in p.iterrows():
	user1 = str(row['user1']);
	user2 = str(row['user2']);
	dg.add_node(user2);
	dg.add_node(user1);
	dg.add_edge(user1,user2);

p = pd.read_csv('/Users/Amulya/Downloads/Epinions/dataset/2001.csv',delimiter=',')
p['common_neighbors'] = '0'
p['user1degree'] = '0'
p['user2degree'] = '0'
for index,row in p.iterrows():
	user1 = str(row['user1']);
	user2 = str(row['user2']);
	try:
		p.at[index,'common_neighbors'] = len(sorted(nx.common_neighbors(dg,user1,user2)))
	except:
		p.at[index,'common_neighbors'] = '0'
	try:
                p.at[index,'user1degree'] = len(dg.degree(user1))
        except:
                p.at[index,'user1degree'] = dg.degree(user1)
        try:
                p.at[index,'user2degree'] = len(dg.degree(user2))
        except:
                p.at[index,'user2degree'] = dg.degree(user2)
	
	


p.to_csv('/Users/Amulya/Downloads/Epinions/dataset/2001graph.csv',sep='\t',index=None);




	
	
