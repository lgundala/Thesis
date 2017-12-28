import pandas as pd
nodes = {}
node2vec = {}
df = pd.read_csv('trust2001.csv',delimiter='\t')
i = 1
for index, row in p.iterrows():
     u1 = row['user1']
     u2 = row['user2']
     if u1 not in nodes:
             nodes[u1] = i
             i = i+1
     if u2 not in nodes:
             nodes[u2] = i
             i = i+1


df2 = pd.read_csv('epinion.emb',delimiter=' ', skiprows = 1, header=None)
for indes, row in df2.iterrows():
    id = int(row[0])
    node2vec[id] = []
    for i in range(1,41):
        node2vec[id].append(row[i])


df3 = pd.read_csv('dt_epinion_1024.csv',delimiter='\t')
for i in range(0,40):
    p['nv'+str(i)] = 0.0

p['nvcos'] = 0.0
for index, row in p.iterrows():
     u1 = row['user1']
     u2 = row['user2']
     idu1 = 0
     idu2 = 0
     if u1 in nodes:
             idu1 = int(nodes[u1])
     if u2 in nodes:
             idu2 = int(nodes[u2])
     if idu1 in node2vec and idu2 in node2vec:
             p.at[index,'nvcos'] = cosine_similarity(node2vec[idu1],node2vec[idu2])[0][0]
             for i in range(0,40):
                     p.at[index,'nv'+str(i)] = node2vec[idu1][i]*node2vec[idu2][i]



from sklearn.metrics.pairwise import cosine_similarity
for index, row in df3.iterrows():
     u1 = row['user1']
     u2 = row['user2']
     idu1 = 0
     idu2 = 0
     if u1 in nodes:
             idu1 = int(nodes[u1])
     if u2 in nodes:
             idu2 = int(nodes[u2])
     if idu1 in node2vec and idu2 in node2vec:
             c = cosine_similarity(node2vec[idu1],node2vec[idu2])[0][0]
             df3.at[index,'nvcos'] = c

