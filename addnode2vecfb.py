import pandas as pd

f = pd.read_csv('friendship-withoutdate.csv', delimiter=',', header=None);

nodesList=[];
nodes = {}
f.columns = ['user1','user2','createdate']
for i in range(0,len(f)):
               x = f.loc[i]['user1'];
               if x not in nodesList:                  
                   nodesList.append(x);
               y = f.loc[i]['user2'];
               if y not in nodesList:
                   nodesList.append(y);
               nodes[x] = str(nodesList.index(x)+1)
               nodes[y] = str(nodesList.index(y)+1)

f = pd.read_csv('user2006b.csv', delimiter=',');
for i in range(0,len(f)):
               x = f.loc[i]['user1'];
               if x not in nodesList:                  
                   nodesList.append(x);
               y = f.loc[i]['user2'];
               if y not in nodesList:
                   nodesList.append(y)
               nodes[x] = str(nodesList.index(x)+1)
               nodes[y] = str(nodesList.index(y)+1)






node2vec = {}
fr= open('facebook.emb','r')
ll = fr.readlines()[1:]
for l in ll:
    s = l.split()
    node2vec[s[0]] = []
    for i in range(1,len(s)):
        node2vec[s[0]].append(s[i])


fr.close()
fr = open('fb_dt_new.csv','r')
lines = fr.readlines()
header= lines[0].rstrip('\n')
f = open('fb_dt2.csv','w')
f.write(header)
for i in range(0,40):
    f.write('\t'+'nv'+str(i))

    
f.write('\n')
del lines[0]
for line in lines:
    line = line.rstrip('\n')
    s = line.split('\t');
    x = 0
    y = 0
    c = line
    if int(s[0]) in nodes:
        x = nodes[int(s[0])]
    if int(s[1]) in nodes:
        y = nodes[int(s[1])]
    if x in node2vec:
        for i in range(0,40):
            c = c+'\t'+str(node2vec[x][i])
    else:
        for i in range(0,40):
            c = c+'\t'+'0'
    if y in node2vec:
        for i in range(0,40):
            c = c+'\t'+str(node2vec[y][i])
    else:
        for i in range(0,40):
            c = c+'\t'+'0'
    f.write(c)
    f.write('\n')
     



f.close();
fr.close();

        
