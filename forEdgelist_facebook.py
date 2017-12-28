import pandas as pd

f = pd.read_csv('friendship-withoutdate.csv', delimiter=',', header=None);

nodesList=[];
f.columns = ['user1','user2','createdate']
filenew = open('facebook2006-link.edgelist','w');
for i in range(0,len(f)):
    x = f.loc[i]['user1'];
    if x not in nodesList:
        nodesList.append(x);
    y = f.loc[i]['user2'];
    if y not in nodesList:
        nodesList.append(y);
    line = str(nodesList.index(x)+1)+' '+str(nodesList.index(y)+1)
    filenew.write(str(line)+"\n");

f = pd.read_csv('user2006b.csv', delimiter=',');
for i in range(0,len(f)):
    x = f.loc[i]['user1'];
    if x not in nodesList:
        nodesList.append(x);
    y = f.loc[i]['user2'];
    if y not in nodesList:
        nodesList.append(y);
    line = str(nodesList.index(x)+1)+' '+str(nodesList.index(y)+1)
    filenew.write(str(line)+"\n");


filenew.close()

                

               
               
    
