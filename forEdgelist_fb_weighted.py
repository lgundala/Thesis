import pandas as pd

filenew = open('/home/laxmi/Documents/facebook2006-link_w.edgelist','w');
nodesList = []

f = pd.read_csv('fb2006.csv', delimiter='\t');
f = f[f['class']=='n']
for i in range(0,len(f)):
               x = f.loc[i]['user1'];
               if x not in nodesList:                  
                   nodesList.append(x);
               y = f.loc[i]['user2'];
               if y not in nodesList:
                   nodesList.append(y);
	       c = f.loc[i]['hits']
	       line = str(nodesList.index(x)+1)+' '+str(nodesList.index(y)+1)+' '+str(c)
               filenew.write(str(line)+"\n");


filenew.close()

                

               
               
    
