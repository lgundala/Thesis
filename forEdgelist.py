import csv
import pandas as pd


f = pd.read_csv('/Users/Amulya/Downloads/Dataset_test.csv', delimiter=',', encoding='utf-8');
nodesList=[];
del f['Unnamed: 0']
for i in range(0,len(f)):
               x = f.loc[i]['prev'];
               if x not in nodesList:                  
                   nodesList.append(x);
for i in range(0,len(f)):
               x = f.loc[i]['curr'];
               if x not in nodesList:
                   nodesList.append(x);

filenew = open('/Users/Amulya/Documents/wikipedia.edgelist','w+');
for i in range(0,len(f)):
                x = f.loc[i]['prev'];
                y = f.loc[i]['curr'];
                line = str(nodesList.index(x)+1)+' '+str(nodesList.index(y))
                print(line);
                filenew.write(str(line)+"\n");
                
filenew.close()
               
               
    
