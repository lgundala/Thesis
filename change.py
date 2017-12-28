import pandas as pd
for i in range(0,10):
    for j in range(0,5):
            f = pd.read_csv('wiki'+str(i)+'test'+str(j)+'.csv',delimiter='\t')
            fr = pd.read_csv('wiki'+str(i)+'train'+str(j)+'.csv',delimiter='\t')
            for index,row in f.iterrows():
                    st = row['status']
                    if st==0:
                            f.at[index,'status_class'] =1
            for index,row in fr.iterrows():
                    st = row['status']
                    if st==0:
                            fr.at[index,'status_class'] =1
            f.to_csv('wiki'+str(i)+'test'+str(j)+'.csv',sep='\t',index=None)
            fr.to_csv('wiki'+str(i)+'train'+str(j)+'.csv',sep='\t',index=None)
