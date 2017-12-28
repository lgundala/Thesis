import pandas as pd

df = pd.read_csv('2002.csv')
pos2002 = {}
for index, row in df.iterrows():
    u1 = row['user1']
    u2 = row['user2']
    t = str(u1)+';'+str(u2)
    k = str(u2)+';'+str(u1)
    pos2002[k] = 1
    pos2002[t] = 1


            
df = pd.read_csv('2003.csv')
pos2003 = {}
for index, row in df.iterrows():
    u1 = row['user1']
    u2 = row['user2']
    t = str(u1)+';'+str(u2)
    k = str(u2)+';'+str(u1)
    pos2003[k] = 1
    pos2003[t] = 1




df = pd.read_csv('2004.csv')
pos2004 = {}
for index, row in df.iterrows():
    u1 = row['user1']
    u2 = row['user2']
    t = str(u1)+';'+str(u2)
    k = str(u2)+';'+str(u1)
    pos2004[k] = 1
    pos2004[t] = 1




df = pd.read_csv('2005.csv')
pos2005 = {}
for index, row in df.iterrows():
    u1 = row['user1']
    u2 = row['user2']
    t = str(u1)+';'+str(u2)
    k = str(u2)+';'+str(u1)
    pos2005[k] = 1
    pos2005[t] = 1



df = pd.read_csv('2006.csv')
pos2006 = {}
for index, row in df.iterrows():
    u1 = row['user1']
    u2 = row['user2']
    t = str(u1)+';'+str(u2)
    k = str(u2)+';'+str(u1)
    pos2006[k] = 1
    pos2006[t] = 1



df = pd.read_csv('2007.csv')
pos2007 = {}
for index, row in df.iterrows():
    u1 = row['user1']
    u2 = row['user2']
    t = str(u1)+';'+str(u2)
    k = str(u2)+';'+str(u1)
    pos2007[k] = 1
    pos2007[t] = 1


df = pd.read_csv('2008.csv')
pos2008 = {}
for index, row in df.iterrows():
    u1 = row['user1']
    u2 = row['user2']
    t = str(u1)+';'+str(u2)
    k = str(u2)+';'+str(u1)
    pos2008[k] = 1
    pos2008[t] = 1




df = pd.read_csv('2009.csv')
pos2009 = {}
for index, row in df.iterrows():
    u1 = row['user1']
    u2 = row['user2']
    t = str(u1)+';'+str(u2)
    k = str(u2)+';'+str(u1)
    pos2009[k] = 1
    pos2009[t] = 1





df = pd.read_csv('2010.csv')
pos2010 = {}
for index, row in df.iterrows():
    u1 = row['user1']
    u2 = row['user2']
    t = str(u1)+';'+str(u2)
    k = str(u2)+';'+str(u1)
    pos2010[k] = 1
    pos2010[t] = 1




df = pd.read_csv('2011.csv')
pos2011 = {}
for index, row in df.iterrows():
    u1 = row['user1']
    u2 = row['user2']
    t = str(u1)+';'+str(u2)
    k = str(u2)+';'+str(u1)
    pos2011[k] = 1
    pos2011[t] = 1

df['time'] = 0
df['status'] = 1
for index, row in df.iterrows():
    u1 = row['user1']
    u2 = row['user2']
    k = str(u1)+';'+str(u2)
    d = str(u2)+';'+str(u1)
    c = 1
    s = 1
    if k in c1 or d in c1:
        c = c+1
        if k in a2 or d in a2:
            c = c+1
            if k in b2 or d in b2:
                c = c+1
                if k in c2 or d in c2:
                    c = c+1
                    s = 0                      
    df.at[index,'time'] = c
    df.at[index,'status'] = s




import random
pn = y[y['time']==1].copy()
n = len(y[y['time']>1])
npn = pn.ix[random.sample(pn.index,n)]
py = y[y['time']>1].copy()
frames = [py,npn]
re = pd.concat(frames)
re.to_csv('dt_ja.csv',sep='\t',index=None)



ps = y[y['time']>6].copy()

        
