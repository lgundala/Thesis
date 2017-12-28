

def datasets(y):
    f = open('no'+str(y)+'.csv','r')
    w = open('fb'+str(y)+'.csv','w')
    w.write('user1'+'\t'+'user2'+'\t'+'hits'+'\t'+'class'+'\n')
    lines = f.readlines()
    for line in lines:
        line = line.rstrip('\n')
        w.write(line+'\t'+'n'+'\n')
    f.close()
    f = open('yes'+str(y)+'.csv','r')
    lines = f.readlines()
    for line in lines:
        line = line.rstrip('\n')
        w.write(line+'\t'+'y'+'\n')
    f.close()



datasets(2006)
datasets(2007)
datasets(2008)
datasets(2009)
