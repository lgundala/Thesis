c1 = 0.0
c2 = 0.0
c3 = 0.0
c4 = 0.0
c5 = 0.0
c6 = 0.0

for i in range(0,10):
    for j in range(0,5):
        f = open('ep'+str(i)+'cindex'+str(j)+'.csv','r')
        lines = f.readlines()[1:]
        c1 = c1+float(lines[0])
        c2 = c2+float(lines[1])
        c3 = c3+float(lines[2])
        c4 = c4+float(lines[3])
        c5 = c5+float(lines[4])
        c6 = c6+float(lines[5])
        f.close()



f=open('cindex.csv','w')
f.write('cindex 1: '+str(c1/50)+'\n')
f.write('cindex 2: '+str(c2/50)+'\n')
f.write('cindex 3: '+str(c3/50)+'\n')   
f.write('cindex 4: '+str(c4/50)+'\n')
f.write('cindex 5: '+str(c5/50)+'\n')
f.write('cindex 6: '+str(c6/50)+'\n')
