f = open('fb6b8b.csv','r')
lines = f.readlines()[2:]
pos = {}
for line in lines:
     line = line.rstrip('\r\n')
     s = line.split()
     t = s[0]+';'+s[1]
     pos[t] = s[2]
            


f.close()


f = open('fb2006b.csv','r')
lines = f.readlines()[2:]
pos6 = {}
for line in lines:
     line = line.rstrip('\r\n')
     s = line.split()
     t = s[0]+';'+s[1]
     pos6[t] = s[2]
            


f.close()




f = open('fb2007a.csv','r')
lines = f.readlines()[2:]
pos7a = {}
for line in lines:
     line = line.rstrip('\r\n')
     s = line.split()
     t = s[0]+';'+s[1]
     pos7a[t] = s[2]
            


f.close()

f = open('fb2007b.csv','r')
lines = f.readlines()[2:]
pos7b = {}
for line in lines:
     line = line.rstrip('\r\n')
     s = line.split()
     t = s[0]+';'+s[1]
     pos7b[t] = s[2]
            


f.close()


f = open('fb2008a.csv','r')
lines = f.readlines()[2:]
pos8a = {}
for line in lines:
     line = line.rstrip('\r\n')
     s = line.split()
     it = s[0]+';'+s[1]
     pos8a[t] = s[2]
            


f.close()


f = open('fb2008b.csv','r')
lines = f.readlines()[2:]
pos8b = {}
for line in lines:
     line = line.rstrip('\r\n')
     s = line.split()
     t = s[0]+';'+s[1]
     pos8b[t] = s[2]
            


f.close()






fr = open('fb2006b.csv','r')
fc = open('fb_dt_new.csv','w')
fc.write('user1'+'\t'+'user2'+'\t'+'hits'+'\t'+'time'+'\t'+'status'+'\t'+'class'+'\n')
st = 1 ##status
c = 1 ##time
cl = 0 ##class
lines = fr.readlines()[1:]
i = 0
j = 0
k = 0
for line in lines:
    line = line.rstrip('\n')
    s = line.split('\t')
    t = s[0]+';'+s[1]
    j = j+1;
    if s[3] == 'n':
        i = i+1
        if t in pos8b:
             c = 5
             st = 0
             cl = 1
        elif t in pos8a:
             c = 4
             st = 1
             k = k+1;
             print(k)
        elif t in pos7b:
             c = 3
             st = 1
        elif t in pos7a:
             c = 2
             st = 1
        fc.write(str(s[0])+'\t'+str(s[1])+'\t'+str(s[2])+'\t'+str(c)+'\t'+str(st)+'\t'+str(cl)+'\n')
            
print(i)
print(j)

fr.close()
fc.close()
            
        
