import pandas as pd
import itertools
user = {}
user2 = {}
wall = {}
wall2 = {}
totaluser = []

f = open('users.csv','r')
lines = f.readlines()[1:]
for line in lines:
    line = line.rstrip('\n')
    s = line.split(',')
    if s[0] not in user:
        user[s[0]] = {}
        user[s[0]][s[1]] = s[2]
    else:
        if s[1] in user[s[0]]:
            s2 = user[s[0]][s[1]]
            user[s[0]][s[1]] = s2+','+s[2]
        else:
            user[s[0]][s[1]] = s[2]          
    if s[1] not in user2:
        user2[s[1]] = {}
        user2[s[1]][s[0]] = s[2]
    else:
        if s[0] in user2[s[1]]:
            s2 = user2[s[1]][s[0]]
            user2[s[1]][s[0]] = s2+','+s[2]
        else:
            user2[s[1]][s[0]] = s[2] 
    totaluser.append(s[0])
    totaluser.append(s[1])


fr = open('wall.csv','r')
lines = fr.readlines()[1:]
for line in lines:
    line = line.rstrip('\n')
    s = line.split(',')
    if s[0] not in wall:
        wall[s[0]] = {}
        wall[s[0]][s[1]] = s[2]
    else:
        if s[1] in wall[s[0]]:
            s2 = wall[s[0]][s[1]]
            wall[s[0]][s[1]] = s2+','+s[2]
        else:
            wall[s[0]][s[1]] = s[2]          
    if s[1] not in wall2:
        wall2[s[1]] = {}
        wall2[s[1]][s[0]] = s[2]
    else:
        if s[0] in wall2[s[1]]:
            s2 = wall2[s[1]][s[0]]
            wall2[s[1]][s[0]] = s2+','+s[2]
        else:
            wall2[s[1]][s[0]] = s[2] 



f.close()
fr.close()


indirect = {}



def getIndirect(u1, u2):
    if u1 not in user:
            if u1 not in user2:
                if u1 not in indirect:
                    indirect[u1] = {}
                    indirect[u1][u2] = 0
                else:
                    if u2 not in indirect[u1]:
                        indirect[u1][u2] = 0
                    else:
                        w = indirect[u1][u2]
                        indirect[u1][u2] = w+1
            elif u2 not in user2:
                if u1 not in indirect:
                    indirect[u1] = {}
                    indirect[u1][u2] = 0
                else:
                    if u2 not in indirect[u1]:
                        indirect[u1][u2] = 0
                    else:
                        w = indirect[u1][u2]
                        indirect[u1][u2] = w+1




for u, v in wall.iteritems():
    u1 = u
    for v,c in wall[u].iteritems():
        u2 = v
        getIndirect(u1,u2);


                        
for u, v in wall2.iteritems():
    u1 = u
    for v,c in wall2[u].iteritems():
        u2 = v
        getIndirect(u1,u2);



for u,v in wall.iteritems():
    walllist = []
    for v,c in wall[u].iteritems():
        walllist.append(v)
    for subset in itertools.combinations(walllist,2):
        getIndirect(subset[0],subset[1])
        getIndirect(subset[1],subset[0])


Bindirect = {}
f = open('indirectfriends1.csv','w')
for u , v in indirect.iteritems():
    for v, c in indirect[u].iteritems():       
        if c > 0:
            if u not in Bindirect:
                Bindirect[u] = {}
                Bindirect[u][v] = c
                f.write(u+'\t'+v+'\t'+str(c)+'\n')
            else:
                if v in Bindirect[u].iteritems():
                    w = Bindirect[u][v]
                    Bindirect[u][v] = w+c
                else:
                    Bindirect[u][v] = c
                f.write(u+'\t'+v+'\t'+str(Bindirect[u][v])+'\n')

f.close()

            
                
            
        
        
