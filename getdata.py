import pandas as pd
import itertools
user = {}
user2 = {}
wall = {}
wall2 = {}
totaluser = []

def getdatay(y):
    f = open(str(y)+'-u.csv','r')
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
    f.close()


getdatay(2006)
getdatay(2007)
getdatay(2008)
getdatay(2009) 

def getnodatedata(y):
    f = open('friendship-withoutdate.csv','r')
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
    f.close()
getnodatedata(200)

fr = open('2009.csv','r')
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


fr.close()

direct = {}
def getdirect(u1, u2):
    if u1 in user:
        if u2 in user[u1]:
            if u1 not in direct:
                direct[u1] = {}
                direct[u1][u2] = 0
            else:
                if u2 not in direct[u1]:
                    direct[u1][u2] = 0
                else:
                    w = direct[u1][u2]
                    direct[u1][u2] = w+1
    if u1 in user2:
        if u2 in user2[u1]:
            if u2 not in direct:
                direct[u2] = {}
                direct[u2][u1] = 0
            else:
                if u1 not in direct[u2]:
                    direct[u2][u1] = 0
                else:
                    w = direct[u2][u1]
                    direct[u2][u1] = w+1



##for u, v in wall.iteritems():
##    u1 = u
##    for v,c in wall[u].iteritems():
##        u2 = v
##        getdirect(u1,u2);
##
##
##                        
##for u, v in wall2.iteritems():
##    u1 = u
##    for v,c in wall2[u].iteritems():
##        u2 = v
##        getdirect(u1,u2);

        
for u,v in wall.iteritems():
    walllist = []
    for v,c in wall[u].iteritems():
        walllist.append(v)
    for subset in itertools.combinations(walllist,2):
        getdirect(subset[0],subset[1])

for u,v in wall2.iteritems():
    walllist = []
    for v,c in wall2[u].iteritems():
        walllist.append(v)
    for subset in itertools.combinations(walllist,2):
        getdirect(subset[0],subset[1])

f = open('yes2009.csv','w')
for u ,v in direct.iteritems():
    for v,c in direct[u].iteritems():
        f.write(u+'\t'+v+'\t'+str(direct[u][v])+'\n')

f.close()
        
