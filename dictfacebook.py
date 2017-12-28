import pandas as pd


f = open('/Users/Amulya/Documents/Facebook-interaction-data/wallgt2.csv', 'r');
lines = f.readlines()
del lines[0]
d = {}
for line in lines:
    line = line.rstrip('\n')
    s = line.split(',')
    if s[1] in d:
        d[s[1]][s[0]] = s[2]
    elif s[0] in d:
        d[s[0]][s[1]] = s[2]
    elif s[1] not in d:
        d[s[1]] = {}
        d[s[1]][s[0]] = s[2]
    elif s[0] not in d:
        d[s[0]] = {}
        d[s[0]][s[1]] = s[2]
    


    
