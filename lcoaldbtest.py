import sqlite3 as lite
import sys

con = None
f = open('fb2009.csv','r')
lines = f.readlines()[1:]

try:
    con = lite.connect('facebook.db')
    cur = con.cursor()
    cur.execute("CREATE TABLE fb2009(user1 INT, user2 INT, count INT, class TEXT)")
    for l in lines:
        l = l.rstrip('\n')
        s = l.split('\t')
        cur.execute("INSERT INTO fb2009 VALUES("+s[0]+","+s[1]+","+s[2]+","+"'"+s[3]+"'"+")")               
    con.commit()
except lite.Error, e:
    
    print "Error %s:" % e.args[0]
    
 
    
finally:
    f.close()
    if con:
        con.close()
