import sqlite3 as lite
import sys

con = lite.connect('facebook.db')
f = open('friendship-withoutdate.csv','r')
lines = f.readlines()
with con:
    cur = con.cursor()
    cur.execute("create table user_woDate
    for line in lines:
        line = line.rstrip('\n')
        s = line.split(',')
        cur.execute("INSERT INTO user_woDate VALUES("+s[0]+","+s[1]+")");
f.close()



