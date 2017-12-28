import html2text
import re
from os import walk
import nltk
from stop-words import get_stop_words
from nltk.stem.snowball import SnowballStemmer
import sys


stemmer = SnowballStemmer('english');
stop_words = get_stop_words('en')
res = ['<ref>.*?</ref>','<ref .*?</ref>','<ref .*?/>',
	   '==[^=]*==','\\[\\[([^]]*[|])?([^]]*?)\\]\\]','<!--.*?-->',
	   '[^A-Za-z0-9., ]','\\{\\{[^{}]*\\}\\}'];
for i in range(0,len(res)):
	res[i] = re.compile(res[i], flags=re.IGNORECASE);

for dirs in walk('.'):
        for files in dirs:
                for filenames in files:
                        print(filesnames);
                        f = open(filenames,'r',encoding='utf8',errors='ignore')
                        lines = f.readlines();
                        categories = [];
                        cat = '[[Category:';
                        ctlength = len(cat);
                        catFile = open('/home/lgundala/Documents/Categories/'+filenames,'w+');
                        textFile = open('/home/lgundala/Documents/PlainTexts/'+filenames,'w+');
                        for line in lines:
                                    s = line;
                                    if s!='\n' and (s.find('{{')!=0) and (s.find('|')!=0) and (s.find('*')!=0) and s!='\t\t' and s!='\n\n'and s!=' ':
                                            if cat in s:
                                                    startLine = s.find('[[');
                                                    sLength = len(s);
                                                    if startLine >= 0:
                                                            d = s.find(']]');
                                                            s = s[startLine+ctlength:d];
                                                            categories.append(s);
                                                            catFile.write(s);
                                                            catFile.write('\n');
                                            elif cat not in s:
                                                    for i in range(0,len(res)):
                                                            s = re.sub(res[i],' ',s);
                                                    s = s.replace('\t\t','');
                                                    s = s.strip();
                                                    s = ' '.join([stemmer.stem(word) for word in s.split() if word not in stop_words])
                                                    if s!='' and s!=' ':
                                                            textFile.write(s);
                                                            textFile.write('\n');
	textFile.close();
	catFile.close();
	f.close();







