import pandas as pd

fb2006 = pd.read_csv('fb2006.csv',delimiter='\t')
fb2007 = pd.read_csv('fb2007.csv',delimiter='\t')
fb2008 = pd.read_csv('fb2008.csv',delimiter='\t')
fb2009 = pd.read_csv('fb2009.csv',delimiter='\t')



n6 = fb2006[fb2006['class']=='n']
y6 = fb2006[fb2006['class']=='y']
n7 = fb2007[fb2007['class']=='n']
y7 = fb2007[fb2007['class']=='y']
n8 = fb2008[fb2008['class']=='n']
y8 = fb2008[fb2008['class']=='y']
n9 = fb2009[fb2009['class']=='n']
y9 = fb2009[fb2009['class']=='y']





frames=[n6,n7]
pm = pd.concat(frames, ignore_index = True)
dt = pm[pm.duplicated(['user1', 'user2'], keep='last') | pm.duplicated(['user1','user2'])]
fmtotal = dt[dt.duplicated(['user1', 'user2'])]
print 'facebook n2006 and n2007: ',len(fmtotal.index);
fbn20067 = fmtotal.copy()
fmtotal.to_csv('fbn20067.csv', sep='\t', index=None);

frames=[n7,n8]
pm = pd.concat(frames, ignore_index = True)
dt = pm[pm.duplicated(['user1', 'user2'], keep='last') | pm.duplicated(['user1','user2'])]
fmtotal = dt[dt.duplicated(['user1', 'user2'])]
print 'facebook n2007 and n2008: ',len(fmtotal.index);
fbn20078 = fmtotal.copy()
fmtotal.to_csv('fbn20078.csv', sep='\t', index=None);

frames=[n8,n9]
pm = pd.concat(frames, ignore_index = True)
dt = pm[pm.duplicated(['user1', 'user2'], keep='last') | pm.duplicated(['user1','user2'])]
fmtotal = dt[dt.duplicated(['user1', 'user2'])]
print 'facebook n2008 and n2009: ',len(fmtotal.index);
fbn20089 = fmtotal.copy()
fmtotal.to_csv('fbn20089.csv', sep='\t', index=None);

frames=[y6,y7]
pm = pd.concat(frames, ignore_index = True)
dt = pm[pm.duplicated(['user1', 'user2'], keep='last') | pm.duplicated(['user1','user2'])]
fmtotal = dt[dt.duplicated(['user1', 'user2'])]
print 'facebook y2006 and y2007: ',len(fmtotal.index);
fby20067 = fmtotal.copy()
fmtotal.to_csv('fby20067.csv', sep='\t', index=None);

frames=[y7,y8]
pm = pd.concat(frames, ignore_index = True)
dt = pm[pm.duplicated(['user1', 'user2'], keep='last') | pm.duplicated(['user1','user2'])]
fmtotal = dt[dt.duplicated(['user1', 'user2'])]
print 'facebook y2007 and y2008: ',len(fmtotal.index);
fby20078 = fmtotal.copy()
fmtotal.to_csv('fby20078.csv', sep='\t', index=None);

frames=[y8,y9]
pm = pd.concat(frames, ignore_index = True)
dt = pm[pm.duplicated(['user1', 'user2'], keep='last') | pm.duplicated(['user1','user2'])]
fmtotal = dt[dt.duplicated(['user1', 'user2'])]
print 'facebook y2008 and y2009: ',len(fmtotal.index);
fby20089 = fmtotal.copy()
fmtotal.to_csv('fby20089.csv', sep='\t', index=None);

frames=[n6,y7]
pm = pd.concat(frames, ignore_index = True)
dt = pm[pm.duplicated(['user1', 'user2'], keep='last') | pm.duplicated(['user1','user2'])]
fmtotal = dt[dt.duplicated(['user1', 'user2'])]
print 'facebook n2006 and y2007: ',len(fmtotal.index);
fbny20067 = fmtotal.copy()
fmtotal.to_csv('fbny20067.csv', sep='\t', index=None);

frames=[n7,y8]
pm = pd.concat(frames, ignore_index = True)
dt = pm[pm.duplicated(['user1', 'user2'], keep='last') | pm.duplicated(['user1','user2'])]
fmtotal = dt[dt.duplicated(['user1', 'user2'])]
print 'facebook n2007 and y2008: ',len(fmtotal.index);
fbny20078 = fmtotal.copy()
fmtotal.to_csv('fbny20078.csv', sep='\t', index=None);

frames=[n8,y9]
pm = pd.concat(frames, ignore_index = True)
dt = pm[pm.duplicated(['user1', 'user2'], keep='last') | pm.duplicated(['user1','user2'])]
fmtotal = dt[dt.duplicated(['user1', 'user2'])]
print 'facebook n2008 and y2009: ',len(fmtotal.index);
fbny20089 = fmtotal.copy()
fmtotal.to_csv('fbny20089.csv', sep='\t', index=None);


##
##
##frames=[fb2006,fb2007]
##pm = pd.concat(frames, ignore_index = True)
##dt = pm[pm.duplicated(['user1', 'user2'], keep='last') | pm.duplicated(['user1','user2'])]
##fmtotal = dt[dt.duplicated(['user1', 'user2'])]
##print 'facebook 2006 and 2007: ',len(fmtotal.index);
##fb20067 = fmtotal.copy()
##fmtotal.to_csv('fb20067.csv', sep='\t', index=None);
##
##
##
##frames=[fb2007,fb2008]
##pm = pd.concat(frames, ignore_index = True)
##dt = pm[pm.duplicated(['user1', 'user2'], keep='last') | pm.duplicated(['user1','user2'])]
##fmtotal = dt[dt.duplicated(['user1', 'user2'])]
##print 'facebook 2007 and 2008: ',len(fmtotal.index);
##fb20078 = fmtotal.copy()
##fmtotal.to_csv('fb20078.csv', sep='\t', index=None);
##
##
##frames=[fb2008,fb2009]
##pm = pd.concat(frames, ignore_index = True)
##dt = pm[pm.duplicated(['user1', 'user2'], keep='last') | pm.duplicated(['user1','user2'])]
##fmtotal = dt[dt.duplicated(['user1', 'user2'])]
##print 'facebook 2008 and 2009: ',len(fmtotal.index);
##fb20089 = fmtotal.copy()
##fmtotal.to_csv('fb20089.csv', sep='\t', index=None);
##
##
##
##frames=[fb20067,fb20078]
##pm = pd.concat(frames, ignore_index = True)
##dt = pm[pm.duplicated(['user1', 'user2'], keep='last') | pm.duplicated(['user1','user2'])]
##fmtotal = dt[dt.duplicated(['user1', 'user2'])]
##print 'facebook 20067 and 20078: ',len(fmtotal.index);
##fb200678 = fmtotal.copy()
##fmtotal.to_csv('fb200678.csv', sep='\t', index=None);
##
##
##
##frames=[fb20078,fb20089]
##pm = pd.concat(frames, ignore_index = True)
##dt = pm[pm.duplicated(['user1', 'user2'], keep='last') | pm.duplicated(['user1','user2'])]
##fmtotal = dt[dt.duplicated(['user1', 'user2'])]
##print 'facebook 20078 and 20089: ',len(fmtotal.index);
##fb200789 = fmtotal.copy()
##fmtotal.to_csv('fb200789.csv', sep='\t', index=None);
##
##
##frames=[fb200678,fb200789]
##pm = pd.concat(frames, ignore_index = True)
##dt = pm[pm.duplicated(['user1', 'user2'], keep='last') | pm.duplicated(['user1','user2'])]
##fmtotal = dt[dt.duplicated(['user1', 'user2'])]
##print 'facebook 200678 and 200789: ',len(fmtotal.index);
##fb2006789 = fmtotal.copy()
##fmtotal.to_csv('fb2006789.csv', sep='\t', index=None);
