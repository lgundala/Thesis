import pandas as pd

##change folder and file name wrt language 
train_M0new = pd.read_csv('Italian/train_M0new.csv', delimiter=';', encoding='utf-8', header=None);
train_M1new = pd.read_csv('Italian/train_M1new.csv', delimiter=';', encoding='utf-8', header=None);
train_M2new = pd.read_csv('Italian/train_M2new.csv', delimiter=';', encoding='utf-8', header=None);
train_M3new = pd.read_csv('Italian/train_M3new.csv', delimiter=';', encoding='utf-8', header=None);
train_M4new = pd.read_csv('Italian/train_M4new.csv', delimiter=';', encoding='utf-8', header=None);
train_M5new = pd.read_csv('Italian/train_M5new.csv', delimiter=';', encoding='utf-8', header=None);
train_M6new = pd.read_csv('Italian/train_M6new.csv', delimiter=';', encoding='utf-8', header=None);
train_M7new = pd.read_csv('Italian/train_M7new.csv', delimiter=';', encoding='utf-8', header=None);
train_M8new = pd.read_csv('Italian/train_M8new.csv', delimiter=';', encoding='utf-8', header=None);
train_M9new = pd.read_csv('Italian/train_M9new.csv', delimiter=';', encoding='utf-8', header=None);
test_M0new = pd.read_csv('Italian/test_M0new.csv', delimiter=';', encoding='utf-8', header=None);
test_M1new = pd.read_csv('Italian/test_M1new.csv', delimiter=';', encoding='utf-8', header=None);
test_M2new = pd.read_csv('Italian/test_M2new.csv', delimiter=';', encoding='utf-8', header=None);
test_M3new = pd.read_csv('Italian/test_M3new.csv', delimiter=';', encoding='utf-8', header=None);
test_M4new = pd.read_csv('Italian/test_M4new.csv', delimiter=';', encoding='utf-8', header=None);
test_M5new = pd.read_csv('Italian/test_M5new.csv', delimiter=';', encoding='utf-8', header=None);
test_M6new = pd.read_csv('Italian/test_M6new.csv', delimiter=';', encoding='utf-8', header=None);
test_M7new = pd.read_csv('Italian/test_M7new.csv', delimiter=';', encoding='utf-8', header=None);
test_M8new = pd.read_csv('Italian/test_M8new.csv', delimiter=';', encoding='utf-8', header=None);
test_M9new = pd.read_csv('Italian/test_M9new.csv', delimiter=';', encoding='utf-8', header=None);
FrenchNpTime = pd.read_csv('npTimeFeatures.txt', delimiter=',', encoding='utf-8', header=None);
FrenchPpTime = pd.read_csv('ppTimeFeatures.txt', delimiter=',', encoding='utf-8', header=None);

frames=[FrenchNpTime,FrenchPpTime]
pm = pd.concat(frames)

test_M0new = pd.merge(test_M0new, pm, how='left', on=0);
test_M1new = pd.merge(test_M1new, pm, how='left', on=0);
test_M2new = pd.merge(test_M2new, pm, how='left', on=0);
test_M3new = pd.merge(test_M3new, pm, how='left', on=0);
test_M4new = pd.merge(test_M4new, pm, how='left', on=0);
test_M5new = pd.merge(test_M5new, pm, how='left', on=0);
test_M6new = pd.merge(test_M6new, pm, how='left', on=0);
test_M7new = pd.merge(test_M7new, pm, how='left', on=0);
test_M8new = pd.merge(test_M8new, pm, how='left', on=0);
test_M9new = pd.merge(test_M9new, pm, how='left', on=0);
train_M0new = pd.merge(train_M0new, pm, how='left', on=0);
train_M1new = pd.merge(train_M1new, pm, how='left', on=0);
train_M2new = pd.merge(train_M2new, pm, how='left', on=0);
train_M3new = pd.merge(train_M3new, pm, how='left', on=0);
train_M4new = pd.merge(train_M4new, pm, how='left', on=0);
train_M5new = pd.merge(train_M5new, pm, how='left', on=0);
train_M6new = pd.merge(train_M6new, pm, how='left', on=0);
train_M7new = pd.merge(train_M7new, pm, how='left', on=0);
train_M8new = pd.merge(train_M8new, pm, how='left', on=0);
train_M9new = pd.merge(train_M9new, pm, how='left', on=0);

del test_M0new[610]
del test_M1new[610]
del test_M2new[610]
del test_M3new[610]
del test_M4new[610]
del test_M5new[610]
del test_M6new[610]
del test_M7new[610]
del test_M8new[610]
del test_M9new[610]



del train_M0new[610]
del train_M1new[610]
del train_M2new[610]
del train_M3new[610]
del train_M4new[610]
del train_M5new[610]
del train_M6new[610]
del train_M7new[610]
del train_M8new[610]
del train_M9new[610]
##
##
##del test_M0new['8_x']
##del test_M1new['8_x']
##del test_M2new['8_x']
##del test_M3new['8_x']
##del test_M4new['8_x']
##del test_M5new['8_x']
##del test_M6new['8_x']
##del test_M7new['8_x']
##del test_M8new['8_x']
##del test_M9new['8_x']
##


##del train_M0new['8_x']
##del train_M1new['8_x']
##del train_M2new['8_x']
##del train_M3new['8_x']
##del train_M4new['8_x']
##del train_M5new['8_x']
##del train_M6new['8_x']
##del train_M7new['8_x']
##del train_M8new['8_x']
##del train_M9new['8_x']

del test_M0new['1_x']
del test_M1new['1_x']
del test_M2new['1_x']
del test_M3new['1_x']
del test_M4new['1_x']
del test_M5new['1_x']
del test_M6new['1_x']
del test_M7new['1_x']
del test_M8new['1_x']
del test_M9new['1_x']



del train_M0new['1_x']
del train_M1new['1_x']
del train_M2new['1_x']
del train_M3new['1_x']
del train_M4new['1_x']
del train_M5new['1_x']
del train_M6new['1_x']
del train_M7new['1_x']
del train_M8new['1_x']
del train_M9new['1_x']

del test_M0new['2_x']
del test_M1new['2_x']
del test_M2new['2_x']
del test_M3new['2_x']
del test_M4new['2_x']
del test_M5new['2_x']
del test_M6new['2_x']
del test_M7new['2_x']
del test_M8new['2_x']
del test_M9new['2_x']



del train_M0new['2_x']
del train_M1new['2_x']
del train_M2new['2_x']
del train_M3new['2_x']
del train_M4new['2_x']
del train_M5new['2_x']
del train_M6new['2_x']
del train_M7new['2_x']
del train_M8new['2_x']
del train_M9new['2_x']

del test_M0new['3_x']
del test_M1new['3_x']
del test_M2new['3_x']
del test_M3new['3_x']
del test_M4new['3_x']
del test_M5new['3_x']
del test_M6new['3_x']
del test_M7new['3_x']
del test_M8new['3_x']
del test_M9new['3_x']



del train_M0new['3_x']
del train_M1new['3_x']
del train_M2new['3_x']
del train_M3new['3_x']
del train_M4new['3_x']
del train_M5new['3_x']
del train_M6new['3_x']
del train_M7new['3_x']
del train_M8new['3_x']
del train_M9new['3_x']

del test_M0new['4_x']
del test_M1new['4_x']
del test_M2new['4_x']
del test_M3new['4_x']
del test_M4new['4_x']
del test_M5new['4_x']
del test_M6new['4_x']
del test_M7new['4_x']
del test_M8new['4_x']
del test_M9new['4_x']



del train_M0new['4_x']
del train_M1new['4_x']
del train_M2new['4_x']
del train_M3new['4_x']
del train_M4new['4_x']
del train_M5new['4_x']
del train_M6new['4_x']
del train_M7new['4_x']
del train_M8new['4_x']
del train_M9new['4_x']

del test_M0new['5_x']
del test_M1new['5_x']
del test_M2new['5_x']
del test_M3new['5_x']
del test_M4new['5_x']
del test_M5new['5_x']
del test_M6new['5_x']
del test_M7new['5_x']
del test_M8new['5_x']
del test_M9new['5_x']



del train_M0new['5_x']
del train_M1new['5_x']
del train_M2new['5_x']
del train_M3new['5_x']
del train_M4new['5_x']
del train_M5new['5_x']
del train_M6new['5_x']
del train_M7new['5_x']
del train_M8new['5_x']
del train_M9new['5_x']

del test_M0new['6_x']
del test_M1new['6_x']
del test_M2new['6_x']
del test_M3new['6_x']
del test_M4new['6_x']
del test_M5new['6_x']
del test_M6new['6_x']
del test_M7new['6_x']
del test_M8new['6_x']
del test_M9new['6_x']



del train_M0new['6_x']
del train_M1new['6_x']
del train_M2new['6_x']
del train_M3new['6_x']
del train_M4new['6_x']
del train_M5new['6_x']
del train_M6new['6_x']
del train_M7new['6_x']
del train_M8new['6_x']
del train_M9new['6_x']

del test_M0new['37_y']
del test_M1new['37_y']
del test_M2new['37_y']
del test_M3new['37_y']
del test_M4new['37_y']
del test_M5new['37_y']
del test_M6new['37_y']
del test_M7new['37_y']
del test_M8new['37_y']
del test_M9new['37_y']



del train_M0new['37_y']
del train_M1new['37_y']
del train_M2new['37_y']
del train_M3new['37_y']
del train_M4new['37_y']
del train_M5new['37_y']
del train_M6new['37_y']
del train_M7new['37_y']
del train_M8new['37_y']
del train_M9new['37_y']

del test_M0new['38_y']
del test_M1new['38_y']
del test_M2new['38_y']
del test_M3new['38_y']
del test_M4new['38_y']
del test_M5new['38_y']
del test_M6new['38_y']
del test_M7new['38_y']
del test_M8new['38_y']
del test_M9new['38_y']



del train_M0new['38_y']
del train_M1new['38_y']
del train_M2new['38_y']
del train_M3new['38_y']
del train_M4new['38_y']
del train_M5new['38_y']
del train_M6new['38_y']
del train_M7new['38_y']
del train_M8new['38_y']
del train_M9new['38_y']


test_M0new.to_csv('test_M0.csv',sep=',',encoding='utf-8',header=0);
test_M1new.to_csv('test_M1.csv',sep=',',encoding='utf-8',header=0);
test_M2new.to_csv('test_M2.csv',sep=',',encoding='utf-8',header=0);
test_M3new.to_csv('test_M3.csv',sep=',',encoding='utf-8',header=0);
test_M4new.to_csv('test_M4.csv',sep=',',encoding='utf-8',header=0);
test_M5new.to_csv('test_M5.csv',sep=',',encoding='utf-8',header=0);
test_M6new.to_csv('test_M6.csv',sep=',',encoding='utf-8',header=0);
test_M7new.to_csv('test_M7.csv',sep=',',encoding='utf-8',header=0);
test_M8new.to_csv('test_M8.csv',sep=',',encoding='utf-8',header=0);
test_M9new.to_csv('test_M9.csv',sep=',',encoding='utf-8',header=0);
train_M0new.to_csv('train_M0.csv',sep=',',encoding='utf-8',header=0);
train_M1new.to_csv('train_M1.csv',sep=',',encoding='utf-8',header=0);
train_M2new.to_csv('train_M2.csv',sep=',',encoding='utf-8',header=0);
train_M3new.to_csv('train_M3.csv',sep=',',encoding='utf-8',header=0);
train_M4new.to_csv('train_M4.csv',sep=',',encoding='utf-8',header=0);
train_M5new.to_csv('train_M5.csv',sep=',',encoding='utf-8',header=0);
train_M6new.to_csv('train_M6.csv',sep=',',encoding='utf-8',header=0);
train_M7new.to_csv('train_M7.csv',sep=',',encoding='utf-8',header=0);
train_M8new.to_csv('train_M8.csv',sep=',',encoding='utf-8',header=0);
train_M9new.to_csv('train_M9.csv',sep=',',encoding='utf-8',header=0);

frames = [test_M1new, test_M1new, test_M2new, test_M3new, test_M4new, test_M5new, test_M6new ,test_M7new, test_M8new, test_M9new, train_M0new, train_M1new, train_M3new, train_M4new, train_M5new, train_M6new, train_M7new, train_M8new, train_M9new]
pm = pd.concat(frames)
pm.to_csv('allFiles.csv', sep=',',encoding='utf-8',header=0);

