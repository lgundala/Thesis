import csv
import networkx as nx
def sort(array):
    less = []
    equal = []
    greater = []
    if len(array) > 1:
        pivot = array[0]
        for x in array:
            if x < pivot:
                less.append(x)
            if x == pivot:
                equal.append(x)
            if x > pivot:
                greater.append(x)
        # Don't forget to return something!
        return sort(less)+equal+sort(greater)  # Just use the + operator to join lists
    # Note that you want equal ^^^^^ not pivot
    else:  # You need to hande the part at the end of the recursion - when you only have one element in your array, just return the array.
        return array
def getstuff(filename,delimit):
    with open(filename, "r",encoding='utf8') as csvfile:
        datareader = csv.reader(csvfile,delimiter=delimit)
        count = 0
        for row in datareader:
            yield [cell for cell in row]          
ay=[12,4,5,6,7,3,1,15];
ab = sort(ay);
print(ab);
i = 1;
G1 = nx.DiGraph() ##febother directed graph
G2 = nx.Graph() ## febother undirected graph - for calculation purposes
reader = getstuff('Dataset_sample_10.csv',',');
for fl in reader:
    prev = fl[0];
    curr = fl[1];
    G1.add_node(prev);
    G1.add_node(curr);
    G1.add_edge(prev,curr,weight=fl[3]);
    G2.add_node(prev);
    G2.add_node(curr);
    G2.add_edge(prev,curr,weight=fl[3]);
Gtotal = nx.Graph()
reader2 = getstuff('FebLinksFebOther.csv','\t');
for fl in reader2:
    prev = fl[0];
    curr = fl[1];
    Gtotal.add_node(prev);
    Gtotal.add_node(curr);
    Gtotal.add_edge(prev,curr,weight=fl[3]);
Glinks = nx.DiGraph()
reader3 = getstuff('FebruaryLinkswithTabs.csv','\t');
for fl in reader3:
    prev = fl[0];
    curr = fl[1];
    Glinks.add_node(prev);
    Glinks.add_node(curr);
    Glinks.add_edge(prev,curr,weight=fl[3]);
gedges = {}
totaledges = {}

for u in G1.nodes():
              gedges[u] = [v for v in G1.neighbors(u)];
for u in Gtotal.nodes():
              totaledges[u] = Gtotal.edges(u,data=True);
for u in gedges:
    sigma = 0;
    pi = 1;
    sigmaE = 0;
    c = 0;
    for v in gedges.get(u):
        p = G1.get_edge_data(u,v);
        nst = p['weight'];
        print(u,v,nst);
        for k in totaledges(u):
            n = 0;
            w = k[2];
            n = n +w['weight'];
        ns = n;
        pst = nst/ns;
        ws = log(ns);
        pi = pi*(1-pst);
        if f= f1:
            c_cap = ws*sigma;
        else if f = f2:
            c_cap = ws*(1-pi)
        else if f = f3:
            c_cap = ws*sigma/(sigma+sigmaE);
        ay.append((u,v,c_cap));
        c = c_cap;


        
    
                  
                
              
    
              
