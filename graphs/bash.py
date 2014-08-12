import networkx as nx


g = nx.read_gml('distestGauss.gml')
h = nx.degree_histogram(g)
d = nx.degree(g)
print 'histogram: '
print h
print 'degree'
print d


sum = 0

for x in h:
    sum = sum + x

print sum
