import sys

import matplotlib.pyplot as plt
from networkx import nx, json_graph


plt.axis('off')

# graph = nx.read_gml("test_gml")
graph = nx.read_weighted_edgelist("8_0.01diamter3_newtest.weighted.edgelist")

# some properties
print("node degree clustering")
for v in nx.nodes(graph):
    print('%s %d %f' % (v, nx.degree(graph, v), nx.clustering(graph, v)))

# print the adjacency list to terminal
try:
    nx.write_adjlist(graph, sys.stdout)
except TypeError:
    nx.write_adjlist(graph, sys.stdout.buffer)

node_pos = nx.spring_layout(graph)

edge_weight = nx.get_edge_attributes(graph, 'weight')
# Draw the nodes
nx.draw_networkx(graph, node_pos, node_color='grey', node_size=100)

# Draw the edges
nx.draw_networkx_edges(graph, node_pos, edge_color='black')

# Draw the edge labels
nx.draw_networkx_edge_labels(graph, node_pos, edge_color='red', edge_labels=edge_weight)
print(edge_weight)
print(graph)
print(graph.nodes())
print("THE GRAPH")
plt.show()


print("THE ADJACENCY LIST IS ")
# try:
#     nx.write_adjlist(G, sys.stdout)
# except TypeError:
#     nx.write_adjlist(G, sys.stdout.buffer)

for a in [(n, nbrdict) for n, nbrdict in graph.adjacency()]:
    print (a)