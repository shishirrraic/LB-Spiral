import sys
from random import randint

from networkx import nx
from networkx.readwrite import json_graph

n = 8
p =0.01

G = nx.gnp_random_graph(n,p)
if nx.is_connected(G) is False:
    print("NOT CONNECTED, NEED TO ADD EDGES")

    # at first connect nodes with 0 neighbors
    for i in range(0, n):
        node = None
        neighbors = []
        for a in [n for n in G.neighbors(i)]:
            neighbors.append(a)
        if len(neighbors) == 0:
            node = i

            print("NODE ", node, " HAS 0 neighbours. CONNECTING IT TO THE LONGEST COMPONENT")

            longest_connected_comp = sorted(nx.connected_components(G), key=len, reverse=True)[0]
            # a node in longest_connected_comp
            comp_node = next(iter(longest_connected_comp))
            print("SOURCE NODE ", node)
            print("DESTINATION NODE ", comp_node)
            G.add_edge(node, comp_node, weight=randint(1, 1))

    for a in range(0, len(sorted(nx.connected_components(G), key=len, reverse=True))):
        print(sorted(nx.connected_components(G), key=len, reverse=True)[a])

    #     attach smaller components to the longest component
    for i in list(reversed(range(1, len(sorted(nx.connected_components(G), key=len, reverse=True))))):
        source_node = next(iter(sorted(nx.connected_components(G), key=len, reverse=True)[i]))
        dest_node = next(iter(sorted(nx.connected_components(G), key=len, reverse=True)[0]))
        print("SOURCE NODEE",source_node)
        print("DEST NODEE",dest_node)
        G.add_edge(int(source_node), int(dest_node), weight = randint(1, 1))

assert nx.is_connected(G)

print(sorted(sorted(nx.connected_components(G), key=len, reverse=True)))

# some properties
print("node degree clustering")
for v in nx.nodes(G):
    print('%s %d %f' % (v, nx.degree(G, v), nx.clustering(G, v)))

# G.add_edge(0, 3, weight=4)
for e in G.edges:
    a = e[0]
    b = e[1]

    w = randint(1,1)
    G.add_edge(int(e[0]), int(e[1]), weight=w)
# print (a)

# Drawing the graph
node_pos = nx.spring_layout(G)

edge_weight = nx.get_edge_attributes(G, 'weight')

# Draw the nodes
nx.draw_networkx(G, node_pos, node_color='grey', node_size=100)

# Draw the edges
nx.draw_networkx_edges(G, node_pos, edge_color='black')

# Draw the edge labels
nx.draw_networkx_edge_labels(G, node_pos, edge_color='red', edge_labels=edge_weight)

# write to a file
# nx.write_gml(G, "test_gml")
# json_graph.node_link_data(G)
# nx.write_edgelist(G,"test.edgeList")

# print the adjacency list to terminal
print("THE ADJACENCY LIST IS ")
# try:
#     nx.write_adjlist(G, sys.stdout)
# except TypeError:
#     nx.write_adjlist(G, sys.stdout.buffer)

for a in [(n, nbrdict) for n, nbrdict in G.adjacency()]:
    print (a)

for i in range(0,n):
    print("NODE " ,i)
    temp = []
    for a in [n for n in G.neighbors(i)]:
        temp.append(a)
    print(temp)
    assert len(temp)>0
# diameter
paths_for_diameter = nx.shortest_path_length(G, weight='weight')
ecc = nx.eccentricity(G, sp=dict(paths_for_diameter))
diameter = nx.diameter(G, e=ecc)
print("OK diameter without weight?", nx.diameter(G))
print('The graph diameter with weight is ', diameter)

# nx.write_weighted_edgelist(G, str(n)+'_'+str(p)+'diamter'+str(diameter)+'_newtest.weighted.edgelist')

