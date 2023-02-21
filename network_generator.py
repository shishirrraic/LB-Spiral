import sys
import os
import math
from random import randint
import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph

num_nodes = 64
erdos_renyi_prob = 0.01
internet_graph_seed = None  # optional


def add_edge_weights(graph):
    for e in graph.edges:
        w = randint(1, 10)
        graph.add_edge(e[0], e[1], weight=w)


def get_diameter(graph):
    paths_for_diameter = nx.shortest_path_length(graph, weight='weight')
    ecc = nx.eccentricity(graph, sp=dict(paths_for_diameter))
    diameter = nx.diameter(graph, e=ecc)
    return diameter


# write to a file
def write_to_a_file(graph, param):
    diameter = get_diameter(graph)
    nx.write_graphml(graph, './graphs/' + str(num_nodes) + str(param) + '_diameter' + str(diameter) + 'test.edgelist')


def build_random_graph():
    random_graph = nx.gnp_random_graph(num_nodes, erdos_renyi_prob)

    if nx.is_connected(random_graph) is False:
        # print("NOT CONNECTED, NEED TO ADD EDGES")
        # at first connect nodes with 0 neighbors
        for i in range(0, num_nodes):
            print("HERE")
            node = None
            neighbors = []
            for a in [n for n in random_graph.neighbors(i)]:
                neighbors.append(a)
            if len(neighbors) == 0:
                node = i

                # print("NODE ", node, " HAS 0 neighbours. CONNECTING IT TO THE LONGEST COMPONENT")
                longest_connected_comp = sorted(nx.connected_components(random_graph), key=len, reverse=True)[0]
                # a node in longest_connected_comp
                comp_node = next(iter(longest_connected_comp))
                random_graph.add_edge(node, comp_node, weight=randint(1, 1))

        #     attach smaller components to the longest component
        for i in list(reversed(range(1, len(sorted(nx.connected_components(random_graph), key=len, reverse=True))))):
            print("HERE1")
            source_node = next(iter(sorted(nx.connected_components(random_graph), key=len, reverse=True)[i]))
            dest_node = next(iter(sorted(nx.connected_components(random_graph), key=len, reverse=True)[0]))
            random_graph.add_edge(int(source_node), int(dest_node), weight=randint(1, 1))

    assert nx.is_connected(random_graph)
    add_edge_weights(random_graph)

    return random_graph


def build_internet_graph():
    internet_graph = nx.random_internet_as_graph(num_nodes)
    add_edge_weights(internet_graph)

    assert nx.is_connected(internet_graph)
    return internet_graph


def build_grid_graph():
    grid_row = int(math.sqrt(num_nodes))
    grid_col = int(math.sqrt(num_nodes))
    grid_graph = nx.grid_graph([grid_row, grid_col])
    grid_graph = nx.convert_node_labels_to_integers(grid_graph)
    # grid_graph = nx.grid_2d_graph(grid_row, grid_col)
    assert nx.is_connected(grid_graph)
    add_edge_weights(grid_graph)
    return grid_graph


def build_graphs():
    # random_graph = build_random_graph()
    # write_to_a_file(random_graph, "random")
    # draw(random_graph)

    internet_graph = build_internet_graph()
    write_to_a_file(internet_graph, "internet")
    # draw(internet_graph)

    # grid_graph = build_grid_graph()
    # write_to_a_file(grid_graph, "grid")
    # draw(grid_graph)


def draw(graph):
    plt.axis('off')

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
    nx.draw_networkx_edge_labels(graph, node_pos, edge_labels=edge_weight)
    print(edge_weight)
    print(graph)
    print(graph.nodes())
    print("THE GRAPH")
    plt.show()


def test_graphs():
    directory = './graphs/'

    for file in os.listdir(directory):
        filename = os.fsdecode(file)

        print(os.path.join(directory, filename))
        graph = nx.read_graphml(os.path.join(directory, filename))

        draw(graph)


if __name__ == '__main__':
    build_graphs()
    # test_graphs()
