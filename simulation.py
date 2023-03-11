import math
from network import Network
from binary_tree import BinaryTree
from object import Object
from peer import Peer
from cluster import Cluster
from collections import Counter, OrderedDict
import sys
from binary_tree import print_tree, get_leaves
import matplotlib.pyplot as plt
import networkx as nx
import copy
import random
import json
import os
import pickle
import log
import time

logger = log.get_logger(__name__)


class Simulation:
    """
        The Simulation class
        Attributes:
        ----------
            op_count (int): number of operations
            peer_count (int): number of peers
            peers (list): peers in the network
            height_of_clusters (int): height of the clusters
            root_cluster_id (str): root cluster
            size (int): network's size
        """

    def __init__(self, operation_count, peer_count, graph_input):
        logger.debug("INSIDE SIMULATION INIT")

        self.__op_count = int(operation_count)
        self.__peer_count = int(peer_count)

        plt.axis('off')

        graph = nx.read_graphml('./graphs/' + graph_input)

        logger.debug("{}".format(graph))

        # some properties
        logger.debug("node degree clustering")
        for v in nx.nodes(graph):
            logger.debug('{} {} {}'.format(v, nx.degree(graph, v), nx.clustering(graph, v)))

        # print the adjacency list to terminal
        try:
            nx.write_adjlist(graph, sys.stdout)
        except TypeError:
            nx.write_adjlist(graph, sys.stdout.buffer)

        # node_pos = nx.spring_layout(graph)
        #
        # edge_weight = nx.get_edge_attributes(graph, 'weight')
        # # Draw the nodes
        # nx.draw_networkx(graph, node_pos, node_color='grey', node_size=100)
        # # Draw the edges
        # nx.draw_networkx_edges(graph, node_pos, edge_color='black')
        # # Draw the edge labels
        # nx.draw_networkx_edge_labels(graph, node_pos, edge_color='red', edge_labels=edge_weight)

        # plt.show()

        paths_for_diameter = nx.shortest_path_length(graph, weight='weight')
        ecc = nx.eccentricity(graph, sp=dict(paths_for_diameter))
        self.__diameter = nx.diameter(graph, e=ecc)

        logger.debug('The graph diameter is {}'.format(self.__diameter))
        self.__height_of_cluster = math.ceil(math.log(self.__diameter, 2)) + 1

        if not os.path.exists('./graphs/' + graph_input + '_network'):
            logger.debug("CREATING NEW NETWORK")
            self.__network = Network(graph)
            self.setup_peers(graph)
            # self._build_clusters_exactly_logn_membership(graph)
            self.build_clusters_at_least_one_and_at_most_logn(graph)
            # self._build_clusters_no_overlapping(graph)
            self.build_tree(graph)
            # self.save_network('./graphs/' + graph_input + '_network')
            logger.debug("DONE CREATING NEW NETWORK")
            exit(0)

        else:
            logger.debug("LOADING NETWORK FROM INPUT FILE")
            # load network
            self.load_network('./graphs/' + graph_input + '_network')

    def get_network(self):
        return self.__network

    def setup_peers(self, graph):
        for n in graph.nodes():
            neighbors = {}
            for e in graph.edges(n):
                neighbors[int(e[1])] = graph[e[0]][e[1]]['weight']
            peer = Peer(int(n), neighbors)
            self.__network.add_peer(peer_id=int(n), peer=peer)

    def at_least_one(self, clustered_peers_count):
        for i in range(self.__peer_count):
            if clustered_peers_count[str(i)] < 1:
                logger.debug("AT LEAST ONE FALSE BECAUSE OF {}".format(str(i)))
                return i

        return -1

    def at_most_logn(self, clustered_peers_count):
        for i in range(self.__peer_count):
            if clustered_peers_count[str(i)] > math.log(self.__peer_count, 2):
                logger.debug("AT MOST LOG N FALSE BECAUSE OF {}".format(str(i)))
                return i

        return -1

    def build_clusters_no_overlapping(self, graph):
        logger.debug("{}".format(nx.dijkstra_path(graph, '0', '7', 'weight')))
        logger.debug("{}".format(nx.dijkstra_path_length(graph, '0', '7', 'weight')))
        paths_for_diameter = nx.shortest_path_length(graph, weight='weight')
        for path in nx.shortest_path_length(graph, weight='weight'):
            logger.debug("{}".format(path))

        ecc = nx.eccentricity(graph, sp=dict(paths_for_diameter))
        diameter = nx.diameter(graph, e=ecc)

        logger.debug("THE GRAPH DIAMETER IS {}".format(diameter))
        height_of_cluster = math.ceil(math.log(diameter, 2)) + 1
        logger.debug("HEIGHT OF HIERARCHY IS {}".format(height_of_cluster))

        # lowest level clusters
        logger.debug("LOWEST LEVEL CLUSTERS")
        for n in graph.nodes():
            # paths = nx.single_source_dijkstra_path_length(graph, n, 0, weight='weight')
            cluster_graph = graph.subgraph([n])
            cluster = Cluster('c' + str(n) + '_l' + '0', cluster_graph, 0)
            self.__network.add_cluster_level(0)

            self.__network.add_cluster(0, cluster)
            self.__network.draw_cluster(cluster.get_cluster_id())

        for i in range(int(height_of_cluster)):
            self.__network.add_cluster_level(i + 1)

            logger.debug("AT LEVEL --------- ", i + 1)
            distance = pow(2, i + 1)

            logger.debug("THE DISTANCE LIMIT IS {}".format(distance))
            clustered_peers = []
            for n in range(self.__peer_count):
                # for n in graph.nodes():
                logger.debug("CLUSTERING PEER {}".format(n))
                if n in clustered_peers:
                    logger.debug("PEER {} already clustered".format(n))
                    continue

                paths_found = nx.single_source_dijkstra_path_length(graph, str(n), distance, weight='weight')
                peers_to_cluster = []
                logger.debug("PATHS FOUND IN THE LEVEL {}".format(paths_found))
                for peer in paths_found:
                    if int(peer) in clustered_peers:
                        logger.debug("PEER ALREADY CLUSTERED IN ANOTHER CLUSTER")
                    else:
                        clustered_peers.append(int(peer))
                        peers_to_cluster.append(int(peer))

                cluster_graph = graph.subgraph([str(i) for i in peers_to_cluster])

                # cluster = Cluster('c' + str(n) + '_l' + str(i + 1), peers_to_cluster)
                cluster = Cluster('c' + str(n) + '_l' + str(i + 1), cluster_graph, i + 1)
                self.__network.add_cluster(i + 1, cluster)
                # self._network.draw_cluster(cluster.cluster_id)

    def build_clusters_exactly_logn_membership(self, graph):
        paths_for_diameter = nx.shortest_path_length(graph, weight='weight')
        for path in nx.shortest_path_length(graph, weight='weight'):
            logger.debug("{}".format(path))

        ecc = nx.eccentricity(graph, sp=dict(paths_for_diameter))
        diameter = nx.diameter(graph, e=ecc)

        logger.debug('THE GRAPH DIAMETER IS {}'.format(diameter))
        height_of_cluster = math.ceil(math.log(diameter, 2)) + 1
        logger.debug('HEIGHT OF THE HIERARCHY IS {}'.format(height_of_cluster))
        # lowest level clusters

        self.__network.add_cluster_level(0)

        for n in graph.nodes():
            # paths = nx.single_source_dijkstra_path_length(graph, n, 0, weight='weight')
            cluster_graph = graph.subgraph([n])
            cluster = Cluster('c' + str(n) + '_l' + '0', cluster_graph, 0)
            self.__network.add_cluster(0, cluster)
            # self._network.draw_cluster(cluster.cluster_id)

        for i in range(int(height_of_cluster)):
            self.__network.add_cluster_level(i + 1)
            clustered_peers_list = []
            cluster_ids_list = []

            logger.debug('AT LEVEL ------- {}'.format(i + 1))
            distance = pow(2, i + 1)

            logger.debug('THE DISTANCE LIMIT IS {}'.format(distance))
            clustered_peers = []
            # for naming the clusters properly
            n = 0
            while n < self.__peer_count:
                incomplete = False
                # for n in graph.nodes():
                logger.debug('CLUSTERING PEER {}'.format(n))

                paths_found = nx.single_source_dijkstra_path_length(graph, str(n), distance, weight='weight')
                peers_to_cluster = []
                logger.debug('PATHS FOUND IN LEVEL {}'.format(paths_found))
                for peer in paths_found:
                    clustered_peers.append(int(peer))
                    peers_to_cluster.append(int(peer))
                    clustered_peers_list.append(peer)

                cluster_graph = graph.subgraph([str(i) for i in peers_to_cluster])

                c_id = 'c' + str(n) + '_l' + str(i + 1)
                cluster_ids_list.append(c_id)
                cluster_ids_count = Counter(cluster_ids_list)
                c_id = c_id + "_" + str(cluster_ids_count[c_id])
                cluster = Cluster(c_id, cluster_graph, i + 1)
                self.__network.add_cluster(i + 1, cluster)
                # self._network.draw_cluster(cluster.cluster_id)
                clustered_peers_count = Counter(clustered_peers_list)
                for j in range(self.__peer_count):
                    if clustered_peers_count[str(j)] < math.log(self.__peer_count, 2):
                        incomplete = True

                if not incomplete:
                    # make peers fall in exactly log(n) clusters
                    for peer in range(self.__peer_count):
                        # count of peers in all clusters
                        peer_count = clustered_peers_count[str(peer)]
                        while peer_count > math.log(self.__peer_count, 2):
                            for cluster in reversed(self.__network.clusters_by_level(i + 1)):
                                if clustered_peers_count[str(peer)] > math.log(self.__peer_count, 2) and \
                                        cluster.get_graph().has_node(str(peer)):
                                    cluster.get_graph().remove_node(str(peer))
                                    peer_count = peer_count - 1
                                    if str(peer) in clustered_peers_list:
                                        clustered_peers_list.remove(str(peer))
                                    clustered_peers_count = Counter(clustered_peers_list)

                                    break

                    break

                n += 1
                if n == self.__peer_count:
                    n = 0

        # delete empty clusters
        for i in range(int(height_of_cluster)):
            for cluster in self.__network.clusters_by_level(i):
                if nx.number_of_nodes(cluster.get_graph()) == 0:
                    self.__network.remove_cluster_by_id(cluster.get_cluster_id(), cluster.get_level())

        for i in range(int(height_of_cluster)):
            for cluster in self.__network.clusters_by_level(i):
                assert (int(nx.number_of_nodes(cluster.get_graph())) != 0)

        # make sure that there are log(n) peers in every level
        for i in range(int(height_of_cluster) - 1):
            logger.debug("VERIFYING IN LEVEL {}".format(i + 1))
            clustered_peer_list = []
            for cluster in self.__network.clusters_by_level(i + 1):
                for node in cluster.get_graph().nodes:
                    clustered_peer_list.append(str(node))
            clustered_peer_count = Counter(clustered_peer_list)
            for cl in clustered_peer_count:
                assert (clustered_peer_count[cl] == math.log(self.__peer_count, 2))

    def build_clusters_at_least_one_and_at_most_logn(self, graph):
        paths_for_diameter = nx.shortest_path_length(graph, weight='weight')

        ecc = nx.eccentricity(graph, sp=dict(paths_for_diameter))
        diameter = nx.diameter(graph, e=ecc)

        logger.debug('THE GRAPH DIAMETER IS {}'.format(diameter))
        height_of_cluster = math.ceil(math.log(diameter, 2)) + 1
        self.__network.set_height_of_clusters(height_of_cluster)
        logger.debug('HEIGHT OF HIERARCHY IS  {}'.format(height_of_cluster))
        # lowest level clusters
        logger.debug('LOWEST LEVEL CLUSTERS')
        self.__network.add_cluster_level(0)

        # level 0 clusters
        node_index = 0
        node_list = random.sample(range(0, self.__peer_count), self.__peer_count)
        while node_index < self.__peer_count:
        # for n in graph.nodes():
            n = str(node_list[node_index])
            paths = nx.single_source_dijkstra_path_length(graph, n, 0, weight='weight')
            logger.debug("{}".format(paths))
            cluster_graph = graph.subgraph([n])
            cluster = Cluster('c' + str(n) + '_l' + '0', cluster_graph, 0)
            self.__network.add_cluster(0, cluster)
            # self._network.draw_cluster(cluster.cluster_id)
            node_index += 1

        # form upper level clusters
        for i in range(int(height_of_cluster)):
            self.__network.add_cluster_level(i + 1)
            clustered_peers_list = []
            #     for naming the cluster properly
            cluster_ids_list = []

            logger.debug('AT LEVEL ----- {}'.format(i + 1))
            distance = pow(2, i + 1)
            logger.debug('THE DISTANCE LIMIT IS {}'.format(distance))
            clusterize = {}

            # changing n to build a more randomized cluster
            node_index = 0
            node_list = random.sample(range(0, self.__peer_count), self.__peer_count)

            #     iterate over the peers once
            while node_index < self.__peer_count:
                n = node_list[node_index]
                logger.info('CLUSTERING PEER {}'.format(n))

                paths_found = nx.single_source_dijkstra_path_length(graph, str(n), distance, weight='weight')

                peers_to_cluster = []
                logger.debug('PATHS FOUND IN LEVEL {}'.format(paths_found))
                tmp_peers_list_to_cluster = []
                for peer in paths_found:
                    peers_to_cluster.append(peer)
                    tmp_peers_list_to_cluster.append(peer)

                c_id = 'c' + str(n) + '_l' + str(i + 1)

                # for naming the clusters properly
                cluster_ids_list.append(c_id)
                cluster_ids_count = Counter(cluster_ids_list)

                c_id = c_id + "_" + str(cluster_ids_count[c_id])

                temp_clustered_peers_list = copy.deepcopy(clustered_peers_list)
                temp_clustered_peers_list.extend(tmp_peers_list_to_cluster)
                duplicate = False
                if self.at_most_logn(Counter(temp_clustered_peers_list)) < 0:
                    # check if duplicate peers or not
                    for inner_key, inner_value in clusterize.items():

                        if Counter(inner_value) == Counter(tmp_peers_list_to_cluster):
                            logger.debug("FOUND DUPLICATE CLUSTER")
                            duplicate = True
                            break
                    logger.debug("DOESN'T VIOLATE AT MOST LOG N")
                    if not duplicate:
                        clustered_peers_list.extend(tmp_peers_list_to_cluster)
                        clusterize[c_id] = peers_to_cluster

                node_index += 1

            logger.debug("CHECKING MEMBERSHIP")
            logger.debug("{}".format(self.at_least_one(Counter(clustered_peers_list)) < 0))
            logger.debug("{}".format(self.at_most_logn(Counter(clustered_peers_list)) < 0))
            assert (self.at_most_logn(Counter(clustered_peers_list)) < 0)

            # if not built yet build a cluster and remove peers who appear more than logn times
            missing_cluster_id = self.at_least_one(Counter(clustered_peers_list))
            while missing_cluster_id > -1:
                logger.info("PEER {} IS MISSING, BUILDING A CLUSTER AROUND IT.".format(missing_cluster_id))
                paths_found = nx.single_source_dijkstra_path_length(graph, str(missing_cluster_id), distance,
                                                                    weight='weight')
                peers_to_cluster = []
                logger.debug('PATHS FOUND IN LEVEL {}'.format(paths_found))
                tmp_peers_list_to_cluster = []
                for peer in paths_found:
                    peers_to_cluster.append(peer)
                    # clustered_peers_list.append(peer)
                    tmp_peers_list_to_cluster.append(peer)

                tmp_peers_list_to_cluster = [str(missing_cluster_id)]
                c_id = 'c' + str(missing_cluster_id) + '_l' + str(i + 1)

                # for naming the clusters properly
                cluster_ids_list.append(c_id)
                cluster_ids_count = Counter(cluster_ids_list)
                c_id = c_id + "_" + str(cluster_ids_count[c_id])

                temp_clustered_peers_list = copy.deepcopy(clustered_peers_list)
                temp_clustered_peers_list.extend(tmp_peers_list_to_cluster)
                duplicate = False
                if self.at_most_logn(Counter(temp_clustered_peers_list)) < 0:
                    # check if duplicate peers or not
                    for inner_key, inner_value in clusterize.items():
                        if Counter(inner_value) == Counter(tmp_peers_list_to_cluster):
                            logger.debug("FOUND DUPLICATE CLUSTER")
                            duplicate = True
                            break
                    logger.debug("DOESN'T VIOLATE AT MOST LOG N")
                    if not duplicate:
                        logger.debug("WHY HERE")
                        clustered_peers_list.extend(tmp_peers_list_to_cluster)
                        clusterize[c_id] = tmp_peers_list_to_cluster
                else:
                    #     find the peer that appears more than logn
                    excess_cluster_id = self.at_most_logn(Counter(temp_clustered_peers_list))
                    while excess_cluster_id != -1:
                        logger.debug("{}".format(tmp_peers_list_to_cluster))
                        logger.debug("{}".format(excess_cluster_id))
                        tmp_peers_list_to_cluster.remove(str(excess_cluster_id))
                        tmp_clustered_peers_list = copy.deepcopy(clustered_peers_list)
                        tmp_clustered_peers_list.extend(tmp_peers_list_to_cluster)
                        # removing
                        logger.debug("% APPEARS IN MORE THAN LOGN CLUSTER, REMOVING IT", excess_cluster_id)
                        excess_cluster_id = self.at_most_logn(Counter(tmp_clustered_peers_list))
                    logger.info("PREPARING THE CLUSTER FROM MODIFIED PEERS LIST {}".format(tmp_peers_list_to_cluster))
                    clustered_peers_list.extend(tmp_peers_list_to_cluster)
                    clusterize[c_id] = tmp_peers_list_to_cluster
                    assert (self.at_most_logn(Counter(clustered_peers_list)) < 0)

                missing_cluster_id = self.at_least_one(Counter(clustered_peers_list))

            assert (self.at_least_one(Counter(clustered_peers_list)) < 0)
            assert (self.at_most_logn(Counter(clustered_peers_list)) < 0)

            logger.info("{}".format(clusterize))
            logger.info("{}".format(Counter(clustered_peers_list)))

            logger.info("FINALLY ADDING CLUSTERS")
            for key in clusterize:
                assert (nx.number_of_nodes(graph.subgraph([str(i) for i in clusterize[key]])) == len(clusterize[key]))

                cluster_graph = graph.subgraph([str(i) for i in clusterize[key]])
                cluster = Cluster(key, cluster_graph, i + 1)
                if i + 1 == height_of_cluster:
                    cluster.set_root()
                    self.__network.set_root_cluster(cluster.get_cluster_id())
                self.__network.add_cluster(i + 1, cluster)
                # self._network.draw_cluster(cluster.cluster_id)

            logger.info("CLUSTERIZE {}".format(clusterize))

        return

    def build_tree(self, graph):
        logger.debug("BUILDING THE TREE")
        # for clusters in each upper level besides 0
        logger.debug("THE HEIGHT OF CLUSTERS IS {}".format(self.__height_of_cluster))

        uncontained_clusters = []
        for i in range(2, self.__height_of_cluster + 1):
            # for i in (range(2,3)):

            # at first, gather bottom level clusters which are contained in the upper level
            cluster_contain_in_upper_level_map = {}
            # list of clusters that were added to some other parent level clusters
            ignore_cluster_list = []

            for current_cluster in self.__network.clusters_by_level(i):
                # just checking
                clustered_peers_list = []
                logger.debug("JUST CHECKING ASSERT REDUNDANT CLUSTERS")
                for cluster in self.__network.clusters_by_level(i):
                    nodes = sorted(list(cluster.get_graph().nodes))
                    for node in nodes:
                        clustered_peers_list.append(node)
                logger.debug("{}".format(clustered_peers_list))
                assert (self.at_least_one(Counter(clustered_peers_list)) < 0)
                assert (self.at_most_logn(Counter(clustered_peers_list)) < 0)

                # ignore intermediate clusters
                logger.debug(current_cluster.get_cluster_id())
                cluster_contain_in_upper_level_map[current_cluster.get_cluster_id()] = []
                skip_cluster = False
                lower_level_clusters = self.__network.clusters_by_level(i - 1)

                for uncontained_cluster in uncontained_clusters:
                    lower_level_clusters.append(uncontained_cluster)

                # for lower_level_cluster in self._network.clusters_by_level(i - 1):
                for lower_level_cluster in lower_level_clusters:
                    if lower_level_cluster.intermediate():
                        logger.debug("INTERMEDIATE CLUSTER ID {}".format(current_cluster.get_cluster_id()))
                        logger.debug("INTERMEDIATE CLUSTER PEERS {}".format(current_cluster.get_peers()))
                    else:
                        if lower_level_cluster.get_cluster_id() not in ignore_cluster_list:
                            # check to see for duplicates
                            for key in cluster_contain_in_upper_level_map[current_cluster.get_cluster_id()]:
                                if Counter(self.__network.find_cluster_by_id(key).get_graph().nodes) == Counter(
                                        lower_level_cluster.get_graph().nodes):
                                    logger.debug("DUPLICATE FOUND")
                                    skip_cluster = True
                                    continue

                                for cluster_to_ignore in ignore_cluster_list:
                                    if Counter(lower_level_cluster.get_graph().nodes) == Counter(
                                            self.__network.find_cluster_by_id(cluster_to_ignore).get_graph().nodes):
                                        logger.debug("DUPLICATE ENTRIES FOUND")
                                        skip_cluster = True
                                        continue

                            # check if same members
                            if skip_cluster:
                                continue

                            result = set(lower_level_cluster.get_graph().nodes).issubset(
                                current_cluster.get_graph().nodes)

                            # logger.debug("THE MAP IS ", cluster_contain_in_upper_level_map)
                            if result:
                                cluster_contain_in_upper_level_map[current_cluster.get_cluster_id()].append(
                                    lower_level_cluster.get_cluster_id())
                                ignore_cluster_list.append(lower_level_cluster.get_cluster_id())

            # delete all uncontained clusters that are in ignore_cluster_list
            temp_uncontained_clusters = copy.deepcopy(uncontained_clusters)

            for cluster in uncontained_clusters:
                if cluster.get_cluster_id() in ignore_cluster_list:
                    logger.debug("FOUND CLUSTER {} TO REMOVE".format(cluster.get_cluster_id()))
                    index_to_delete = None
                    for index in range(0, len(temp_uncontained_clusters)):
                        if temp_uncontained_clusters[index].get_cluster_id() == cluster.get_cluster_id():
                            index_to_delete = index
                            logger.debug("INDEX TO DELETE IS {}".format(index_to_delete))
                            break
                    del (temp_uncontained_clusters[index_to_delete])
                    # temp_uncontained_cluster.remove(cluster)
            uncontained_clusters = copy.deepcopy(temp_uncontained_clusters)
            for cluster in uncontained_clusters:
                logger.debug(cluster.get_cluster_id())

            for lower_level_cluster in self.__network.clusters_by_level(i - 1):
                if lower_level_cluster.get_cluster_id() not in ignore_cluster_list:
                    if not lower_level_cluster.intermediate():
                        uncontained_clusters.append(lower_level_cluster)

            logger.debug("PREPARED CLUSTER CONTAIN MAP {}".format(cluster_contain_in_upper_level_map))

            # build the binary trees
            for root_cluster_key in cluster_contain_in_upper_level_map:
                current_tree = BinaryTree(root_cluster_key)
                height_of_tree = int(math.log(
                    1 if len(cluster_contain_in_upper_level_map[root_cluster_key]) == 0 else 2 ** (
                            len(cluster_contain_in_upper_level_map[root_cluster_key]) - 1).bit_length(), 2))

                if len(cluster_contain_in_upper_level_map[root_cluster_key]) == 1:
                    height_of_tree = 1

                lower_level_trees = []

                for bottommost_level_cluster in cluster_contain_in_upper_level_map[root_cluster_key]:
                    temp_tree = BinaryTree(bottommost_level_cluster)
                    temp_tree.set_data(self.__network.find_cluster_by_id(bottommost_level_cluster).get_peers())
                    lower_level_trees.append(temp_tree)

                for level in range(height_of_tree):
                    new_lower_level_trees = []

                    total_nodes_in_this_level = len(lower_level_trees)
                    for node in range(total_nodes_in_this_level):
                        if len(lower_level_trees) >= 2:
                            # name the cluster root different
                            tree_name = ''
                            if level + 1 == height_of_tree:
                                tree_name = root_cluster_key
                            else:
                                tree_name = lower_level_trees[0].get_root_id() + "_" + lower_level_trees[
                                    1].get_root_id()

                            temp_tree = BinaryTree(tree_name)
                            temp_data = []
                            temp_data.extend(lower_level_trees[0].get_data())
                            temp_data.extend(lower_level_trees[1].get_data())
                            temp_tree.set_data(temp_data)

                            temp_tree.set_left_child(lower_level_trees[0])
                            temp_tree.set_right_child(lower_level_trees[1])

                            # set the parents of lower level trees
                            lower_level_trees[0].set_parent(temp_tree)
                            lower_level_trees[1].set_parent(temp_tree)

                            lower_level_trees.pop(0)
                            lower_level_trees.pop(0)

                            # create a cluster out of the tree
                            if level + 1 != height_of_tree:
                                # no cluster for the root cluster
                                cluster_graph = graph.subgraph([str(s) for s in temp_tree.get_data()])
                                cluster = Cluster(tree_name, cluster_graph, i)
                                cluster.set_intermediate()
                                self.__network.add_cluster(i, cluster)

                            new_lower_level_trees.append(temp_tree)

                            if level + 1 == height_of_tree:
                                current_tree = temp_tree
                                print_tree(temp_tree)

                        elif len(lower_level_trees) == 1:
                            if level + 1 == height_of_tree:
                                tree_name = ''
                                if level + 1 == height_of_tree:
                                    tree_name = root_cluster_key
                                else:
                                    tree_name = lower_level_trees[0].get_root_id()

                                temp_tree = BinaryTree(tree_name)
                                temp_tree.set_data(lower_level_trees[0].get_data())
                                temp_tree.set_left_child(lower_level_trees[0])

                                # set the parents of lower level trees
                                # set parent only if not same node in lower level
                                lower_level_trees[0].set_parent(temp_tree)
                                lower_level_trees.pop(0)

                                # create a cluster out of the tree
                                if level + 1 != height_of_tree:
                                    # no cluster for the root cluster
                                    cluster_graph = graph.subgraph([str(s) for s in temp_tree.get_data()])
                                    cluster = Cluster(tree_name, cluster_graph, i)
                                    cluster.set_intermediate()
                                    self.__network.add_cluster(i, cluster)

                                new_lower_level_trees.append(temp_tree)
                                if level + 1 == height_of_tree:
                                    current_tree = temp_tree

                            else:
                                new_lower_level_trees.append(lower_level_trees.pop(0))

                    lower_level_trees = new_lower_level_trees

                logger.debug("TREE FOR CLUSTER {} IS ".format(root_cluster_key))
                (self.__network.find_cluster_by_id(root_cluster_key)).set_tree(current_tree)
                # set the tree children here, traversing starts from the children
                print_tree(current_tree)
                (self.__network.find_cluster_by_id(root_cluster_key)).set_tree_leaves(get_leaves(current_tree))

            for clust in self.__network.clusters_by_level(i):
                clust.print()
                # clus.tree.display_tree()

        logger.debug("UNCONTAINED CLUSTERS")
        for cluster in uncontained_clusters:
            cluster.print()
        assert len(uncontained_clusters) == 0

        # def _sort_clusters_in_a_level(self):
        #     for i in (range(0, self._height_of_cluster + 1)):
        #         index = 0
        #         for clus in self._network.clusters_by_level(i):
        #             clus.setIndex(index)
        #             index = index+1
        #     print("PRINTING CLUSTERS")
        #     for i in (range(0, self._height_of_cluster + 1)):
        #         for clus in self._network.clusters_by_level(i):
        #             clus.print()

        logger.debug("HEIGHT OF CLUSTERS {}".format(self.__height_of_cluster))
        for i in range(0, self.__height_of_cluster + 1):
            logger.debug("AT LEVEL {} THERE ARE {} CLUSTERS".format(i, len(self.__network.clusters_by_level(i))))
            self.__network.print_cluster_by_level(i)

    # make lower level clusters connected to the upper level clusters
    # the lower level will be connected to the tree nodes of upper level cluster

    def save_network(self, output_file):
        with open(output_file, 'wb') as output:
            pickle.dump(self.__network, output, pickle.HIGHEST_PROTOCOL)

    def load_network(self, input_file):
        with open(input_file, 'rb') as input_f:
            self.__network = pickle.load(input_f)

    def run(self, rounds):
        start = time.time()

        # simulations to run
        # initial publisher and object
        # r.seed(rnd)
        for rnd in range(0, rounds):
            r = random.Random()
            r.seed(rnd)
            random_id = str(r.randint(0, self.__network.size() - 1))
            publisher = self.__network.find_cluster_by_id('c' + random_id + '_l0')
            obj = Object('obj1', publisher.get_cluster_id())

            self.__network.publish(publisher, obj)
            results = []

            # r.seed("test")

            optimal_processing_load = OrderedDict()
            for i in range(0, self.__peer_count):
                optimal_processing_load[str(i)] = 0

            processing_load = OrderedDict()
            for i in range(0, self.__peer_count):
                processing_load[str(i)] = 0

            for i in range(1, self.__op_count + 1):
                logger.info("OPERATION {}".format(i))
                logger.info("{}".format(obj.get_owner()))
                owner = self.__network.find_cluster_by_id(obj.get_owner())
                # select random peer to create a move request except the owner peer.
                random_id = str(r.randint(0, self.__network.size() - 1))
                while 'c' + random_id + '_l0' == obj.get_owner():
                    random_id = str(r.randint(0, self.__network.size() - 1))

                logger.info('c{}_l0'.format(random_id))
                mover = self.__network.find_cluster_by_id('c' + random_id + '_l0')
                logger.info("{}".format(mover.get_cluster_id()))
                assert ('c' + random_id + '_l0' != obj.get_owner())
                logger.info("MOVING FROM {} to {}".format(owner.get_cluster_id(), mover.get_cluster_id()))
                logger.info("RANDOM ID FOR MOVER IS {}".format(random_id))
                logger.info("SIZE OF NETWORK IS {}".format(self.__network.size()))
                res = self.__network.move(mover, obj)
                res['msg'] = "MOVING FROM {} TO {}".format(str(owner.get_cluster_id()), str(mover.get_cluster_id()))
                res['mover_id'] = str(mover.get_id())

                # optimal processing load calculation
                res['optimal_cost'] = nx.dijkstra_path_length(self.__network.get_graph(), mover.get_id(),
                                                              owner.get_id(), 'weight')
                optimal_processing_load[mover.get_id()] = optimal_processing_load[mover.get_id()] + 1

                res['stretch'] = res['total_cost'] / res['optimal_cost']

                logger.info("COST IS {}".format(res['total_cost']))
                logger.info("Optimal cost is {}".format(res['optimal_cost']))
                logger.info("stretch is {}".format(res['stretch']))
                logger.info(obj.get_owner())

                for i in range(0, self.__peer_count):
                    processing_load[str(i)] = processing_load[str(i)] + res['processing_load'][str(i)]

                results.append(res)

            output = dict()

            logger.info("{}".format(results))

            # prepare the result
            total_cost = 0
            total_cost_optimal = 0

            total_stretch = 0

            total_hops = 0
            total_t_hops = 0

            for result in results:
                total_cost_optimal = total_cost_optimal + result['optimal_cost']
                total_cost = total_cost + result['total_cost']
                total_stretch = total_stretch + result['stretch']
                total_hops = total_hops + result['hops']
                total_t_hops = total_t_hops + result['t_hops']

            # output = dict()
            # output['results'] = results

            output['COST_OPTIMAL'] = total_cost_optimal
            output['COST'] = total_cost

            output['STRETCH'] = total_stretch

            output['PROCESSING_LOAD_OPTIMAL'] = optimal_processing_load
            output['PROCESSING_LOAD'] = processing_load

            output['HOPS'] = total_hops
            output['TREE_HOPS'] = total_t_hops
            path = 'results'

            if not os.path.exists(path):
                os.makedirs(path)

            file = open(
                os.path.join(path, 'SPIRAL_O' + str(self.__op_count) + '_P' + str(self.__peer_count) + '_R' + str(
                    rnd) + '.json'), 'w'
            )

            json.dump(output, file)
            file.close()

        end = time.time()
        logger.debug("TIME ELAPSED: {}".format(end - start))
