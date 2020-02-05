import math

from network import Network
from binaryTree import BinaryTree
from object import Object

from peer import Peer
from cluster import Cluster
from collections import Counter, OrderedDict
import sys
from binaryTree import printTree, get_leafs

import matplotlib.pyplot as plt
from networkx import nx, json_graph
import copy
import random
import json
import os
import pickle


class Simulation:
    # __conf = {
    #     "operationCount": self._op_count,
    #     "peerCount": self._peer_count,
    #
    # }
    #
    # @staticmethod
    # def config(name):
    #     return Simulation.__conf[name]
    #
    def __init__(self, operationCount, peerCount, graphInput):
        print("Inside Simulation init")

        self._op_count = int(operationCount)
        self._peer_count = int(peerCount)

        plt.axis('off')

        graph = nx.read_weighted_edgelist(graphInput)

        print(graph)

        # some properties
        print("node degree clustering")
        for v in nx.nodes(graph):
            print('%s %d %f' % (v, nx.degree(graph, v), nx.clustering(graph, v)))

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
        self._diameter = nx.diameter(graph, e=ecc)

        print('The graph diameter is ', self._diameter)
        self._height_of_cluster = math.ceil(math.log(self._diameter, 2)) + 1

        if not os.path.exists(graphInput+'_network'):
            print("CREATING NEW NETWORK")
            self._network = Network(graph)
            self._setup_peers(graph)
            # self._build_clusters_exactly_logn_membership(graph)
            self._build_clusters_at_least_one_and_at_most_logn(graph)
            # self._build_clusters_no_overlapping(graph)
            self._build_tree(graph)
            self.save_network(graphInput + '_network')
            exit(0)

        else:
            print("LOADING NETWORK FROM INPUT FILE")
            #     load network
            self.load_network(graphInput + '_network')

    def _setup_peers(self, graph):

        for n in graph.nodes():
            print('Node id is ', n)
            print('The neighbors are')
            print(graph.edges(n))
            for e in graph.edges(n):
                print(graph[e[0]][e[1]]['weight'])
                neighbors = {}
                for e in graph.edges(n):
                    neighbors[int(e[1])] = graph[e[0]][e[1]]['weight']

            print(neighbors)
            peer = Peer(int(n), neighbors)
            self._network.add_peer(peer_id=int(n), peer=peer)

    def at_least_one(self, clustered_peers_count):
        for i in range(self._peer_count):
            if clustered_peers_count[str(i)] < 1:
                print("At least one False because of this ", str(i))
                return i
        return -1

    def at_most_logn(self, clustered_peers_count):
        for i in range(self._peer_count):
            if clustered_peers_count[str(i)] > math.log(self._peer_count, 2):
                print("At most log n False because of this ", str(i))
                return i
        return -1

    def _build_clusters_no_overlapping(self, graph):
        print(nx.dijkstra_path(graph, '0', '7', 'weight'))
        print(nx.dijkstra_path_length(graph, '0', '7', 'weight'))
        paths_for_diameter = nx.shortest_path_length(graph, weight='weight')
        for path in nx.shortest_path_length(graph, weight='weight'):
            print(path)

        ecc = nx.eccentricity(graph, sp=dict(paths_for_diameter))
        # ecc = nx.eccentricity(graph, sp=shortest_paths_for_diameter)
        diameter = nx.diameter(graph, e=ecc)
        # for path in paths_for_diameter:
        #     print(path)

        print('The graph diameter is ', diameter)
        height_of_cluster = math.ceil(math.log(diameter, 2)) + 1
        print('height_of_the hierarchy is ', height_of_cluster)
        # lowest level clusters
        print('lowest level clusters')
        for n in graph.nodes():
            paths = nx.single_source_dijkstra_path_length(graph, n, 0, weight='weight')
            print(paths)
            cluster_graph = graph.subgraph([n])
            cluster = Cluster('c' + str(n) + '_l' + '0', cluster_graph, 0)
            self._network.add_cluster_level(0)

            self._network.add_cluster(0, cluster)
            self._network.draw_cluster(cluster.cluster_id)

        for i in range(int(height_of_cluster)):
            self._network.add_cluster_level(i+1)

            print('AT LEVEL ------- ', i + 1)
            distance = pow(2, i + 1)

            print('THE DISTANCE LIMIT IS ', distance)
            clustered_peers = []
            for n in range(self._peer_count):
                # for n in graph.nodes():
                print('clustering peer ', n)
                if n in clustered_peers:
                    print('peer ', n, ' already clustered')
                    continue
                paths_found = nx.single_source_dijkstra_path_length(graph, str(n), distance, weight='weight')
                peers_to_cluster = []
                print('paths found in the level ', paths_found)
                for peer in paths_found:

                    if int(peer) in clustered_peers:
                        print('peer already clustered in another cluster')
                    else:
                        clustered_peers.append(int(peer))
                        peers_to_cluster.append(int(peer))

                cluster_graph = graph.subgraph([str(i) for i in peers_to_cluster])

                # cluster = Cluster('c' + str(n) + '_l' + str(i + 1), peers_to_cluster)
                cluster = Cluster('c' + str(n) + '_l' + str(i + 1), cluster_graph, i + 1)
                self._network.add_cluster(i + 1, cluster)
                # self._network.draw_cluster(cluster.cluster_id)

    def _build_clusters_exactly_logn_membership(self, graph):
        paths_for_diameter = nx.shortest_path_length(graph, weight='weight')
        for path in nx.shortest_path_length(graph, weight='weight'):
            print(path)

        ecc = nx.eccentricity(graph, sp=dict(paths_for_diameter))
        # ecc = nx.eccentricity(graph, sp=shortest_paths_for_diameter)
        diameter = nx.diameter(graph, e=ecc)
        # for path in paths_for_diameter:
        #     print(path)

        print('The graph diameter is ', diameter)
        height_of_cluster = math.ceil(math.log(diameter, 2)) + 1
        print('height_of_the hierarchy is ', height_of_cluster)
        # lowest level clusters
        print('lowest level clusters')
        self._network.add_cluster_level(0)

        for n in graph.nodes():
            paths = nx.single_source_dijkstra_path_length(graph, n, 0, weight='weight')
            print(paths)
            cluster_graph = graph.subgraph([n])
            cluster = Cluster('c' + str(n) + '_l' + '0', cluster_graph, 0)
            self._network.add_cluster(0, cluster)
            # self._network.draw_cluster(cluster.cluster_id)

        for i in range(int(height_of_cluster)):
            self._network.add_cluster_level(i + 1)
            clustered_peers_list = []
            cluster_ids_list = []

            print('AT LEVEL ------- ', i + 1)
            distance = pow(2, i + 1)

            print('THE DISTANCE LIMIT IS ', distance)
            clustered_peers = []
            # for naming the clusters properly
            n = 0
            while n < self._peer_count:
                incomplete = False
                # for n in graph.nodes():
                print('clustering peer ', n)

                paths_found = nx.single_source_dijkstra_path_length(graph, str(n), distance, weight='weight')
                peers_to_cluster = []
                print('paths found in the level ', paths_found)
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
                self._network.add_cluster(i + 1, cluster)
                # self._network.draw_cluster(cluster.cluster_id)
                clustered_peers_count = Counter(clustered_peers_list)
                # print(self._network._clusters)
                for j in range(self._peer_count):
                    if clustered_peers_count[str(j)] < math.log(self._peer_count, 2):
                        incomplete = True

                if not incomplete:
                    # make peers fall in exactly log(n) clusters
                    for peer in range(self._peer_count):
                        # count of peers in all clusters
                        peer_count = clustered_peers_count[str(peer)]
                        while peer_count > math.log(self._peer_count, 2):
                            for cluster in reversed(self._network.clusters_by_level(i + 1)):
                                if clustered_peers_count[str(peer)] > math.log(self._peer_count, 2) and cluster.graph.has_node(
                                        str(peer)):
                                    cluster.graph.remove_node(str(peer))

                                    peer_count = peer_count - 1
                                    if str(peer) in clustered_peers_list:
                                        clustered_peers_list.remove(str(peer))
                                    clustered_peers_count = Counter(clustered_peers_list)

                                    break

                    break

                n += 1
                if n == self._peer_count:
                    n = 0

        # delete empty clusters
        for i in range(int(height_of_cluster)):
            for cluster in self._network.clusters_by_level(i):
                if nx.number_of_nodes(cluster.graph) == 0:
                    self._network.remove_cluster_by_id(cluster.cluster_id, cluster.level)

        for i in range(int(height_of_cluster)):
            for cluster in self._network.clusters_by_level(i):
                assert (nx.number_of_nodes(cluster.graph) is not 0)

        # make sure that there are log(n) peers in every level
        for i in range(int(height_of_cluster) - 1):
            print("Verifying in level ", i + 1)
            clustered_peer_list = []
            for cluster in self._network.clusters_by_level(i + 1):
                for node in cluster.graph.nodes:
                    clustered_peer_list.append(str(node))
            clustered_peer_count = Counter(clustered_peer_list)
            for cl in clustered_peer_count:
                assert (clustered_peer_count[cl] == math.log(self._peer_count, 2))

        # for i in range(int(height_of_cluster)):
        #     for cluster in self._network.clusters_by_level(i):
        #         self._network.draw_cluster(cluster.cluster_id)

    # def _build_clusters_at_least_one_and_at_most_logn(self, graph):
    #     paths_for_diameter = nx.shortest_path_length(graph, weight='weight')
    #     for path in nx.shortest_path_length(graph, weight='weight'):
    #         print(path)
    #
    #     ecc = nx.eccentricity(graph, sp=dict(paths_for_diameter))
    #     diameter = nx.diameter(graph, e=ecc)
    #
    #     print('The graph diameter is ', diameter)
    #     height_of_cluster = math.ceil(math.log(diameter, 2))
    #     print('height_of_the hierarchy is ', height_of_cluster)
    #     # lowest level clusters
    #     print('lowest level clusters')
    #     self._network.add_cluster_level(0)
    #
    #     # level 0 clusters
    #     for n in graph.nodes():
    #         paths = nx.single_source_dijkstra_path_length(graph, n, 0, weight='weight')
    #         print(paths)
    #         cluster_graph = graph.subgraph([n])
    #         cluster = Cluster('c' + str(n) + '_l' + '0', cluster_graph, 0)
    #         self._network.add_cluster(0, cluster)
    #         # self._network.draw_cluster(cluster.cluster_id)
    #
    #     # upper level clusters
    #     for i in range(int(height_of_cluster)):
    #         self._network.add_cluster_level(i + 1)
    #         clustered_peers_list = []
    #
    #         # for naming the clusters properly
    #         cluster_ids_list = []
    #
    #         print('AT LEVEL ------- ', i + 1)
    #         distance = pow(2, i + 1)
    #         print('THE DISTANCE LIMIT IS ', distance)
    #         clusterize = {}
    #
    #         n = 0
    #         # iterate over the peers
    #         while n < self._peer_count:
    #             print('clustering peer ', n)
    #
    #             paths_found = nx.single_source_dijkstra_path_length(graph, str(n), distance, weight='weight')
    #             peers_to_cluster = []
    #             print('paths found in the level ', paths_found)
    #
    #             for peer in paths_found:
    #                 peers_to_cluster.append(peer)
    #                 clustered_peers_list.append(peer)
    #
    #             c_id = 'c' + str(n) + '_l' + str(i + 1)
    #
    #             # for naming the clusters properly
    #             cluster_ids_list.append(c_id)
    #             cluster_ids_count = Counter(cluster_ids_list)
    #
    #             c_id = c_id + "_" + str(cluster_ids_count[c_id])
    #
    #             clusterize[c_id] = peers_to_cluster
    #
    #             # cluster = Cluster(c_id, cluster_graph, i + 1)
    #             # self._network.add_cluster(i + 1, cluster)
    #             # self._network.draw_cluster(cluster.cluster_id)
    #
    #             clustered_peers_count = Counter(clustered_peers_list)
    #             print("CLUSTERED PEERS COUNT", clustered_peers_count)
    #             if self.at_least_one(clustered_peers_count) < 0:
    #                 cluster_indices_to_ignore_all = []
    #                 for k in range(self._peer_count):
    #                     print("checking for ", k )
    #                     if clustered_peers_count[str(k)] > math.log(self._peer_count, 2):
    #                         # at most logn should remove.
    #                         print(str(k), "thatis in ", clustered_peers_count[str(k)], " clusters")
    #
    #                         #         find a cluster with this peer
    #                         deleted = False
    #                         cluster_indices_to_ignore_peerwise = []
    #
    #                         while not deleted:
    #                             #     find the cluster with minimum membership with this peer
    #                             min_cluster_size = self._peer_count
    #                             min_cluster_index = 0
    #                             for inner_key in list(clusterize):
    #                                 assert (clustered_peers_count[str(k)]) >= len(cluster_indices_to_ignore_peerwise)
    #                                 if inner_key not in cluster_indices_to_ignore_peerwise:
    #                                     if str(k) in clusterize[inner_key]:
    #                                         if len(clusterize[inner_key]) < min_cluster_size:
    #                                             min_cluster_size = len(clusterize[inner_key])
    #                                             min_cluster_index = inner_key
    #
    #                             # The cluster which contains the minimum membership is
    #                             print("THE CLUSTER WITH MIN MEMBERSHIP IS ", min_cluster_index,
    #                                   clusterize[min_cluster_index])
    #
    #                             # delete the cluster if removing it maintains the first property
    #                             # form a temporary clustered peers list
    #                             temp_clustered_peers_list = []
    #                             for check_index in clusterize:
    #                                 if check_index is not min_cluster_index:
    #                                     for item in clusterize[check_index]:
    #                                         temp_clustered_peers_list.append(item)
    #                             temp_clustered_peers_count = Counter(temp_clustered_peers_list)
    #
    #                             at_least_one_check = self.at_least_one(temp_clustered_peers_count)
    #                             if at_least_one_check < 0:
    #                                 print("THE PEER IS NOT GOOD FOR REMOVAL")
    #                                 delete = [key for key in clusterize if key == min_cluster_index]
    #                                 for key in delete:
    #                                     for peer_id in clusterize[key]:
    #                                         clustered_peers_list.remove(peer_id)
    #                                     del clusterize[key]
    #                                 print("Deleted")
    #                                 break;
    #
    #                             else:
    #                                 print("THIS PEER NOT GOOD AFTER REMOVAL ", at_least_one_check)
    #                                 cluster_indices_to_ignore_peerwise.append(min_cluster_index)
    #
    #                             # delete the cluster
    #                             temp_clusterize = {}
    #                         print ("DONE FOR PEER ", k)
    #                         cluster_indices_to_ignore_all.extend(cluster_indices_to_ignore_peerwise)
    #                 #     update the clustered_peers_count
    #                     clustered_peers_count = Counter(clustered_peers_list)
    #
    #                 print("FORMING THE CLUSTERS AND ADDING THEM TO THE NETWORK")
    #                 assert(self.at_least_one(clustered_peers_count) < 0)
    #                 assert(self.at_most_logn(clustered_peers_count) < 0)
    #                 #   avoid clusters with same elements in clusterize;
    #
    #                 new_clusturize = {}
    #                 temp_peers_list = []
    #                 new_temp_peers_list = []
    #                 print("BEFORE ", clusterize)
    #                 for key in clusterize:
    #                     for item in clusterize[key]:
    #                         temp_peers_list.append(item)
    #                     duplicate_peers = False
    #                     print(len(new_clusturize))
    #                     for new_key in new_clusturize:
    #                         for new_item in new_clusturize[new_key]:
    #                             new_temp_peers_list.append(new_item)
    #
    #                         print("OHHHH")
    #                         print(Counter(temp_peers_list))
    #                         print(Counter(new_temp_peers_list))
    #                         if Counter(new_temp_peers_list) == Counter(temp_peers_list):
    #                             duplicate_peers = True
    #                         new_temp_peers_list = []
    #                     if not duplicate_peers:
    #                         new_clusturize[key] = clusterize[key]
    #                     temp_peers_list = []
    #
    #                 print("AFTER ", new_clusturize)
    #                 for key in new_clusturize:
    #                     cluster_graph = graph.subgraph([str(i) for i in new_clusturize[key]])
    #                     cluster = Cluster(key, cluster_graph, i + 1)
    #                     self._network.add_cluster(i + 1, cluster)
    #                 # done with this level
    #                 break
    #             n += 1
    #             if n == self._peer_count:
    #                 n = 0
    #
    #     print (new_clusturize)
    #     # delete empty clusters
    #     for i in range(int(height_of_cluster)):
    #         for cluster in self._network.clusters_by_level(i):
    #             if nx.number_of_nodes(cluster.graph) == 0:
    #                 self._network.remove_cluster_by_id(cluster.cluster_id, cluster.level)
    #
    #     for i in range(int(height_of_cluster)):
    #         for cluster in self._network.clusters_by_level(i):
    #             assert (nx.number_of_nodes(cluster.graph) is not 0)
    #
    #     # make sure that there are log(n) peers in every level
    #     for i in range(int(height_of_cluster) - 1):
    #         print("Verifying in level ", i + 1)
    #         clustered_peer_list = []
    #         for cluster in self._network.clusters_by_level(i + 1):
    #             for node in cluster.graph.nodes:
    #                 clustered_peer_list.append(str(node))
    #         clustered_peer_count = Counter(clustered_peer_list)
    #         for cl in clustered_peer_count:
    #             assert (clustered_peer_count[cl] <= math.log(self._peer_count, 2))
    #
    #     # for i in range(int(height_of_cluster) + 1):
    #     #     for cluster in self._network.clusters_by_level(i):
    #     #         self._network.draw_cluster(cluster.cluster_id)

    def _build_clusters_at_least_one_and_at_most_logn(self, graph):
        paths_for_diameter = nx.shortest_path_length(graph, weight='weight')
        for path in nx.shortest_path_length(graph, weight='weight'):
            print(path)

        ecc = nx.eccentricity(graph, sp=dict(paths_for_diameter))
        diameter = nx.diameter(graph, e=ecc)

        print('The graph diameter is ', diameter)
        height_of_cluster = math.ceil(math.log(diameter, 2)) + 1
        self._network.set_height_of_clusters(height_of_cluster)
        print('height_of_the hierarchy is ', height_of_cluster)
        # lowest level clusters
        print('lowest level clusters')
        self._network.add_cluster_level(0)

        # level 0 clusters
        for n in graph.nodes():
            paths = nx.single_source_dijkstra_path_length(graph, n, 0, weight='weight')
            print(paths)
            cluster_graph = graph.subgraph([n])
            cluster = Cluster('c' + str(n) + '_l' + '0', cluster_graph, 0, str(n))
            self._network.add_cluster(0, cluster)
            # self._network.draw_cluster(cluster.cluster_id)


        # form upper level clusters
        for i in range(int(height_of_cluster)):
            self._network.add_cluster_level(i + 1)
            clustered_peers_list = []
        #     for naming the cluster properly
            cluster_ids_list = []

            print ('AT LEVEL ----- ', i + 1)
            distance = pow(2, i + 1)
            print('THE DISTANCE LIMIT IS ', distance)
            clusterize = {}

            n = 0
        #     iterate over the peers once
            while n < self._peer_count:
                print('clustering peer ', n)

                paths_found = nx.single_source_dijkstra_path_length(graph, str(n), distance, weight='weight')

                peers_to_cluster = []
                print('paths found in the level ', paths_found)
                tmp_peers_list_to_cluster = []
                for peer in paths_found:
                    peers_to_cluster.append(peer)
                    # clustered_peers_list.append(peer)
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
                            print ("FOUND DUPLICATE CLUSTER")
                            duplicate = True
                            break
                    print("DOESN'T VIOLATE AT MOST LOG N")
                    if not duplicate:
                        clustered_peers_list.extend(tmp_peers_list_to_cluster)
                        clusterize[c_id] = peers_to_cluster

                n += 1

            print("CHECKING MEMBERSHIP")
            print(self.at_least_one(Counter(clustered_peers_list)) < 0)
            print(self.at_most_logn(Counter(clustered_peers_list)) < 0)
            assert (self.at_most_logn(Counter(clustered_peers_list)) < 0)

            # if not built yet build a cluster and remove peers who appear more than logn times
            missing_cluster_id = self.at_least_one(Counter(clustered_peers_list))
            while missing_cluster_id > -1:
                print("PEER ", missing_cluster_id, " IS MISSING, BUILDING A CLUSTER AROUND IT.")
                paths_found = nx.single_source_dijkstra_path_length(graph, str(missing_cluster_id), distance, weight='weight')
                peers_to_cluster = []
                print('paths found in the level ', paths_found)
                tmp_peers_list_to_cluster = []
                for peer in paths_found:
                    peers_to_cluster.append(peer)
                    # clustered_peers_list.append(peer)
                    tmp_peers_list_to_cluster.append(peer)
                # todo creating single node cluster
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
                            print("FOUND DUPLICATE CLUSTER")
                            duplicate = True
                            break
                    print("DOESN'T VIOLATE AT MOST LOG N")
                    if not duplicate:
                        print("WHY HERE")
                        clustered_peers_list.extend(tmp_peers_list_to_cluster)
                        clusterize[c_id] = tmp_peers_list_to_cluster
                else:
                #     find the peer that appears more than logn
                    excess_cluster_id = self.at_most_logn(Counter(temp_clustered_peers_list))

                    while excess_cluster_id != -1:
                        print(tmp_peers_list_to_cluster)
                        print(excess_cluster_id)
                        tmp_peers_list_to_cluster.remove(str(excess_cluster_id))
                        tmp_clustered_peers_list = copy.deepcopy(clustered_peers_list)
                        tmp_clustered_peers_list.extend(tmp_peers_list_to_cluster)
                        # removing
                        print(excess_cluster_id, " APPEARS IN MORE THAN LOGN CLUSTER, REMOVING IT")
                        excess_cluster_id = self.at_most_logn(Counter(tmp_clustered_peers_list))
                    print("PREPARING THE CLUSTER FROM MODIFIED PEERS LIST ", tmp_peers_list_to_cluster)
                    clustered_peers_list.extend(tmp_peers_list_to_cluster)
                    clusterize[c_id] = tmp_peers_list_to_cluster
                    assert (self.at_most_logn(Counter(clustered_peers_list)) < 0)

                missing_cluster_id = self.at_least_one(Counter(clustered_peers_list))

            print("ASSERTING")
            assert (self.at_least_one(Counter(clustered_peers_list)) < 0)
            assert (self.at_most_logn(Counter(clustered_peers_list)) < 0)

            print(clusterize)
            print(Counter(clustered_peers_list))

            print("FINALLY ADDING CLUSTERS")
            for key in clusterize:
                print (len(clusterize[key]))
                print(nx.number_of_nodes(graph.subgraph([str(i) for i in clusterize[key]])))
                assert(nx.number_of_nodes(graph.subgraph([str(i) for i in clusterize[key]])) == len(clusterize[key]))
                cluster_graph = graph.subgraph([str(i) for i in clusterize[key]])
                cluster = Cluster(key, cluster_graph, i + 1)
                if i + 1 == height_of_cluster:
                    cluster.root = True
                    self._network._root_cluster = cluster
                self._network.add_cluster(i + 1, cluster)
                # self._network.draw_cluster(cluster.cluster_id)

            print("CLUSTERIZE ", clusterize)

        return

    def _build_tree(self, graph):
        print("Building the tree")
        # for clusters in each upper level besides 0
        print ("The height of clusters is ", self._height_of_cluster)

        uncontained_clusters = []
        for i in (range(2, self._height_of_cluster+1)):
        # for i in (range(2, 3)):
            # for each cluster in this level
            print (i)
            # print("At level ", 3, " there are ", len(self._network.clusters_by_level(3)), " clusters.")
            # self._network.print_cluster_by_level(3)

            # print("At level ", 4, " there are ", len(self._network.clusters_by_level(4)), " clusters.")
            # self._network.print_cluster_by_level(4)

            # print("At level ", 5, " there are ", len(self._network.clusters_by_level(5)), " clusters.")
            # self._network.print_cluster_by_level(5)
            # at first, gather bottom level clusters which are contained in the upper level
            cluster_contain_in_upper_level_map = {}
            # list of clusters that were added to some other parent level clusters
            ignore_cluster_list = []

            for current_cluster in self._network.clusters_by_level(i):

                # just checking
                clustered_peers_list = []
                print("Just checking Assert redundant clusters BEFORE")
                for cluster in self._network.clusters_by_level(i):
                    for node in cluster.graph.nodes:
                        clustered_peers_list.append(node)
                print(clustered_peers_list)
                assert (self.at_least_one(Counter(clustered_peers_list)) < 0)
                assert (self.at_most_logn(Counter(clustered_peers_list)) < 0)


                # ignore intermediate clusters

                print(current_cluster.cluster_id)
                cluster_contain_in_upper_level_map[current_cluster.cluster_id] = []
                skip_cluster = False
                lower_level_clusters = self._network.clusters_by_level(i - 1)
                print(len(lower_level_clusters))
                print(len(uncontained_clusters))

                for uncontained_cluster in uncontained_clusters:
                    lower_level_clusters.append(uncontained_cluster)
                print(len(lower_level_clusters))
                print("LOWER LEVEL CLUSTERS",lower_level_clusters)
                # for lower_level_cluster in self._network.clusters_by_level(i - 1):
                for lower_level_cluster in lower_level_clusters:
                    if lower_level_cluster.intermediate:
                        print(current_cluster.cluster_id)
                        print(current_cluster.peers)
                    else:
                        if lower_level_cluster.cluster_id not in ignore_cluster_list:
                            # check to see for duplicates
                            for key in cluster_contain_in_upper_level_map[current_cluster.cluster_id]:
                                # print(Counter(self._network.find_cluster_by_id(key).graph.nodes) == Counter(lower_level_cluster.graph.nodes))
                                if Counter(self._network.find_cluster_by_id(key).graph.nodes) == Counter(
                                        lower_level_cluster.graph.nodes):
                                    print("DUPLICATE FOUND")
                                    skip_cluster = True
                                    continue

                                for cluster_to_ignore in ignore_cluster_list:
                                    # print("CLUSTER TO IGNORE ID")
                                    # print(cluster_to_ignore)
                                    # print(lower_level_cluster.cluster_id)
                                    if Counter(lower_level_cluster.graph.nodes) == Counter(
                                            self._network.find_cluster_by_id(cluster_to_ignore).graph.nodes):
                                        print("DUPLICATE ENTRIES FOUND")
                                        skip_cluster = True
                                        continue

                            # check if same members

                            if skip_cluster == True:
                                continue
                            # print("Current cluster id ", current_cluster.cluster_id, " with nodes ", current_cluster.graph.nodes)
                            # print("Lower level cluster id ", lower_level_cluster.cluster_id, " with nodes ", lower_level_cluster.graph.nodes)
                            result = set(lower_level_cluster.graph.nodes).issubset(current_cluster.graph.nodes)
                            # print("CONTAIN ", result)

                            # print("THE MAP IS ", cluster_contain_in_upper_level_map)
                            if result:
                                cluster_contain_in_upper_level_map[current_cluster.cluster_id].append(
                                    lower_level_cluster.cluster_id)
                                ignore_cluster_list.append(lower_level_cluster.cluster_id)

            print("SIZE LOWER LEVEL CLUSTERS ", len(self._network.clusters_by_level(i-1)))
            print("IGNORE CLUSTER LIST SIZE",len(ignore_cluster_list))
            print (ignore_cluster_list)

            # delete all uncontained_clusters that are in ignore_cluster_list
            temp_uncontained_clusters = copy.deepcopy(uncontained_clusters)
            print("BEFORE",uncontained_clusters)
            print(len(uncontained_clusters))
            for cluster in uncontained_clusters:
                print("PRINTING A CLUSTER")
                print(cluster.cluster_id)

            print(len(uncontained_clusters))
            for cluster in uncontained_clusters:
                print("CHECKING CHECKING")
                print(cluster.cluster_id)
                print(ignore_cluster_list)
                print(cluster.cluster_id in ignore_cluster_list)
                if cluster.cluster_id in ignore_cluster_list:
                    print("FOUND CLUSTER", cluster.cluster_id)
                    print("REMOVING")
                    index_to_delete = None
                    for index in range(0,len(temp_uncontained_clusters)):
                        if temp_uncontained_clusters[index].cluster_id == cluster.cluster_id:
                            index_to_delete = index
                            print("index_TO_DELETE IS ", index_to_delete)
                            break
                    del(temp_uncontained_clusters[index_to_delete])
                    # temp_uncontained_clusters.remove(cluster)
            uncontained_clusters = copy.deepcopy(temp_uncontained_clusters)
            print("AFTERE",uncontained_clusters)
            for cluster in uncontained_clusters:
                print(cluster.cluster_id)

            for lower_level_cluster in self._network.clusters_by_level(i-1):
                if lower_level_cluster.cluster_id not in ignore_cluster_list:
                    if not lower_level_cluster.intermediate:
                        uncontained_clusters.append(lower_level_cluster)


            print("PREPARED CLUSTER CONTAIN MAP", cluster_contain_in_upper_level_map)

            # build the binary trees
            current_tree = BinaryTree('temp')

            for root_cluster_key in cluster_contain_in_upper_level_map:

                tree_leafs = []
                current_tree = BinaryTree(root_cluster_key)
                print("ROOT CLUSTER IS ", root_cluster_key)
                print (len(cluster_contain_in_upper_level_map[root_cluster_key]))
                height_of_tree = int(math.log(1 if len(cluster_contain_in_upper_level_map[root_cluster_key]) == 0 else 2**(len(cluster_contain_in_upper_level_map[root_cluster_key]) - 1).bit_length(), 2) )


                if len(cluster_contain_in_upper_level_map[root_cluster_key])== 1:
                    height_of_tree = 1
                print("HEIGHT OF TREE IS ", height_of_tree + 1)

                tree_nodes = cluster_contain_in_upper_level_map[root_cluster_key]

                lower_level_trees = []
                for bottommmost_level_cluster in cluster_contain_in_upper_level_map[root_cluster_key]:
                    print("REACHED and count is ")
                    print(bottommmost_level_cluster)
                    temp_tree = BinaryTree(bottommmost_level_cluster)

                    temp_tree.set_data(self._network.find_cluster_by_id(bottommmost_level_cluster).peers)
                    lower_level_trees.append(temp_tree)

                for level in range(height_of_tree):
                # for level in range(1):
                    new_lower_level_trees = []

                    print ("LEVEL ",level, "of the tree")
                    print("NEW LEVEL NODE COUNT ", len(lower_level_trees))

                    total_nodes_in_this_level = len(lower_level_trees)
                    for node in range(total_nodes_in_this_level):
                        if len(lower_level_trees) >= 2:
                            # name the cluster root different
                            tree_name = ''
                            if level + 1 == height_of_tree:
                                tree_name = root_cluster_key
                            else:
                                tree_name = lower_level_trees[0].getNodeValue() + "_" + lower_level_trees[1].getNodeValue()
                            temp_tree = BinaryTree(tree_name)
                            temp_data = []
                            temp_data.extend(lower_level_trees[0].get_data())
                            temp_data.extend(lower_level_trees[1].get_data())
                            temp_tree.set_data(temp_data)

                            temp_tree.left = lower_level_trees[0]
                            temp_tree.right = lower_level_trees[1]

                            # set the parents of lower level trees

                            lower_level_trees[0].set_parent(temp_tree)
                            lower_level_trees[1].set_parent(temp_tree)

                            lower_level_trees.pop(0)
                            lower_level_trees.pop(0)

                            # create a cluster out of the tree
                            if level+1 != height_of_tree:
                                # no cluster for the root cluster
                                print("KOKOKOKOKOKO",temp_tree.get_data())
                                cluster_graph = graph.subgraph([str(s) for s in temp_tree.get_data()])
                                cluster = Cluster(tree_name, cluster_graph, i)
                                cluster.set_intermediate()
                                self._network.add_cluster(i, cluster)

                            new_lower_level_trees.append(temp_tree)

                            print(level+1)
                            print(height_of_tree)
                            if level+1 == height_of_tree:
                                current_tree = temp_tree
                                print("PRINTING TREE")
                                printTree(temp_tree)
                                print("printed tree")
                        elif len(lower_level_trees) == 1:
                            # todo ignore same single cluster in multiple levels
                            if level + 1 == height_of_tree:
                                print("SHOULD REACH ONCE")
                                print(lower_level_trees[0].getNodeValue())
                                tree_name = ''
                                if level + 1 == height_of_tree:
                                    tree_name = root_cluster_key
                                else:
                                    tree_name = lower_level_trees[0].getNodeValue()

                                temp_tree = BinaryTree(tree_name)
                                temp_tree.set_data(lower_level_trees[0].get_data())
                                temp_tree.left = lower_level_trees[0]

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
                                    self._network.add_cluster(i, cluster)

                                new_lower_level_trees.append(temp_tree)
                                if level + 1 == height_of_tree:
                                    current_tree = temp_tree
                            else:
                                new_lower_level_trees.append(lower_level_trees.pop(0))

                        print("AND NEW LEVEL NODE COUNT ", len(lower_level_trees))

                    print("NEW LEVEL NODE COUNT ", len(new_lower_level_trees))
                    lower_level_trees = new_lower_level_trees

                print("Tree for cluster ", root_cluster_key, " is ")
                (self._network.find_cluster_by_id(root_cluster_key)).set_tree(current_tree)
                # set the tree children here, traversing starts from the children

                printTree(current_tree)
                (self._network.find_cluster_by_id(root_cluster_key)).set_tree_leafs(get_leafs(current_tree))

            print("KO")
            for clus in self._network.clusters_by_level(i):
                clus.print()
                # clus.tree.display_tree()
        print("UNCONTAINED CLUSTERS ")
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
        print("Height of clusters", self._height_of_cluster)
        for i in range(0,self._height_of_cluster+1):
            print("At level ", i, " there are ", len(self._network.clusters_by_level(i)), " clusters.")
            self._network.print_cluster_by_level(i)

    # make lower level clusters connected to the upper level clusters.
    # the lower level will be connected to the tree nodes of upper level cluster
    def setup_clusters(self):
        for i in (range(0, self._height_of_cluster)):
            index = 0
            for clus in self._network.clusters_by_level(i):
                self

    def save_network(self, output_file):
        with open(output_file, 'wb') as output:
            pickle.dump(self._network, output, pickle.HIGHEST_PROTOCOL)

    def load_network(self, input_file):
        with open(input_file, 'rb') as input:
            self._network = pickle.load(input)

    def run(self, rounds):
        # simulations to run
        # initial publisher and object
        for round in range(0, rounds):

            publisher = self._network.find_cluster_by_id('c0_l0')
            obj = Object('obj1', publisher.cluster_id)

            self._network.publish(publisher, obj)
            results = []
            r = random.Random()
            r.seed("test"+str(round))
            # r.seed("test")

            optimal_processing_load = OrderedDict()
            for i in range(0, self._peer_count):
                optimal_processing_load[str(i)] = 0


            for i in range(1, self._op_count + 1):
                print("Operation ", i)
                print(obj.owner_id)
                owner = self._network.find_cluster_by_id(obj.owner_id)
                # select random peer to create a move request except the owner peer.
                # todo need to change requester, make peer instead of cluster
                random_id = str(r.randint(0, (self._network.size() - 1)))
                while 'c' + random_id + '_l0' == obj.owner_id:
                    random_id = str(r.randint(0, self._network.size() - 1))

                print('c' + random_id + '_l0')
                mover = self._network.find_cluster_by_id('c' + random_id + '_l0')
                print(mover.cluster_id)
                assert 'c' + random_id + '_l0' != obj.owner_id
                print('MOVING FROM ', owner.cluster_id, ' to ', mover.cluster_id)
                print("RANDOM ID FOR MOVER IS ", random_id)
                print("SIZE OF NETWORK IS ", self._network.size())
                res = self._network.move(mover, obj)
                res['msg'] = 'MOVING FROM ' + str(owner.cluster_id) + ' to ' + str(mover.cluster_id)
                res['mover_id'] = str(mover.id)
                res['optimal_cost'] = nx.dijkstra_path_length(self._network.get_graph(), mover.id, owner.id, 'weight')
                res['stretch'] = res['LB_SPIRAL_cost'] / res['optimal_cost']
                print("LB_SPIRAL cost IS ", res['LB_SPIRAL_cost'])
                print("Optimal cost IS ", res['optimal_cost'])
                print("stretch IS ", res['stretch'])
                print(obj.owner_id)
                optimal_processing_load[str(mover.id)] = optimal_processing_load[str(mover.id)] + 1

                results.append(res)
            print(results)

            #         prepare the result
            total_cost_optimal = 0
            total_cost_LB_SPIRAL = 0
            total_communication_cost = 0
            total_hops_only = 0
            stretch = 0
            processing_load = OrderedDict()
            for i in range(0, self._peer_count):
                processing_load[str(i)] = 0

            for result in results:
                total_cost_optimal = total_cost_optimal + result['optimal_cost']
                total_cost_LB_SPIRAL = total_cost_LB_SPIRAL + result['LB_SPIRAL_cost']
                total_communication_cost = total_communication_cost + result['inform_cost_only']
                total_hops_only = total_hops_only + result['hopsOnly']
                stretch = stretch + result['stretch']
                processing_load[result['mover_id']] = processing_load[result['mover_id']] + result['processing_load']

            output = dict()
            output['results'] = results
            output['total_optimal'] = total_cost_optimal
            output['total_LB_SPIRAL'] = total_cost_LB_SPIRAL
            output['stretch'] = stretch
            output['processing_load'] = processing_load
            output['optimal_processing_load'] = optimal_processing_load
            output['total_inform_cost_only'] = total_communication_cost
            output['total_hops_only'] = total_hops_only
            path = 'results'

            if not os.path.exists(path):
                os.makedirs(path)

            file = open(
                os.path.join(path, str(self._op_count) + '-Operations__' + str(self._peer_count) + '-Peers'+str(round)+'.txt'), 'w')
            json.dump(output, file)
            file.close()



