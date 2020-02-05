import matplotlib.pyplot as plt
import networkx as nx
import random
import numpy
import sys

from typing import Iterator, Iterable, Set, Union


class Network:
    """
    The network connecting the various miners.
    """

    def __init__(self, graph):
        """
        Initializes the network.
        """
        self._network_graph = graph
        self._clusters = {}
        self._peers = {}
        self._height_of_clusters = None
        self._root_cluster_id = None
        self._size = nx.number_of_nodes(graph)


        self._intermediate_clusters = {}

    def add_peer(self, peer_id, peer):
        """
        Adds a peer to the network.
        """
        self._peers[peer_id] = peer
        # self._network_graph.add_node(peer.get_peer_id())

        # miner.set_network(self)
        # miner.add_block(self._GENESIS_BLOCK)

    def add_cluster(self, level, cluster):
        """
        Adds a to the network.
        """
        cluster_size = len(self._clusters[int(level)])
        cluster.index = cluster_size
        self._clusters[int(level)].append(cluster)

    def add_cluster_level(self, level):
        """
        Adds a to the network.
        """
        print("AADDING LEVEL ",level)
        self._clusters[int(level)] = []

    def set_height_of_clusters(self, height):
        self._height_of_clusters = height

    def get_graph(self):
        return self._network_graph

    def size(self):
        return self._size

    def cluster_by_id(self, cluster_id):
        return self._clusters[cluster_id]

    def clusters_by_level(self, level):

        # return self._clusters[level]
        print("LEVEL IS ",level)
        print("CLUSTERS IN LEVEL IS ",self._clusters[level])
        return [a for a in self._clusters[level] if not a.intermediate]

    def find_cluster_by_id(self, cluster_id):
        for level in self._clusters:
            for cluster in self._clusters[level]:
                if cluster_id == cluster.cluster_id:
                    return cluster
        return None

    def print_cluster_by_level(self, level):
        for cl in self._clusters[level]:
            if not cl.intermediate:
                print("CLUSTER ID ", cl.cluster_id, " and nodes contained ", cl.graph.nodes, " leader is ", cl.leader_id)

    def print_Intermediate_clusters_by_level(self, level):
        for cl in self._clusters[level]:
            if cl.intermediate:
                print("INTERMEDIATE CLUSTER ID ", cl.cluster_id, " and nodes contained ", cl.graph.nodes, " leader is ", cl.leader_id)
                print("PEERS ", cl.peers)

    def remove_cluster_by_id(self, cluster_id, level):
        temp = []
        for cluster in self._clusters[level]:
            if cluster.cluster_id is not cluster_id:
                temp.append(cluster)
        self._clusters[level] = temp

    def draw_cluster(self, cluster_id):
        print('drawing cluster ', cluster_id)
        cluster = self.find_cluster_by_id(cluster_id)
        node_pos = nx.spring_layout(self._network_graph)

        edge_weight = nx.get_edge_attributes(self._network_graph, 'weight')
        # Draw the nodes
        nx.draw_networkx_nodes(self._network_graph, pos = node_pos, nodelist=[str(i) for i in cluster.graph])
        nx.draw_networkx(self._network_graph, node_pos, node_color='grey', node_size=100)
        # Draw the edges
        nx.draw_networkx_edges(self._network_graph, node_pos, edge_color='black')
        # Draw the edge labels
        nx.draw_networkx_edge_labels(self._network_graph, node_pos, edge_color='red', edge_labels=edge_weight)
        plt.title(cluster_id)
        plt.show()
        graph = cluster.graph
        print("The graph is ",graph)

    def shortest_path_in_cluster(self, cluster_id):
        self

    def check_for_intersection(self, cluster_id, path):
        print("PATH", path)
        print("CLUSTER ID ", cluster_id)
        if cluster_id in path:
            return cluster_id
        return None

    def publish(self, cluster, obj):
        cluster.publish(obj)
        assert cluster.object is not None
        # use to set leader
        peer_id = cluster.peers[0]

        i = 1
        hops = 0
        level_clusters = self.clusters_by_level(i)
        level_cluster_size = len(self.clusters_by_level(i))
        path = []
        prev = cluster.cluster_id

        prev_cluster = cluster

        path.append(prev)
        # last cluster to visit in the level
        last_cluster = None
        for cl_index in range(level_cluster_size):
            print(cl_index)
            print(level_clusters[cl_index].peers)
            if cluster.peers[0] in level_clusters[cl_index].peers:
                hops = hops + 1
                level_clusters[cl_index].set_previous(prev)
                prev = level_clusters[cl_index].cluster_id

                level_clusters[cl_index].set_previous_cluster(prev_cluster)
                prev_cluster = level_clusters[cl_index]

                path.append(level_clusters[cl_index].cluster_id)

                last_cluster = level_clusters[cl_index]

                # make leader
                # the only peer in the cluster
                level_clusters[cl_index].set_leader(peer_id)

                print("FOUND ", cluster.peers[0])
        i= i+1

        print("DINTERMEDIATE PREV CLUSTER")
        ttt = last_cluster
        while ttt is not None:
            print(ttt.cluster_id)
            ttt = ttt.previous_cluster


        # need to consider intra cluster routing from this level
        while i <= self._height_of_clusters:
            print("OK LAST CLUSTER IS ", prev_cluster.cluster_id)

            cl_index = 0
            level_clusters = self.clusters_by_level(i)
            level_cluster_size = len(self.clusters_by_level(i))

            print("LEVEL CLUSTER SIZE ", level_cluster_size)
            for cl_index in range(level_cluster_size):
                level_leaf_found = False
                print("CL_INDEX IS ", cl_index)
                print("CLUSTER ID IS ", level_clusters[cl_index].cluster_id)
                print(level_clusters[cl_index].peers)
                print(level_clusters[cl_index].tree_leafs)

                for leaf in level_clusters[cl_index].tree_leafs:
                    print("LEAF CONTENT ", leaf.rootid)
                    if leaf.rootid == prev_cluster.cluster_id:
                        level_leaf_found = True
                        # start tree traversal
                        while leaf.parent is not None :
                            # end when root is reached. Root has no parent
                            if leaf.rootid != leaf.parent.rootid:
                                hops = hops + 1
                                print("leaf root id", leaf.rootid)
                                print("leaf.parent.root id ", leaf.parent.rootid)
                                path.append(leaf.parent.rootid)

                                # set prev
                                self.find_cluster_by_id(leaf.parent.rootid).set_previous(prev)
                                prev = self.find_cluster_by_id(leaf.parent.rootid).cluster_id

                                # set previous cluster
                                print("WHAT  IS THIS")
                                print(leaf.parent.rootid)
                                self.find_cluster_by_id(leaf.parent.rootid).set_previous_cluster(prev_cluster)
                                prev_cluster = self.find_cluster_by_id(leaf.parent.rootid)

                                print("PREV PREV CLUSTER ", prev_cluster.cluster_id)

                            print("INTERMEDIATE PREV CLUSTER")
                            print(prev_cluster.cluster_id)
                            print(prev_cluster.previous_cluster.cluster_id)
                            print(prev_cluster.previous_cluster.previous_cluster.cluster_id)
                            # print(prev_cluster.previous_cluster.previous_cluster.previous_cluster.cluster_id)
                            print ('leaf', leaf.rootid)
                            print ('leaf.parent', leaf.parent.rootid)
                            leaf = leaf.parent

                            print('AFTER leaf', leaf.rootid)
                            # print('AFTER leaf.parent', leaf.parent.rootid)

                            if leaf.parent is None:

                                # set leader
                                self.find_cluster_by_id(leaf.rootid).set_leader(peer_id)
                                print('i is ',i )
                                print('self._height of clusters', self._height_of_clusters)
                                if i == self._height_of_clusters:
                                    self._root_cluster_id = leaf.rootid
                                    break

                                # found the root
                                prev_cluster = self.find_cluster_by_id(leaf.rootid)

                                # set previous for the root
                                self.find_cluster_by_id(leaf.rootid).set_previous(prev)
                                prev = self.find_cluster_by_id(leaf.rootid).cluster_id

                                # set previous cluster

                                # print("SETTING FOR cluster ", leaf.rootid)
                                # print("PREVIOUS CLUSTER ", prev_cluster.cluster_id)
                                # self.find_cluster_by_id(leaf.rootid).set_previous_cluster(prev_cluster)
                                # prev_cluster = self.find_cluster_by_id(leaf.rootid)
                                # print("SETTING leaf rootid", leaf.rootid)
                                # print("SETTING prev_cluster clusterid", prev_cluster.cluster_id)

                                # don't set leader in the tree, the tree's root cluter's leader is the leader in all levels of tree
                                # print ("UPDATED ",last_cluster.cluster_id)
                            print(prev_cluster.cluster_id)
                            print("=INTERMEDIATE PREV CLUSTER")
                            print(prev_cluster.cluster_id)
                            print(prev_cluster.previous_cluster.cluster_id)
                            print(prev_cluster.previous_cluster.previous_cluster.cluster_id)
                            # print(prev_cluster.previous_cluster.previous_cluster.previous_cluster.cluster_id)

                        break
                if level_leaf_found:
                    break

            # prev_cluster = last_cluster
            print("--INTERMEDIATE PREV CLUSTER")
            ttt = prev_cluster
            while ttt is not None:
                print (ttt.cluster_id)
                ttt = ttt.previous_cluster

            # print(prev_cluster.previous_cluster.previous_cluster.previous_cluster.cluster_id)

            #     after the tree traversal, need to traverse all the clusters in this level containing the peer

            level_clusters = self.clusters_by_level(i)
            level_cluster_size = len(self.clusters_by_level(i))
            for inner_cl_index in range(prev_cluster.index+1, level_cluster_size):
                if cluster.peers[0] in level_clusters[inner_cl_index].peers:
                    hops = hops + 1
                    level_clusters[inner_cl_index].set_previous(prev)
                    prev = level_clusters[inner_cl_index].cluster_id

                    level_clusters[inner_cl_index].set_previous_cluster(prev_cluster)
                    prev_cluster = level_clusters[inner_cl_index]

                    path.append(level_clusters[inner_cl_index].cluster_id)

                    prev_cluster = level_clusters[inner_cl_index]

                    # make leader
                    # the only peer in the cluster
                    level_clusters[cl_index].set_leader(peer_id)

                    print("FOUND ", cluster.peers[0])

            print("PATH IS ",path)

            i = i + 1

        print("FINAL PATHI IS ", path)
        cluster.object.set_path(path)

        print("NETWORK ROOT CLUSTER IS ")
        cluster = self.find_cluster_by_id(self._root_cluster_id)
        print("CINTERMEDIATE PREV CLUSTER")
        ttt = prev_cluster
        while ttt is not None:
            print(ttt.cluster_id)
            ttt = ttt.previous_cluster

        print("HOPS TOTAL = ", hops)
        return path

    def move(self, cluster, obj):
        print("THE ORIGINAL PATH IS")
        print(obj.path)
        owner_id = self.find_cluster_by_id(obj.owner_id).id
        mover_id = self.find_cluster_by_id(cluster.cluster_id).id
        res = self.find_intersection(cluster, obj, owner_id, mover_id)
        assert res['intersection'] is not None
        assert res['path'] is not None
        assert res['hops'] is not None

        print('intersection at ', res['intersection'])
        print('intersection path at ', res['path'])
        print('hops count ', res['hops'])

        # inform the leader in each hop
        inform_cost = 0
        for path in res['path']:
            print("PATH INSIDE INFORM COST CALCULATION IS ", path)
            #     if the mover node is not the leader in the cluster it should inform the cluster so calculate the distance form the node to the leader.
            print("LEADER ID")
            print(self.find_cluster_by_id(path).leader_id)
            print(self.find_cluster_by_id(path).intermediate)

            print("ID")
            print(mover_id)
            if self.find_cluster_by_id(path).intermediate == False:
                if mover_id == self.find_cluster_by_id(path).leader_id:
                    print("SAME")
                else:
                    inform_cost = inform_cost + nx.dijkstra_path_length(self.find_cluster_by_id(path).graph, mover_id,
                                                                        self.find_cluster_by_id(path).leader_id, 'weight')
                    print("UPDATED INFORM COST ", inform_cost)
            self.find_cluster_by_id(path).set_leader(mover_id)

        print("INFORM COST IS ", inform_cost)
        # need to update the path

        #     need to delete the pointers of the previous owner
        print("THE INDEX TO DELETE FROM ", obj.path.index(res['intersection']))
        index_to_delete_form = obj.path.index(res['intersection']) - 1

        print(obj.path[index_to_delete_form])
        cluster_to_delete_path_from = self.find_cluster_by_id(obj.path[index_to_delete_form])
        tmp_cluster = self.find_cluster_by_id(obj.path[index_to_delete_form])
        print("BBBEFOREEE")
        while tmp_cluster is not None:
            print(tmp_cluster.cluster_id)
            tmp_cluster = tmp_cluster.previous_cluster
        delete_hop = 0
        while cluster_to_delete_path_from is not None:
            temp_cluster = cluster_to_delete_path_from.previous_cluster
            cluster_to_delete_path_from.previous_cluster = None
            cluster_to_delete_path_from = temp_cluster
            delete_hop = delete_hop + 1

        print("AAAFTERRRR")
        tmp_cluster = self.find_cluster_by_id(obj.path[index_to_delete_form])
        while tmp_cluster is not None:
            print(tmp_cluster.cluster_id)
            tmp_cluster = tmp_cluster.previous_cluster

        print("DELETE CHECKING")
        #  check for previous cluster links
        for i in range(index_to_delete_form+1):
            print(self.find_cluster_by_id(obj.path[i]).cluster_id )
            assert self.find_cluster_by_id(obj.path[i]).previous_cluster is None

        # set previous cluster in the new path
        print("PRINTING PATHSS")
        res['path'].reverse()
        for i in range(0, len(res['path'])-1):
            print(res['path'][i])
            if res['path'][i+1] is not None:
                self.find_cluster_by_id(res['path'][i]).set_previous_cluster(self.find_cluster_by_id((res['path'][i+1])))

        # calculate the processing load
        print("PROCESSING LOAD")
        processing_load = 0
        for j in range(0, len(res['path'])):
            if mover_id in self.find_cluster_by_id(res['path'][j]).peers:
                processing_load = processing_load + 1
            print(res['path'][j])

        print("PROCESSING LOAD IS", processing_load)
        # change the ownership equivalent to moving the file
        obj.set_owner(cluster.cluster_id)

        print("PRINTING THE WHOLE PREVIOUS CLUSTER ")
        # print(self.find_cluster_by_id('c2_l4_1').cluster_id)
        # print(self.find_cluster_by_id('c2_l4_1').previous_cluster.cluster_id)
        cluster = self.find_cluster_by_id(self._root_cluster_id)
        # cluster = self.find_cluster_by_id(res['intersection'])

        ttt = cluster
        while ttt is not None:
            print(ttt.cluster_id)
            ttt = ttt.previous_cluster

        # update the object's path
        root_cluster = self.find_cluster_by_id(self._root_cluster_id)
        print(root_cluster)
        path = [root_cluster.cluster_id]
        while root_cluster.previous_cluster is not None:
            root_cluster = root_cluster.previous_cluster
            path.append(root_cluster.cluster_id)
        obj.set_path(list(reversed(path)))

        print("NEW OBJECT PATH IS ", obj.path)
        # while temp_cluster is not None:
        #     print("---", temp_cluster.cluster_id)
        #     temp_cluster = temp_cluster.previous_cluster

        # need to calculate the shortest path from source to node in the intersected cluster
        print("THE INTERSECTED CLUSTER ID IS ")
        print(self.find_cluster_by_id(res['intersection']).cluster_id)
        print(mover_id)
        print(owner_id)
        print(self.find_cluster_by_id(res['intersection']).graph.nodes)
        print(nx.dijkstra_path_length(self.find_cluster_by_id(res['intersection']).graph, mover_id, owner_id, 'weight'))
        print('delete hop count ', delete_hop)

        print("DONE MOVE")

        res['delete_hops'] = delete_hop
        res['shortest_path_length_in_intersected_cluster'] = nx.dijkstra_path_length(self.find_cluster_by_id(res['intersection']).graph, mover_id, owner_id, 'weight')
        # res['LB_SPIRAL_cost'] = res['hops'] + res['delete_hops'] + res['shortest_path_length_in_intersected_cluster']
        # res['LB_SPIRAL_cost'] = res['hops'] + res['shortest_path_length_in_intersected_cluster'] + inform_cost
        res['LB_SPIRAL_cost'] = res['hops'] + inform_cost * 2
        res['inform_cost_only'] = inform_cost * 2
        res['hopsOnly'] = res['hops']
        res['shortest_path_length_in_intersected_cluster'] =res['shortest_path_length_in_intersected_cluster']
        res['processing_load'] = processing_load
        return res

    # def find_intersection1(self, cluster, obj):
    #     # use to set leader
    #     peer_id = cluster.peers[0]
    #
    #     i = 1
    #     hops = 0
    #     level_clusters = self.clusters_by_level(i)
    #     level_cluster_size = len(self.clusters_by_level(i))
    #     new_path = []
    #     prev = cluster.cluster_id
    #
    #     prev_cluster = cluster
    #
    #     new_path.append(prev)
    #     # last cluster to visit in the level
    #     last_cluster = None
    #
    #     intersection = False
    #     while 1:
    #         for cl_index in range(level_cluster_size):
    #             print(cl_index)
    #             print(level_clusters[cl_index].peers)
    #             if cluster.peers[0] in level_clusters[cl_index].peers:
    #                 level_clusters[cl_index].set_previous(prev)
    #                 prev = level_clusters[cl_index].cluster_id
    #
    #                 level_clusters[cl_index].set_previous_cluster(prev_cluster)
    #                 prev_cluster = level_clusters[cl_index]
    #
    #                 new_path.append(level_clusters[cl_index].cluster_id)
    #
    #                 last_cluster = level_clusters[cl_index]
    #
    #                 # make leader
    #                 # the only peer in the cluster
    #                 level_clusters[cl_index].set_leader(peer_id)
    #
    #                 intersection = self.check_for_intersection(cluster.cluster_id, obj.path)
    #                 if intersection is not None:
    #                     d = dict()
    #                     d['intersection'] = intersection
    #                     d['path'] = new_path
    #                     return d
    #
    #                 print("FOUND ", cluster.peers[0])
    #
    #         i = i + 1
    #         # need to consider intra cluster routing from this level
    #         while i <= self._height_of_clusters:
    #             print("OK LAST CLUSTER IS ", last_cluster.cluster_id)
    #
    #             cl_index = 0
    #             level_clusters = self.clusters_by_level(i)
    #             level_cluster_size = len(self.clusters_by_level(i))
    #             for cl_index in range(level_cluster_size):
    #                 level_leaf_found = False
    #                 print("CL_INDEX IS ", cl_index)
    #                 print("CLUSTER ID IS ", level_clusters[cl_index].cluster_id)
    #                 print(level_clusters[cl_index].peers)
    #                 print(level_clusters[cl_index].tree_leafs)
    #
    #                 for leaf in level_clusters[cl_index].tree_leafs:
    #                     print("LEAF CONTENT ", leaf.rootid)
    #                     if leaf.rootid == last_cluster.cluster_id:
    #                         level_leaf_found = True
    #                         # start tree traversal
    #                         while leaf.parent is not None:
    #                             # end when root is reached. Root has no parent
    #                             if leaf.rootid != leaf.parent.rootid:
    #                                 hops = hops + 1
    #                                 print("leaf root id", leaf.rootid)
    #                                 print("leaf.parent.root id ", leaf.parent.rootid)
    #                                 new_path.append(leaf.parent.rootid)
    #
    #                                 # set prev
    #                                 self.find_cluster_by_id(leaf.parent.rootid).set_previous(prev)
    #                                 prev = self.find_cluster_by_id(leaf.parent.rootid).cluster_id
    #
    #                                 # set previous cluster
    #                                 self.find_cluster_by_id(leaf.parent.rootid).set_previous_cluster(prev_cluster)
    #                                 prev_cluster = self.find_cluster_by_id(leaf.parent.rootid)
    #
    #                                 intersection = self.check_for_intersection(leaf.rootid, obj.path)
    #                                 if intersection is not None:
    #                                     d = dict()
    #                                     d['intersection'] = intersection
    #                                     d['path'] = new_path
    #                                     print("FOUND AND RETURNING")
    #                                     return d
    #
    #                             leaf = leaf.parent
    #                             if leaf.parent is None:
    #                                 # found the root
    #                                 last_cluster = self.find_cluster_by_id(leaf.rootid)
    #
    #                                 # set previous for the root
    #                                 self.find_cluster_by_id(leaf.rootid).set_previous(prev)
    #                                 prev = self.find_cluster_by_id(leaf.rootid).cluster_id
    #
    #                                 # set previous cluster
    #                                 self.find_cluster_by_id(leaf.rootid).set_previous_cluster(prev_cluster)
    #                                 prev_cluster = self.find_cluster_by_id(leaf.rootid)
    #
    #                                 # set leader
    #                                 self.find_cluster_by_id(leaf.rootid).set_leader(peer_id)
    #
    #                                 # don't set leader in the tree, the tree's root cluter's leader is the leader in all levels of tree
    #                                 print("UPDATED ", last_cluster.cluster_id)
    #
    #                                 intersection = self.check_for_intersection(leaf.rootid, obj.path)
    #                                 if intersection is not None:
    #                                     d = dict()
    #                                     d['intersection'] = intersection
    #                                     d['path'] = new_path
    #                                     print("FOUND AND RETURNING")
    #                                     return d
    #
    #                         break
    #                 if level_leaf_found:
    #                     break
    #
    #             #     need to traverse all the clusters in this level containing the peer
    #
    #             level_clusters = self.clusters_by_level(i)
    #             level_cluster_size = len(self.clusters_by_level(i))
    #             for inner_cl_index in range(prev_cluster.index + 1, level_cluster_size):
    #                 if cluster.peers[0] in level_clusters[inner_cl_index].peers:
    #                     level_clusters[inner_cl_index].set_previous(prev)
    #                     prev = level_clusters[inner_cl_index].cluster_id
    #
    #                     level_clusters[inner_cl_index].set_previous_cluster(prev_cluster)
    #                     prev_cluster = level_clusters[inner_cl_index]
    #
    #                     new_path.append(level_clusters[inner_cl_index].cluster_id)
    #
    #                     last_cluster = level_clusters[inner_cl_index]
    #
    #                     # make leader
    #                     # the only peer in the cluster
    #                     level_clusters[cl_index].set_leader(peer_id)
    #
    #                     print("FOUND ", cluster.peers[0])
    #
    #                     intersection = self.check_for_intersection(level_clusters[inner_cl_index].cluster_id, obj.path)
    #                     if intersection is not None:
    #                         d = dict()
    #                         d['intersection'] = intersection
    #                         d['path'] = new_path
    #                         print("FOUND AND RETURNING")
    #                         return d
    #
    #             print("PATH IS ", new_path)
    #
    #             i = i + 1
    #
    #         print("FINAL PATHI IS ", new_path)
    #         cluster.object.set_path(new_path)
    #         d = dict()
    #         d['intersection'] = intersection
    #         d['path'] = new_path
    #         return d
    #
    #
    #     print("FINISHING MOVE")

    def find_intersection(self, cluster, obj, owner_id, mover_id):
        print("OBJECTS PATH IS ", obj.path)
        # use to set leader
        peer_id = cluster.peers[0]
        # keep track of level with i
        i = 1
        hops = 0
        level_clusters = self.clusters_by_level(i)
        level_cluster_size = len(self.clusters_by_level(i))
        new_path = []
        prev = cluster.cluster_id

        prev_cluster = cluster

        new_path.append(prev)
        # last cluster to visit in the level

        intersection = False
        while 1:
            for cl_index in range(level_cluster_size):
                print("CL INDEX IS ",cl_index)
                print("LEVEL CLUSTER PEERS ",level_clusters[cl_index].peers)
                if cluster.peers[0] in level_clusters[cl_index].peers:
                    hops = hops + 1
                    level_clusters[cl_index].set_previous(prev)
                    prev = level_clusters[cl_index].cluster_id

                    # level_clusters[cl_index].set_previous_cluster(prev_cluster)
                    prev_cluster = level_clusters[cl_index]

                    new_path.append(level_clusters[cl_index].cluster_id)

                    # make leader
                    # the only peer in the cluster
                    # set the leader after calculating communication cost from mover to the leader.
                    # level_clusters[cl_index].set_leader(peer_id)

                    # intersection = self.check_for_intersection(cluster.cluster_id, obj.path)
                    intersection = self.check_for_intersection(level_clusters[cl_index].cluster_id, obj.path)

                    if intersection is not None and owner_id in level_clusters[cl_index].graph.nodes and mover_id in level_clusters[cl_index].graph.nodes:
                        d = dict()
                        d['intersection'] = intersection
                        d['path'] = new_path
                        d['hops'] = hops
                        print("FOUND AND RETURNING 1")
                        return d

                    print("FOUND ", cluster.peers[0])

            i = i + 1

            ttt = prev_cluster
            while ttt is not None:
                print(ttt.cluster_id)
                ttt = ttt.previous_cluster
            # need to consider intra cluster routing from this level
            while i <= self._height_of_clusters:
                print("CHECKING IN LEVEL ", i)

                # for asdf in range(0, len(self._clusters)):
                #     for c in self._clusters[asdf]:
                #         if str(65) in c.peers:
                #             print(c.cluster_id)
                #             print (c.peers)

                print("OK LAST CLUSTER IS ", prev_cluster.cluster_id)

                cl_index = 0
                level_clusters = self.clusters_by_level(i)
                level_cluster_size = len(self.clusters_by_level(i))
                print("LEVEL CLUSTER SIZE ", level_cluster_size)
                print("LEVEL CLUSTER ID FIRST ELEMENT ",level_clusters[0].cluster_id)
                print(level_clusters[0].peers)
                for cl_index in range(level_cluster_size):
                    level_leaf_found = False
                    print("CL_INDEX IS ", cl_index)
                    print("CLUSTER ID IS ", level_clusters[cl_index].cluster_id)
                    print(level_clusters[cl_index].peers)
                    print(level_clusters[cl_index].tree_leafs)

                    for leaf in level_clusters[cl_index].tree_leafs:
                        print("LEAF CONTENT ", leaf.rootid)
                        if leaf.rootid == prev_cluster.cluster_id:
                            level_leaf_found = True
                            # start tree traversal
                            # todo note no intersection at intermediate levels in the tree
                            while leaf.parent is not None:
                                # end when root is reached. Root has no parent
                                if leaf.rootid != leaf.parent.rootid:
                                    hops = hops + 1
                                    print("leaf root id", leaf.rootid)
                                    print("find intersection leaf.parent.root id ", leaf.parent.rootid)

                                    # set prev
                                    self.find_cluster_by_id(leaf.parent.rootid).set_previous(prev)
                                    prev = self.find_cluster_by_id(leaf.parent.rootid).cluster_id

                                    # set previous cluster
                                    # self.find_cluster_by_id(leaf.parent.rootid).set_previous_cluster(prev_cluster)
                                    prev_cluster = self.find_cluster_by_id(leaf.parent.rootid)

                                    # intersection = self.check_for_intersection(leaf.rootid, obj.path)
                                    # if intersection is not None:
                                    #     d = dict()
                                    #     d['intersection'] = intersection
                                    #     d['path'] = new_path
                                    #     d['hops'] = hops
                                    #     print("FOUND AND RETURNING")
                                    #     return d

                                    new_path.append(leaf.parent.rootid)


                                # print(prev_cluster.cluster_id)
                                # print(prev_cluster.previous_cluster.cluster_id)
                                # print(prev_cluster.previous_cluster.previous_cluster.cluster_id)

                                leaf = leaf.parent
                                if leaf.parent is None:

                                    if i == self._height_of_clusters:
                                        self._root_cluster_id = leaf.rootid

                                        intersection = self.check_for_intersection(leaf.rootid, obj.path)
                                        if intersection is not None and owner_id in self.find_cluster_by_id(
                                                leaf.rootid).graph.nodes and mover_id in self.find_cluster_by_id(
                                                leaf.rootid).graph.nodes:
                                            d = dict()
                                            d['intersection'] = intersection
                                            d['path'] = new_path
                                            d['hops'] = hops
                                            print("FOUND AND RETURNING ROOOOT")
                                            return d

                                        break
                                    # found the root
                                    prev_cluster = self.find_cluster_by_id(leaf.rootid)

                                    # set previous for the root
                                    self.find_cluster_by_id(leaf.rootid).set_previous(prev)
                                    prev = self.find_cluster_by_id(leaf.rootid).cluster_id

                                    # set previous cluster
                                    # self.find_cluster_by_id(leaf.rootid).set_previous_cluster(prev_cluster)
                                    # prev_cluster = self.find_cluster_by_id(leaf.rootid)

                                    # set leader
                                    # self.find_cluster_by_id(leaf.rootid).set_leader(peer_id)
                                    # set the leader after calculating communication cost from mover to the leader.

                                    # don't set leader in the tree, the tree's root cluter's leader is the leader in all levels of tree
                                    print("UPDATED ", prev_cluster.cluster_id)

                                    intersection = self.check_for_intersection(leaf.rootid, obj.path)
                                    if intersection is not None and owner_id in self.find_cluster_by_id(leaf.rootid).graph.nodes and mover_id in self.find_cluster_by_id(leaf.rootid).graph.nodes:
                                        d = dict()
                                        d['intersection'] = intersection
                                        d['path'] = new_path
                                        d['hops'] = hops
                                        print("FOUND AND RETURNING 2")
                                        return d
                                print("WHAS")
                                # print(prev_cluster.cluster_id)
                                # print(prev_cluster.previous_cluster.cluster_id)
                                # print(prev_cluster.previous_cluster.previous_cluster.cluster_id)
                                # print(prev_cluster.previous_cluster.previous_cluster.previous_cluster.cluster_id)
                                # print(prev_cluster.previous_cluster.previous_cluster.previous_cluster.previous_cluster.cluster_id)
                                # print(prev_cluster.previous_cluster.previous_cluster.previous_cluster.previous_cluster.previous_cluster.cluster_id)

                            break
                    if level_leaf_found:
                        break

                #     need to traverse all the clusters in this level containing the peer

                level_clusters = self.clusters_by_level(i)
                level_cluster_size = len(self.clusters_by_level(i))
                for inner_cl_index in range(prev_cluster.index + 1, level_cluster_size):
                    if cluster.peers[0] in level_clusters[inner_cl_index].peers:
                        hops = hops + 1
                        level_clusters[inner_cl_index].set_previous(prev)
                        prev = level_clusters[inner_cl_index].cluster_id

                        # level_clusters[inner_cl_index].set_previous_cluster(prev_cluster)
                        prev_cluster = level_clusters[inner_cl_index]

                        new_path.append(level_clusters[inner_cl_index].cluster_id)

                        # make leader
                        # the only peer in the cluster
                        # set the leader after calculating communication cost from mover to the leader.

                        # level_clusters[cl_index].set_leader(peer_id)

                        print("FOUND ", cluster.peers[0])

                        intersection = self.check_for_intersection(level_clusters[inner_cl_index].cluster_id, obj.path)
                        if intersection is not None and owner_id in level_clusters[inner_cl_index].graph.nodes and mover_id in level_clusters[inner_cl_index].graph.nodes:
                            d = dict()
                            d['intersection'] = intersection
                            d['path'] = new_path
                            d['hops'] = hops
                            print(intersection)
                            print("FOUND AND RETURNING 3")
                            print(level_clusters[cl_index].graph.nodes)
                            print(level_clusters[cl_index].cluster_id)
                            return d

                print("PATH IS ", new_path)

                i = i + 1

            # NO MATCH FOUND UNTIL ROOT, SO INTERSECTION IS AT ROOT
            print("FINAL PATHI IS asdf ", new_path)
            print(self.find_cluster_by_id(self._root_cluster_id).peers)
            print(cluster.cluster_id)
            print("INTERSECTION ", intersection)
            # cluster.object.set_path(new_path)

            obj.set_path(new_path)
            d = dict()
            d['intersection'] = intersection
            d['path'] = new_path
            d['hops'] = hops
            print("zHOPS TOTAL MOVE = ", hops)
            return d

        print("FINISHING MOVE")

    # def find_intersection2(self, cluster, obj):
    #     # use to set leader
    #     peer_id = cluster.peers[0]
    #
    #     i = 1
    #     hops = 0
    #     level_clusters = self.clusters_by_level(i)
    #     level_cluster_size = len(self.clusters_by_level(i))
    #     new_path = []
    #     prev = cluster.cluster_id
    #
    #     prev_cluster = cluster
    #
    #     new_path.append(prev)
    #     # last cluster to visit in the level
    #
    #     intersection = False
    #     while 1:
    #         for cl_index in range(level_cluster_size):
    #             print("CL INDEX IS ",cl_index)
    #             print("LEVEL CLUSTER PEERS ",level_clusters[cl_index].peers)
    #             if cluster.peers[0] in level_clusters[cl_index].peers:
    #                 hops = hops + 1
    #                 level_clusters[cl_index].set_previous(prev)
    #                 prev = level_clusters[cl_index].cluster_id
    #
    #                 level_clusters[cl_index].set_previous_cluster(prev_cluster)
    #                 prev_cluster = level_clusters[cl_index]
    #
    #                 new_path.append(level_clusters[cl_index].cluster_id)
    #
    #                 # make leader
    #                 # the only peer in the cluster
    #                 level_clusters[cl_index].set_leader(peer_id)
    #
    #                 # intersection = self.check_for_intersection(cluster.cluster_id, obj.path)
    #                 intersection = self.check_for_intersection(level_clusters[cl_index].cluster_id, obj.path)
    #                 if intersection is not None:
    #                     d = dict()
    #                     d['intersection'] = intersection
    #                     d['path'] = new_path
    #                     d['hops'] = hops
    #                     return d
    #
    #                 print("FOUND ", cluster.peers[0])
    #
    #         i = i + 1
    #
    #         print("EMEDIATE")
    #         ttt = prev_cluster
    #         while ttt is not None:
    #             print(ttt.cluster_id)
    #             ttt = ttt.previous_cluster
    #         # need to consider intra cluster routing from this level
    #         while i <= self._height_of_clusters:
    #             print("OK LAST CLUSTER IS ", prev_cluster.cluster_id)
    #
    #             cl_index = 0
    #             level_clusters = self.clusters_by_level(i)
    #             level_cluster_size = len(self.clusters_by_level(i))
    #             for cl_index in range(level_cluster_size):
    #                 level_leaf_found = False
    #                 print("CL_INDEX IS ", cl_index)
    #                 print("CLUSTER ID IS ", level_clusters[cl_index].cluster_id)
    #                 print(level_clusters[cl_index].peers)
    #                 print(level_clusters[cl_index].tree_leafs)
    #
    #                 for leaf in level_clusters[cl_index].tree_leafs:
    #                     print("LEAF CONTENT ", leaf.rootid)
    #                     if leaf.rootid == prev_cluster.cluster_id:
    #                         level_leaf_found = True
    #                         # start tree traversal
    #                         # todo note no intersection at intermediate levels in the tree
    #                         while leaf.parent is not None:
    #                             # end when root is reached. Root has no parent
    #                             if leaf.rootid != leaf.parent.rootid:
    #                                 hops = hops + 1
    #                                 print("leaf root id", leaf.rootid)
    #                                 print("leaf.parent.root id ", leaf.parent.rootid)
    #
    #                                 # set prev
    #                                 self.find_cluster_by_id(leaf.parent.rootid).set_previous(prev)
    #                                 prev = self.find_cluster_by_id(leaf.parent.rootid).cluster_id
    #
    #                                 # set previous cluster
    #                                 self.find_cluster_by_id(leaf.parent.rootid).set_previous_cluster(prev_cluster)
    #                                 prev_cluster = self.find_cluster_by_id(leaf.parent.rootid)
    #
    #                                 intersection = self.check_for_intersection(leaf.rootid, obj.path)
    #                                 if intersection is not None:
    #                                     d = dict()
    #                                     d['intersection'] = intersection
    #                                     d['path'] = new_path
    #                                     d['hops'] = hops
    #                                     print("FOUND AND RETURNING")
    #                                     return d
    #
    #                                 new_path.append(leaf.parent.rootid)
    #
    #
    #                             print(prev_cluster.cluster_id)
    #                             print(prev_cluster.previous_cluster.cluster_id)
    #                             print(prev_cluster.previous_cluster.previous_cluster.cluster_id)
    #
    #                             leaf = leaf.parent
    #                             if leaf.parent is None:
    #
    #                                 if i == self._height_of_clusters:
    #                                     self._root_cluster_id = leaf.rootid
    #                                     break
    #                                 # found the root
    #                                 prev_cluster = self.find_cluster_by_id(leaf.rootid)
    #
    #                                 # set previous for the root
    #                                 self.find_cluster_by_id(leaf.rootid).set_previous(prev)
    #                                 prev = self.find_cluster_by_id(leaf.rootid).cluster_id
    #
    #                                 # set previous cluster
    #                                 # self.find_cluster_by_id(leaf.rootid).set_previous_cluster(prev_cluster)
    #                                 # prev_cluster = self.find_cluster_by_id(leaf.rootid)
    #
    #                                 # set leader
    #                                 self.find_cluster_by_id(leaf.rootid).set_leader(peer_id)
    #
    #                                 # don't set leader in the tree, the tree's root cluter's leader is the leader in all levels of tree
    #                                 print("UPDATED ", prev_cluster.cluster_id)
    #
    #                                 intersection = self.check_for_intersection(leaf.rootid, obj.path)
    #                                 if intersection is not None:
    #                                     d = dict()
    #                                     d['intersection'] = intersection
    #                                     d['path'] = new_path
    #                                     d['hops'] = hops
    #                                     print("FOUND AND RETURNING")
    #                                     return d
    #                             print("WHAS")
    #                             print(prev_cluster.cluster_id)
    #                             print(prev_cluster.previous_cluster.cluster_id)
    #                             print(prev_cluster.previous_cluster.previous_cluster.cluster_id)
    #                             # print(prev_cluster.previous_cluster.previous_cluster.previous_cluster.cluster_id)
    #                             # print(prev_cluster.previous_cluster.previous_cluster.previous_cluster.previous_cluster.cluster_id)
    #                             # print(prev_cluster.previous_cluster.previous_cluster.previous_cluster.previous_cluster.previous_cluster.cluster_id)
    #
    #                         break
    #                 if level_leaf_found:
    #                     break
    #
    #             #     need to traverse all the clusters in this level containing the peer
    #
    #             level_clusters = self.clusters_by_level(i)
    #             level_cluster_size = len(self.clusters_by_level(i))
    #             for inner_cl_index in range(prev_cluster.index + 1, level_cluster_size):
    #                 if cluster.peers[0] in level_clusters[inner_cl_index].peers:
    #                     hops = hops + 1
    #                     level_clusters[inner_cl_index].set_previous(prev)
    #                     prev = level_clusters[inner_cl_index].cluster_id
    #
    #                     level_clusters[inner_cl_index].set_previous_cluster(prev_cluster)
    #                     prev_cluster = level_clusters[inner_cl_index]
    #
    #                     new_path.append(level_clusters[inner_cl_index].cluster_id)
    #
    #                     # make leader
    #                     # the only peer in the cluster
    #                     level_clusters[cl_index].set_leader(peer_id)
    #
    #                     print("FOUND ", cluster.peers[0])
    #
    #                     intersection = self.check_for_intersection(level_clusters[inner_cl_index].cluster_id, obj.path)
    #                     if intersection is not None:
    #                         d = dict()
    #                         d['intersection'] = intersection
    #                         d['path'] = new_path
    #                         d['hops'] = hops
    #                         print("FOUND AND RETURNING")
    #                         return d
    #
    #             print("PATH IS ", new_path)
    #
    #             i = i + 1
    #
    #         print("FINAL PATHI IS ", new_path)
    #         cluster.object.set_path(new_path)
    #         d = dict()
    #         d['intersection'] = intersection
    #         d['path'] = new_path
    #         d['hops'] = hops
    #         print("HOPS TOTAL MOVE = ", hops)
    #         return d
    #
    #     print("FINISHING MOVE")

    # def moveCopy(self, cluster, obj):
    #     print("INSIDE MOVE")
    #     print(obj.path)
    #     # while the directory path is not intersected
    #     peer_id = cluster.cluster_id
    #     current_cluster = None
    #     # while current_cluster not in obj.path:
    #     intersection = False
    #     while 1:
    #         # use to set leader
    #         # peer_id = cluster.peers[0]
    #         #
    #         i = 1
    #         hops = 0
    #         level_clusters = self.clusters_by_level(i)
    #         level_cluster_size = len(self.clusters_by_level(i))
    #         new_path = []
    #         prev = cluster.cluster_id
    #
    #         prev_cluster = cluster
    #
    #         new_path.append(prev)
    #         # last cluster to visit in the level
    #         last_cluster = None
    #         for cl_index in range(level_cluster_size):
    #             print(cl_index)
    #             print(level_clusters[cl_index].peers)
    #             if cluster.peers[0] in level_clusters[cl_index].peers:
    #
    #                 # check
    #                 intersection = self.check_for_intersection(cluster.cluster_id, obj.path)
    #                 if intersection is not None:
    #                     print("------------------------------INTERSECTION------------------------------, INNN", intersection)
    #                     break
    #                 else:
    #                     # visit the spiral path in upward phase to the root.
    #                     level_clusters[cl_index].set_previous(prev)
    #                     prev = level_clusters[cl_index].cluster_id
    #
    #                     level_clusters[cl_index].set_previous_cluster(prev_cluster)
    #                     prev_cluster = level_clusters[cl_index]
    #
    #                     new_path.append(level_clusters[cl_index].cluster_id)
    #
    #                     last_cluster = level_clusters[cl_index]
    #
    #                     # make leader
    #                     # the only peer in the cluster
    #                     level_clusters[cl_index].set_leader(peer_id)
    #
    #         if intersection is not None:
    #             break
    #         i = i + 1
    #
    #         # need to consider intra cluster routing from this level
    #         while i <= self._height_of_clusters:
    #             print("OK LAST CLUSTER IS ", last_cluster.cluster_id)
    #
    #             cl_index = 0
    #             level_clusters = self.clusters_by_level(i)
    #             level_cluster_size = len(self.clusters_by_level(i))
    #             for cl_index in range(level_cluster_size):
    #                 level_leaf_found = False
    #
    #                 print("CL_INDEX IS ", cl_index)
    #                 print("CLUSTER ID IS ", level_clusters[cl_index].cluster_id)
    #                 print(level_clusters[cl_index].peers)
    #                 print(level_clusters[cl_index].tree_leafs)
    #
    #                 for leaf in level_clusters[cl_index].tree_leafs:
    #                     print("LEAF CONTENT ", leaf.rootid)
    #
    #
    #
    #                     if leaf.rootid == last_cluster.cluster_id:
    #                         level_leaf_found = True
    #
    #                         # start tree traversal
    #                         while leaf.parent is not None:
    #                             # end when root is reached. Root has no parent
    #                             if leaf.rootid != leaf.parent.rootid:
    #                                 hops = hops + 1
    #                                 print("leaf root id", leaf.rootid)
    #                                 print("leaf.parent.root id ", leaf.parent.rootid)
    #                                 new_path.append(leaf.parent.rootid)
    #
    #                                 # set prev
    #                                 self.find_cluster_by_id(leaf.parent.rootid).set_previous(prev)
    #                                 prev = self.find_cluster_by_id(leaf.parent.rootid).cluster_id
    #
    #                                 # set previous cluster
    #                                 self.find_cluster_by_id(leaf.parent.rootid).set_previous_cluster(prev_cluster)
    #                                 prev_cluster = self.find_cluster_by_id(leaf.parent.rootid)
    #
    #                             # check
    #                             intersection = self.check_for_intersection(leaf.rootid, obj.path)
    #                             if intersection is not None:
    #                                 print("------------------------------INTERSECTION------------------------------ INN ",intersection)
    #                                 break
    #
    #                             leaf = leaf.parent
    #                             if leaf.parent is None:
    #                                 # found the root
    #                                 last_cluster = self.find_cluster_by_id(leaf.rootid)
    #
    #                                 # set previous for the root
    #                                 self.find_cluster_by_id(leaf.rootid).set_previous(prev)
    #                                 prev = self.find_cluster_by_id(leaf.rootid).cluster_id
    #
    #                                 # set previous cluster
    #                                 self.find_cluster_by_id(leaf.rootid).set_previous_cluster(prev_cluster)
    #                                 prev_cluster = self.find_cluster_by_id(leaf.rootid)
    #
    #                                 # set leader
    #                                 self.find_cluster_by_id(leaf.rootid).set_leader(peer_id)
    #
    #                                 # don't set leader in the tree, the tree's root cluter's leader is the leader in all levels of tree
    #                                 print("UPDATED ", last_cluster.cluster_id)
    #
    #                                 # check
    #                                 intersection = self.check_for_intersection(leaf.rootid, obj.path)
    #                                 if intersection is not None:
    #                                     print("------------------------------INTERSECTION------------------------------ INN1 ",intersection)
    #                                     break
    #
    #                         break
    #                 if level_leaf_found:
    #                     break
    #
    #             #     need to traverse all the clusters in this level containing the peer
    #
    #             level_clusters = self.clusters_by_level(i)
    #             level_cluster_size = len(self.clusters_by_level(i))
    #             for inner_cl_index in range(prev_cluster.index + 1, level_cluster_size):
    #                 if cluster.peers[0] in level_clusters[inner_cl_index].peers:
    #                     level_clusters[inner_cl_index].set_previous(prev)
    #                     prev = level_clusters[inner_cl_index].cluster_id
    #
    #                     level_clusters[inner_cl_index].set_previous_cluster(prev_cluster)
    #                     prev_cluster = level_clusters[inner_cl_index]
    #
    #                     new_path.append(level_clusters[inner_cl_index].cluster_id)
    #
    #                     last_cluster = level_clusters[inner_cl_index]
    #
    #                     # make leader
    #                     # the only peer in the cluster
    #                     level_clusters[cl_index].set_leader(peer_id)
    #
    #                     # check
    #                     intersection = self.check_for_intersection(level_clusters[inner_cl_index].cluster_id, obj.path)
    #                     if intersection is not None:
    #                         print("------------------------------INTERSECTION------------------------------ INNNN ",intersection)
    #                         break
    #
    #             print("PATH IS ", new_path)
    #
    #             i = i + 1
    #
    #         print("FINAL PATHI IS ", new_path)
    #         # cluster.object.set_path(new_path)
    #         return new_path
    #
    #
    #
    #
    #
    #
    #     # # need to consider intra cluster routing from this level
    #     # while i <= self._height_of_clusters:
    #     #     print("OK LAST CLUSTER IS ", last_cluster.cluster_id)
    #     #
    #     #     cl_index = 0
    #     #     level_clusters = self.clusters_by_level(i)
    #     #     level_cluster_size = len(self.clusters_by_level(i))
    #     #     for cl_index in range(level_cluster_size):
    #     #         print("CL_INDEX IS ", cl_index)
    #     #         print("CLUSTER ID IS ", level_clusters[cl_index].cluster_id)
    #     #         print(level_clusters[cl_index].peers)
    #     #         print(level_clusters[cl_index].tree_leafs)
    #     #
    #     #         for leaf in level_clusters[cl_index].tree_leafs:
    #     #             print("LEAF CONTENT ", leaf.rootid)
    #     #             if leaf.rootid == last_cluster.cluster_id:
    #     #
    #     #                 # start tree traversal
    #     #                 while leaf.parent is not None:
    #     #                     # end when root is reached. Root has no parent
    #     #                     if leaf.rootid != leaf.parent.rootid:
    #     #                         hops = hops + 1
    #     #                         print("leaf root id", leaf.rootid)
    #     #                         print("leaf.parent.root id ", leaf.parent.rootid)
    #     #                         path.append(leaf.parent.rootid)
    #     #
    #     #                         # set prev
    #     #                         self.find_cluster_by_id(leaf.parent.rootid).set_previous(prev)
    #     #                         prev = self.find_cluster_by_id(leaf.parent.rootid).cluster_id
    #     #
    #     #                         # set previous cluster
    #     #                         self.find_cluster_by_id(leaf.parent.rootid).set_previous_cluster(prev_cluster)
    #     #                         prev_cluster = self.find_cluster_by_id(leaf.parent.rootid)
    #     #
    #     #                     leaf = leaf.parent
    #     #                     if leaf.parent is None:
    #     #                         # found the root
    #     #                         last_cluster = self.find_cluster_by_id(leaf.rootid)
    #     #
    #     #                         # set previous for the root
    #     #                         self.find_cluster_by_id(leaf.rootid).set_previous(prev)
    #     #                         prev = self.find_cluster_by_id(leaf.rootid).cluster_id
    #     #
    #     #                         # set previous cluster
    #     #                         self.find_cluster_by_id(leaf.rootid).set_previous_cluster(prev_cluster)
    #     #                         prev_cluster = self.find_cluster_by_id(leaf.rootid)
    #     #
    #     #                         # set leader
    #     #                         self.find_cluster_by_id(leaf.rootid).set_leader(peer_id)
    #     #
    #     #                         # don't set leader in the tree, the tree's root cluter's leader is the leader in all levels of tree
    #     #                         print("UPDATED ", last_cluster.cluster_id)
    #     #
    #     #                 break
    #     #
    #     #     #     need to traverse all the clusters in this level containing the peer
    #     #
    #     #     level_clusters = self.clusters_by_level(i)
    #     #     level_cluster_size = len(self.clusters_by_level(i))
    #     #     for inner_cl_index in range(prev_cluster.index + 1, level_cluster_size):
    #     #         if cluster.peers[0] in level_clusters[inner_cl_index].peers:
    #     #             level_clusters[inner_cl_index].set_previous(prev)
    #     #             prev = level_clusters[inner_cl_index].cluster_id
    #     #
    #     #             level_clusters[inner_cl_index].set_previous_cluster(prev_cluster)
    #     #             prev_cluster = level_clusters[inner_cl_index]
    #     #
    #     #             path.append(level_clusters[inner_cl_index].cluster_id)
    #     #
    #     #             last_cluster = level_clusters[inner_cl_index]
    #     #
    #     #             # make leader
    #     #             # the only peer in the cluster
    #     #             level_clusters[cl_index].set_leader(peer_id)
    #     #
    #     #             print("FOUND ", cluster.peers[0])
    #     #
    #     #     print("PATH IS ", path)
    #     #
    #     #     i = i + 1
    #     #
    #     # print("FINAL PATHI IS ", path)
    #     # cluster.object.set_path(path)
    #     # return path
    #     print("FINISHING MOVE")




