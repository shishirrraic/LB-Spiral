import matplotlib.pyplot as plt
import networkx as nx
import log

logger = log.get_logger(__name__)


def check_for_intersection(cluster_id, path):
    logger.debug("PATH {}".format(path))
    logger.debug("CLUSTER ID {}".format(cluster_id))
    if cluster_id in path:
        return cluster_id
    return None


class Network:
    """
    The network class
    Attributes:
    ----------
        network_graph (Networkx Graph): the network's underlying graph
        clusters (list): clusters in the network
        peers (list): peers in the network
        height_of_clusters (int): height of the clusters
        root_cluster_id (str): root cluster
        size (int): network's size
    """

    def __init__(self, graph):
        self.__network_graph = graph
        self.__clusters = {}
        self.__peers = {}
        self.__height_of_clusters = None
        self.__root_cluster_id = None
        self.__size = nx.number_of_nodes(graph)

    def add_peer(self, peer_id, peer):
        self.__peers[peer_id] = peer

    def add_cluster(self, level, cluster):
        cluster_size = len(self.__clusters[int(level)])
        cluster.set_index(cluster_size)
        self.__clusters[int(level)].append(cluster)

    def add_cluster_level(self, level):
        self.__clusters[int(level)] = []

    def set_height_of_clusters(self, height):
        self.__height_of_clusters = height

    def get_graph(self):
        return self.__network_graph

    def size(self):
        return self.__size

    def cluster_by_id(self, cluster_id):
        return self.__clusters[cluster_id]

    def clusters_by_level(self, level):
        logger.info("LEVEL IS {}".format(level))
        logger.info("CLUSTERS IN LEVEL IS {}".format(self.__clusters[level]))
        return [a for a in self.__clusters[level] if not a.intermediate()]

    def find_cluster_by_id(self, cluster_id):
        for level in self.__clusters:
            for cluster in self.__clusters[level]:
                if cluster_id == cluster.get_cluster_id():
                    return cluster
        return None

    def print_cluster_by_level(self, level):
        for cl in self.__clusters[level]:
            if not cl.intermediate():
                logger.debug("CLUSTER ID {} and nodes contained {}, leader is {}".format(cl.get_cluster_id(),
                                                                                         cl.get_graph().nodes,
                                                                                         cl.get_leader()))

    def print_intermediate_clusters_by_level(self, level):
        for cl in self.__clusters[level]:
            if cl.intermediate():
                logger.debug(
                    "INTERMEDIATE CLUSTER ID {} and nodes contained {}, leader is {}".format(cl.get_cluster_id(),
                                                                                             cl.get_graph().nodes,
                                                                                             cl.get_leader()))
                logger.debug("PEERS ", cl.get_peers())

    def remove_cluster_by_id(self, cluster_id, level):
        temp = []
        for cluster in self.__clusters[level]:
            if cluster.__cluster_id is not cluster_id:
                temp.append(cluster)
        self.__clusters = temp

    def draw_cluster(self, cluster_id):
        print("DRAWING CLUSTER ", cluster_id)
        cluster = self.find_cluster_by_id(cluster_id)
        node_pos = nx.spring_layout(self.__network_graph)
        edge_weight = nx.get_edge_attributes(self.__network_graph, 'weight')
        # Draw the nodes
        nx.draw_networkx_nodes(self.__network_graph, pos=node_pos, nodelist=[str(i) for i in cluster.__graph])
        nx.draw_networkx(self.__network_graph, node_pos, node_color='grey', node_size=100)
        # Draw the edges
        nx.draw_networkx_edges(self.__network_graph, node_pos, edge_color='black')
        # Draw the edge labels
        nx.draw_networkx_edge_labels(self.__network_graph, node_pos, edge_labels=edge_weight)
        plt.title(cluster_id)
        plt.show()
        graph = cluster.__graph
        print("THE GRAPH IS ", graph)

    def publish(self, cluster, obj):
        cluster.publish(obj)
        assert cluster.get_object() is not None

        # used to set leader
        peer_id = cluster.get_peers()[0]

        i = 1
        hops = 0
        level_clusters = self.clusters_by_level(i)
        level_clusters_size = len(self.clusters_by_level(i))
        path = []
        prev = cluster.get_cluster_id()
        prev_cluster = cluster
        path.append(prev)

        # last cluster to visit in the level
        last_cluster = None

        for cl_index in range(level_clusters_size):
            if cluster.get_peers()[0] in level_clusters[cl_index].get_peers():
                hops = hops + 1
                level_clusters[cl_index].set_previous(prev)
                prev = level_clusters[cl_index].get_cluster_id()

                level_clusters[cl_index].set_previous_cluster(prev_cluster)
                prev_cluster = level_clusters[cl_index]

                path.append(level_clusters[cl_index].get_cluster_id())

                last_cluster = level_clusters[cl_index]

                # make leader
                # the only peer in the cluster
                level_clusters[cl_index].set_leader(peer_id)
        i = i + 1

        # need to consider intra cluster routing from this level
        while i <= self.__height_of_clusters:

            cl_index = 0
            level_clusters = self.clusters_by_level(i)
            level_clusters_size = len(self.clusters_by_level(i))

            for cl_index in range(level_clusters_size):
                level_leaf_found = False

                for leaf in level_clusters[cl_index].get_tree_leaves():
                    if leaf.get_root_id() == prev_cluster.get_cluster_id():
                        level_leaf_found = True
                        # Start tree traversal
                        while leaf.get_parent() is not None:
                            # end when root is reached. Root has no parent
                            if leaf.get_root_id() != leaf.get_parent().get_root_id():
                                hops = hops + 1
                                path.append(leaf.get_parent().get_root_id())

                                # set previous
                                self.find_cluster_by_id(leaf.get_parent().get_root_id()).set_previous(prev)
                                prev = self.find_cluster_by_id(leaf.get_parent().get_root_id()).get_cluster_id()

                                # set previous cluster
                                self.find_cluster_by_id(leaf.get_parent().get_root_id()) \
                                    .set_previous_cluster(prev_cluster)
                                prev_cluster = self.find_cluster_by_id(leaf.get_parent().get_root_id())

                            leaf = leaf.get_parent()

                            if leaf.get_parent() is None:
                                # set leader
                                self.find_cluster_by_id(leaf.get_root_id()).set_leader(peer_id)
                                if i == self.__height_of_clusters:
                                    self.__root_cluster_id = leaf.get_root_id()
                                    break

                                # found the root
                                prev_cluster = self.find_cluster_by_id(leaf.get_root_id())

                                # set previous for the root
                                self.find_cluster_by_id(leaf.get_root_id()).set_previous(prev)
                                prev = self.find_cluster_by_id(leaf.get_root_id()).get_cluster_id()

                        break
                if level_leaf_found:
                    break

            # after the tree traversal, need to traverse all the clusters in this level
            # containing the peer

            level_clusters = self.clusters_by_level(i)
            level_cluster_size = len(self.clusters_by_level(i))
            for inner_cl_index in range(prev_cluster.get_index() + 1, level_cluster_size):
                if cluster.get_peers()[0] in level_clusters[inner_cl_index].get_peers():
                    hops = hops + 1
                    level_clusters[inner_cl_index].set_previous(prev)
                    prev = level_clusters[inner_cl_index].get_cluster_id()

                    level_clusters[inner_cl_index].set_previous_cluster(prev_cluster)
                    prev_cluster = level_clusters[inner_cl_index]

                    path.append(level_clusters[inner_cl_index].get_cluster_id())

                    prev_cluster = level_clusters[inner_cl_index]

                    # make leader, the only peer in the cluster
                    level_clusters[cl_index].set_leader(peer_id)
            i = i + 1

        logger.info("FINAL PATH {}".format(path))
        cluster.get_object().set_path(path)

        logger.info("HOPS TOTAL = {}".format(hops))
        return path

    def move(self, cluster, obj):
        logger.debug("THE ORIGINAL PATH IS {}".format(obj.get_path()))

        owner_id = self.find_cluster_by_id(obj.get_owner()).get_id()
        mover_id = self.find_cluster_by_id(cluster.get_cluster_id()).get_id()
        res = self.find_intersection(cluster, obj, owner_id, mover_id)
        assert res['intersection'] is not None
        assert res['path'] is not None
        assert res['hops'] is not None

        logger.debug('intersection at {}'.format(res['intersection']))
        logger.debug('intersection path at {}'.format(res['path']))
        logger.debug('hops count {}'.format(res['hops']))

        # inform the leader in each hop
        inform_cost = 0

        for path in res['path']:
            logger.debug("PATH INSIDE INFORM COST CALCULATION IS {}".format(path))
            # if the mover node is not the leader in the cluster it should inform the cluster so calculate the
            # distance form the node to the leader.
            logger.debug("MOVER ID {}".format(mover_id))

            if not self.find_cluster_by_id(path).intermediate():
                if mover_id == self.find_cluster_by_id(path).get_leader():
                    logger.debug("SAME")
                else:
                    inform_cost = inform_cost + nx.dijkstra_path_length(
                        self.find_cluster_by_id(path).get_graph(),
                        mover_id,
                        self.find_cluster_by_id(path).get_leader(),
                        'weight'
                    )
                    logger.debug("UPDATED INFORM COST {}".format(inform_cost))
            self.find_cluster_by_id(path).set_leader(mover_id)

        logger.debug("INFORM COST IS {}".format(inform_cost))

        # need to update the path
        # need to delete pointers of the previous owner
        logger.debug("THE INDEX TO DELETE FROM {}".format(obj.get_path().index(res['intersection'])))
        index_to_delete_from = obj.get_path().index(res['intersection']) - 1

        cluster_to_delete_path_from = self.find_cluster_by_id(obj.get_path()[index_to_delete_from])
        tmp_cluster = self.find_cluster_by_id(obj.get_path()[index_to_delete_from])
        while tmp_cluster is not None:
            logger.debug("{}".format(tmp_cluster.get_cluster_id()))
            tmp_cluster = tmp_cluster.get_previous_cluster()
        delete_hop = 0

        while cluster_to_delete_path_from is not None:
            temp_cluster = cluster_to_delete_path_from.get_previous_cluster()
            cluster_to_delete_path_from.set_previous_cluster(None)
            cluster_to_delete_path_from = temp_cluster
            delete_hop = delete_hop + 1

        tmp_cluster = self.find_cluster_by_id(obj.get_path()[index_to_delete_from])
        while tmp_cluster is not None:
            tmp_cluster = tmp_cluster.get_previous_cluster()

        # check for prevuos cluster links, DELETE CHECK
        for i in range(index_to_delete_from + 1):
            logger.debug("{}".format(self.find_cluster_by_id(obj.get_path()[i]).get_cluster_id()))
            assert self.find_cluster_by_id(obj.get_path()[i]).get_previous_cluster() is None

        # set previous cluster in the new path
        logger.debug("PRINTING PATHS")
        res['path'].reverse()
        for i in range(0, len(res['path']) - 1):
            logger.debug("{}".format(res['path'][i]))
            if res['path'][i + 1] is not None:
                self.find_cluster_by_id(res['path'][i]).set_previous_cluster(
                    self.find_cluster_by_id(res['path'][i + 1]))

        # calculate the processing load
        logger.debug("PROCESSING LOAD")
        processing_load = 0
        for j in range(0, len(res['path'])):
            if mover_id in self.find_cluster_by_id(res['path'][j]).get_peers():
                processing_load = processing_load + 1
            logger.debug("{}".format(res['path'][j]))

        logger.debug("PROCESSING LOAD IS {}".format(processing_load))
        # change the ownership equivalent to moving the file
        obj.set_owner(cluster.get_cluster_id())

        # update the object's path
        root_cluster = self.find_cluster_by_id(self.__root_cluster_id)

        path = [root_cluster.get_cluster_id()]
        while root_cluster.get_previous_cluster() is not None:
            root_cluster = root_cluster.get_previous_cluster()
            path.append(root_cluster.get_cluster_id())
        obj.set_path(list(reversed(path)))

        logger.debug("NEW OBJECT PATH IS {}".format(obj.get_path()))

        # need to calculate the shortest path from source to node in the intersected cluster
        logger.debug(
            "THE INTERSECTED CLUSTER ID IS {}".format(self.find_cluster_by_id(res['intersection']).get_cluster_id()))
        logger.debug("MOVER ID {}".format(mover_id))
        logger.debug("OWNER ID {}".format(owner_id))
        logger.debug("NODES {}".format(self.find_cluster_by_id(res['intersection']).get_graph().nodes))
        logger.debug("SHORTEST PATH LENGTH {}".format(nx.dijkstra_path_length(
            self.find_cluster_by_id(res['intersection']).get_graph(), mover_id, owner_id, 'weight')))
        logger.debug('DELETE HOP COUNT {}'.format(delete_hop))

        logger.debug("MOVE DONE")

        res['delete_hops'] = delete_hop
        res['shortest_path_length_in_intersected_cluster'] = nx.dijkstra_path_length(
            self.find_cluster_by_id(res['intersection']).get_graph(), mover_id, owner_id, 'weight')
        # res['LB_SPIRAL_cost'] = res['hops'] + res['delete_hops'] + res['shortest_path_length_in_intersected_cluster']
        # res['LB_SPIRAL_cost'] = res['hops'] + res['shortest_path_length_in_intersected_cluster'] + inform_cost
        res['LB_SPIRAL_cost'] = res['hops'] + inform_cost * 2
        res['inform_cost_only'] = inform_cost * 2
        res['hopsOnly'] = res['hops']
        res['shortest_path_length_in_intersected_cluster'] = res['shortest_path_length_in_intersected_cluster']
        res['processing_load'] = processing_load
        return res

    def find_intersection(self, cluster, obj, owner_id, mover_id):
        logger.debug("OBJECTS PATH IS {}".format(obj.get_path()))

        # keep track of level with i
        i = 1
        hops = 0
        level_clusters = self.clusters_by_level(i)
        level_cluster_size = len(self.clusters_by_level(i))
        new_path = []
        prev = cluster.get_cluster_id()
        prev_cluster = cluster

        new_path.append(prev)

        # last cluster to visit in the level
        intersection = False
        while 1:
            for cl_index in range(level_cluster_size):

                if cluster.get_peers()[0] in level_clusters[cl_index].get_peers():
                    hops = hops + 1
                    level_clusters[cl_index].set_previous(prev)
                    prev = level_clusters[cl_index].get_cluster_id()

                    prev_cluster = level_clusters[cl_index]
                    new_path.append(level_clusters[cl_index].get_cluster_id())

                    intersection = check_for_intersection(level_clusters[cl_index].get_cluster_id(), obj.get_path())

                    if intersection is not None and owner_id in level_clusters[cl_index].get_graph().nodes and\
                            mover_id in level_clusters[cl_index].get_graph().nodes:
                        d = dict()
                        d['intersection'] = intersection
                        d['path'] = new_path
                        d['hops'] = hops
                        return d

            i = i + 1

            # need to consider intra cluster routing from this level
            while i <= self.__height_of_clusters:
                cl_index = 0
                level_clusters = self.clusters_by_level(i)
                level_cluster_size = len(self.clusters_by_level(i))

                for cl_index in range(level_cluster_size):
                    level_leaf_found = False

                    for leaf in level_clusters[cl_index].get_tree_leaves():
                        if leaf.get_root_id() == prev_cluster.get_cluster_id():
                            level_leaf_found = True
                            # start tree traversal
                            while leaf.get_parent() is not None:
                                # end when root is reached. Root has no parent
                                if leaf.get_root_id() != leaf.get_parent().get_root_id():
                                    hops = hops + 1

                                    # set prev
                                    self.find_cluster_by_id(leaf.get_parent().get_root_id()).set_previous(prev)
                                    prev = self.find_cluster_by_id(leaf.get_parent().get_root_id()).get_cluster_id()

                                    # set previous cluster
                                    prev_cluster = self.find_cluster_by_id(leaf.get_parent().get_root_id())

                                    new_path.append(leaf.get_parent().get_root_id())

                                leaf = leaf.get_parent()
                                if leaf.get_parent() is None:
                                    if i == self.__height_of_clusters:
                                        self.__root_cluster_id = leaf.get_root_id()

                                        intersection = check_for_intersection(leaf.get_root_id(), obj.get_path())
                                        if intersection is not None and owner_id in self.find_cluster_by_id(
                                                leaf.get_root_id()).get_graph().nodes and mover_id in self.find_cluster_by_id(
                                            leaf.get_root_id()).get_graph().nodes:
                                            d = dict()
                                            d['intersection'] = intersection
                                            d['path'] = new_path
                                            d['hops'] = hops
                                            logger.debug("FOUND AND RETURNING ROOT")
                                            return d
                                        break
                                    # found the root
                                    prev_cluster = self.find_cluster_by_id(leaf.get_root_id())

                                    # set previous for the root
                                    self.find_cluster_by_id(leaf.get_root_id()).set_previous(prev)
                                    prev = self.find_cluster_by_id(leaf.get_root_id()).get_cluster_id()

                                    intersection = check_for_intersection(leaf.get_root_id(), obj.get_path())
                                    if intersection is not None and owner_id in self.find_cluster_by_id(
                                            leaf.get_root_id()).get_graph().nodes and \
                                            mover_id in self.find_cluster_by_id(leaf.get_root_id()).get_graph().nodes:
                                        d = dict()
                                        d['intersection'] = intersection
                                        d['path'] = new_path
                                        d['hops'] = hops
                                        return d
                            break
                    if level_leaf_found:
                        break

                # need to traverse all the clusters in this level containing the peer
                level_clusters = self.clusters_by_level(i)
                level_cluster_size = len(self.clusters_by_level(i))
                for inner_cl_index in range(prev_cluster.get_index() + 1, level_cluster_size):
                    if cluster.get_peers()[0] in level_clusters[inner_cl_index].get_peers():
                        hops = hops + 1
                        level_clusters[inner_cl_index].set_previous(prev)
                        prev = level_clusters[inner_cl_index].get_cluster_id()

                        # level_clusters[inner_cl_index].set_previous_cluster(prev_cluster)
                        prev_cluster = level_clusters[inner_cl_index]

                        new_path.append(level_clusters[inner_cl_index].get_cluster_id())

                        intersection = check_for_intersection(level_clusters[inner_cl_index].get_cluster_id(),
                                                              obj.get_path())
                        if intersection is not None and owner_id in level_clusters[inner_cl_index].get_graph().nodes and \
                                mover_id in level_clusters[inner_cl_index].graph.nodes:
                            d = dict()
                            d['intersection'] = intersection
                            d['path'] = new_path
                            d['hops'] = hops
                            return d

                i = i + 1

            # No match found until root, so intersection is at root
            logger.debug("FINAL PATH IS {}".format(new_path))
            logger.debug("{}".format(self.find_cluster_by_id(self.__root_cluster_id).get_peers()))
            logger.debug("{}".format(cluster.get_cluster_id()))
            logger.debug("INTERSECTION {}".format(intersection))
            # cluster.get_object().set_path(new_path)

            obj.set_path(new_path)
            d = dict()
            d['intersection'] = intersection
            d['path'] = new_path
            d['hops'] = hops
            logger.debug("HOPS TOTAL MOVE = {}".format(hops))
            return d

