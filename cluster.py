import networkx as nx
import logging

class Cluster:
    """
    A cluster in the network

    Attributes:
        cluster_id (str): cluster id
        graph (networkx graph): graph for the cluster
        level (int): level of the cluster
        peers (list): list of peers in the cluster
        tree (object): binary tree for the cluster
        tree_leaves (list): leaves of the tree
        id (str): id parsed from cluster_id
        object (object): object to work with
        object_path (list): object's path
        leader_id (str): leader of the cluster
        root (boolean): root or not
        previous (str): previous cluster id
        previous_cluster (object): previous cluster
        intermediate (boolean): intermediate cluster or not
    """

    def __init__(self, cluster_id, graph, level, leader_id=None):
        """
        :param cluster_id: The cluster id
        :param graph: The graph
        :param level: cluster level
        :param leader_id: leader of the cluster
        """
        self.__cluster_id = cluster_id
        self.__graph = nx.Graph(graph)
        self.__level = level
        self.__peers = []
        self.__tree = None
        self.__tree_leaves = []
        self.__id = cluster_id.split("_")[0].split("c")[1]

        for node in graph.nodes:
            self.__peers.append(node)
        self.__object = None
        self.__object_path = None
        if leader_id is not None:
            logging.info("SETTING LEADER %s FOR CLUSTER %s", leader_id, self.__cluster_id)
            self.__leader_id = leader_id
        else:
            self.__leader_id = self.__id

        self.__root = False
        self.__previous = None
        self.__previous_cluster = None
        self.__intermediate = False
        self.__data = None

    def set_tree(self, tree):
        self.__tree = tree

    def set_previous(self, cluster_id):
        self.__previous = cluster_id

    def set_previous_cluster(self, cluster):
        self.__previous_cluster = cluster

    def set_leader(self, leader_id):
        self.__leader_id = leader_id

    def set_intermediate(self):
        self.__intermediate = True

    def print(self):
        logging.info("Cluster id is %s", self.__cluster_id)
        logging.info("Nodes are %s", self.__graph.nodes)
        if self.__tree is not None:
            self.__tree.print()
        else:
            logging.info("No Tree")

    def set_tree_leaves(self, leaves):
        self.__tree_leaves = leaves

    def send(self, object):
        pass

    def receive(self, object):
        pass

    def publish(self, obj):
        assert(self.__level == 0)
        obj.set_owner(self.__cluster_id)
        self.__object = obj

    def move(self, object):
        assert (self.__level == 0)

    def lookup(self, object):
        assert(self.__level == 0)





