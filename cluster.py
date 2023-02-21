import networkx as nx
import log

logger = log.get_logger(__name__)


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
        nodes = sorted(list(graph.nodes))
        for node in nodes:
            self.__peers.append(node)
        self.__object = None
        self.__object_path = None
        if leader_id is not None:
            logger.debug("SETTING LEADER {} FOR CLUSTER {}".format(leader_id, self.__cluster_id))
            self.__leader_id = leader_id
        else:
            self.__leader_id = self.__id

        self.__root = False
        self.__previous = None
        self.__previous_cluster = None
        self.__intermediate = False
        self.__index = None

    def set_root(self):
        self.__root = True

    def get_graph(self):
        return self.__graph

    def get_leader(self):
        return self.__leader_id

    def get_object(self):
        return self.__object

    def set_index(self, index):
        self.__index = index

    def get_index(self):
        return self.__index

    def get_cluster_id(self):
        return self.__cluster_id

    def get_id(self):
        return self.__id

    def get_peers(self):
        return self.__peers

    def set_tree(self, tree):
        self.__tree = tree

    def set_previous(self, cluster_id):
        self.__previous = cluster_id

    def set_previous_cluster(self, cluster):
        self.__previous_cluster = cluster

    def get_previous_cluster(self):
        return self.__previous_cluster

    def set_leader(self, leader_id):
        self.__leader_id = leader_id

    def set_intermediate(self):
        self.__intermediate = True

    def intermediate(self):
        return self.__intermediate

    def get_level(self):
        return self.__level

    def print(self):
        logger.debug("Cluster id is {}".format(self.__cluster_id))
        logger.debug("Nodes are {}".format(self.__graph.nodes))
        if self.__tree is not None:
            self.__tree.print()
        else:
            logger.debug("No Tree")

    def set_tree_leaves(self, leaves):
        self.__tree_leaves = leaves

    def get_tree_leaves(self):
        return self.__tree_leaves

    def send(self, object):
        pass

    def receive(self, object):
        pass

    def publish(self, obj):
        assert (self.__level == 0)
        obj.set_owner(self.__cluster_id)
        self.__object = obj

    def move(self, object):
        assert (self.__level == 0)

    def lookup(self, object):
        assert (self.__level == 0)
