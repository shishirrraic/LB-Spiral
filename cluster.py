from networkx import nx
from binaryTree import BinaryTree
from object import Object

class Cluster:
    """A node in the network"""

    def __init__(self, cluster_id, graph, level, leader_id = None):
        """Initilaize the node
        id: id of the node
        """
        self.cluster_id = cluster_id
        self.graph = nx.Graph(graph)
        self.level = level
        self.peers = []
        self.tree = None
        self.tree_leafs = []
        self.id = cluster_id.split("_")[0].split("c")[1]
        self.index = -1

        for node in graph.nodes:
            self.peers.append(node)
        self.object = None
        self.object_path = None
        self.parent_clusters = []
        if leader_id is not None:
            print("SETTING LEADER ", leader_id, "FOR CLUSTER", self.cluster_id)
            self.leader_id = leader_id
        else:
            self.leader_id = self.id

        self.root = False
        self.previous = None
        self.previous_cluster = None
        self.next = None
        self.between = False
        self.intermediate = False

    def get_peer_group_id(self):
        """
        :return: the name of the miner.
        """
        return self.cluster_id

    def set_data(self, clusters):
        self.data = clusters

    def set_tree(self, tree):
        self.tree = tree

    def set_previous(self, cluster_id):
        self.previous = cluster_id

    def set_previous_cluster(self, cluster):
        self.previous_cluster = cluster

    def set_leader(self, leader_id):
        self.leader_id = leader_id

    def set_intermediate(self):
        self.intermediate = True
    def print(self):
        print("Cluster id is ", self.cluster_id)
        print("Nodes are ", self.graph.nodes)
        print("Index is ", self.index)
        if self.tree != None:
            self.tree.print()
        else:
            print("NO TREE")

    def setIndex(self, i):
        self.index = i

    def set_parent_clusters(self, parent_clusters):
        self.parent_clusters = parent_clusters

    def set_tree_leafs(self, leafs):
        self.tree_leafs = leafs

    def send(self, object):
        self

    def receive(self, object):
        self

    def publish(self, obj):
        assert (self.level == 0)
        # todo object id
        obj.set_owner(self.cluster_id)
        self.object = obj

        self

    def move(self, object):
        assert (self.level == 0)
        self

    def lookup(self, object):
        assert (self.level == 0)
        self

