import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout

logger = log.get_logger(__name__)


class BinaryTree:
    """
    A class to hold a binary tree.
    Attributes
    ----------
    left : node
        left node of the tree
    right : node
        right node of the tree
    root_id : str
        root of the tree
    data: lst
        data the tree holds
    parent: parent of the tree
    """

    def __init__(self, root_id):
        self.__left = None
        self.__right = None
        self.__root_id = root_id
        self.__data = None
        self.__parent = None

    def set_left_child(self, left):
        self.__left = left

    def set_right_child(self, right):
        self.__right = right

    def get_left_child(self):
        return self.__left

    def get_right_child(self):
        return self.__right

    def set_node_value(self, value):
        self.__root_id = value

    def get_node_value(self):
        return self.__root_id

    def get_root_id(self):
        return self.__root_id

    def insert_right(self, new_node):
        if self.__right is None:
            self.__right = BinaryTree(new_node)
        else:
            tree = BinaryTree(new_node)
            tree.__right = self.__right
            self.__right = tree

    def insert_left(self, new_node):
        if self.__left is None:
            self.__left = BinaryTree(new_node)
        else:
            tree = BinaryTree(new_node)
            tree.__left = self.__left
            self.__left = tree

    def set_data(self, data):
        self.__data = data

    def get_data(self):
        return self.__data

    def set_parent(self, node):
        self.__parent = node

    def get_parent(self):
        return self.__parent

    def print(self):
        if self is not None:
            print_tree(self.get_left_child())
            print(self.get_node_value())
            print_tree(self.get_right_child())

    def display_tree(self):
        G = nx.Graph()
        self.iterative_preorder(G)
        plt.title('Binary tree ' + self.__root_id)
        nx.draw_networkx(G)
        pos = graphviz_layout(G, prog='dot')
        nx.draw(G, pos, with_labels=False, arrows=True)
        plt.show()

    def iterative_preorder(self, G):
        """
        Iterative preorder traversal of the G
        :param G: networkx Graph
        :return: None
        """
        # Base Case
        if self.__root_id is None:
            return
        # create an empty stack and push root to it
        node_stack = [self]

        # A) Pop all items one by one and for every popped item
        #   1) print
        #   2) push its right child
        #   3) push its left child
        # Right child pushed first to work with left at first from the stack

        while len(node_stack) > 0:
            # Pop the top item from stack and print it
            node = node_stack.pop()
            G.add_node(node.__root_id)

            # Push right and left children of popped node
            if node.__right is not None:
                node_stack.append(node.__right)
                G.add_node(node.__right.__root_id)
                G.add_edge(node.__root_id, node.__right.__root_id)

            if node.__left is not None:
                node_stack.append(node.__left)
                G.add_node(node.__left)
                G.add_edge(node.__root_id, node.__left.__root_id)


def print_tree(tree):
    if tree is not None:
        print_tree(tree.get_left_child())
        if tree.get_parent() is None:
            print('...', tree.get_node_value(), '...')
        else:
            print(tree.get_node_value())
        print_tree(tree.get_right_child())


def test_tree():
    tree = BinaryTree("Root")
    tree.insert_left("depth1LNode")
    tree.insert_right("depth1RNode")
    tree.insert_right("depth1RNode1")
    print_tree(tree)


def get_leaves(tree):
    leaf_nodes = []
    find_leaves(tree, leaf_nodes)
    print("LEAVES ARE ", leaf_nodes)
    return leaf_nodes


def find_leaves(tree, leaf_nodes):
    if tree is None:
        return leaf_nodes

    if tree.get_left_child() is None and tree.get_right_child() is None:
        leaf_nodes.append(tree)

    if tree.get_left_child() is not None:
        find_leaves(tree.get_left_child(), leaf_nodes)

    if tree.get_left_child() is not None:
        find_leaves(tree.get_right_child(), leaf_nodes)
