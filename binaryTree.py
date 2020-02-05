from networkx import nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
class BinaryTree():
    def __init__(self,rootid):
      self.left = None
      self.right = None
      self.rootid = rootid
      self.data = None
      self.parent = None

    def getLeftChild(self):
        return self.left
    def getRightChild(self):
        return self.right
    def setNodeValue(self,value):
        self.rootid = value
    def getNodeValue(self):
        return self.rootid

    def insertRight(self,newNode):
        if self.right == None:
            self.right = BinaryTree(newNode)
        else:
            tree = BinaryTree(newNode)
            tree.right = self.right
            self.right = tree

    def insertLeft(self,newNode):
        if self.left == None:
            self.left = BinaryTree(newNode)
        else:
            tree = BinaryTree(newNode)
            tree.left = self.left
            self.left = tree

    def set_data(self, data):
        self.data = data

    def get_data(self):
        return self.data

    def set_parent(self, node):
        self.parent = node

    def print(self):
        if self != None:
            printTree(self.getLeftChild())
            print(self.getNodeValue())
            printTree(self.getRightChild())

    def display_tree(self):
        G = nx.Graph()
        self.iterative_preorder(G)
        # write dot file to use with graphviz
        # run "dot -Tpng test.dot >test.png"
        # write_dot(G, 'test.dot')

        # same layout using matplotlib with no labels
        plt.title('Binary tree ' + self.rootid)
        nx.draw_networkx(G)
        # pos = graphviz_layout(G)
        pos = graphviz_layout(G, prog='dot')
        nx.draw(G, pos, with_labels=False, arrows=True)
        plt.show()
        # plt.savefig('nx_test.png')

    def iterative_preorder(self, G):

        # Base CAse
        if self.rootid is None:
            return

            # create an empty stack and push root to it
        nodeStack = []
        nodeStack.append(self)

        #  Pop all items one by one. Do following for every popped item
        #   a) print it
        #   b) push its right child
        #   c) push its left child
        # Note that right child is pushed first so that left
        # is processed first */
        while (len(nodeStack) > 0):

            # Pop the top item from stack and print it
            node = nodeStack.pop()
            G.add_node(node.rootid)
            # print node.data,

            # Push right and left children of the popped node
            # to stack
            if node.right is not None:
                nodeStack.append(node.right)

                G.add_node(node.right.rootid)
                G.add_edge(node.rootid, node.right.rootid)
            if node.left is not None:
                nodeStack.append(node.left)

                G.add_node(node.left)
                G.add_edge(node.rootid, node.left.rootid)

def printTree(tree):
        if tree != None:
            printTree(tree.getLeftChild())
            if tree.parent is None:
                print('--------------------------',tree.getNodeValue(),'-----------------------')
            else:
                print(tree.getNodeValue())
            printTree(tree.getRightChild())

def testTree():
    myTree = BinaryTree("Maud")
    myTree.insertLeft("Bob")
    myTree.insertRight("Tony")
    myTree.insertRight("Steven")
    printTree(myTree)

def get_leafs(tree):
    leaf_nodes = []
    find_leafs(tree, leaf_nodes)
    print("LEAFS ARE ", leaf_nodes)
    return leaf_nodes


def find_leafs(tree, leaf_nodes):
    if tree is None:
        return leaf_nodes

    if tree.left is None and tree.right is None:
        leaf_nodes.append(tree)

    if tree.left is not None:
        find_leafs(tree.left, leaf_nodes)

    if tree.left is not None:
        find_leafs(tree.right, leaf_nodes)





