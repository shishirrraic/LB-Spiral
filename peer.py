class Peer:
    """A node in the network"""
    def __init__(self, peer_id, neighbors):
        """Initilaize the node
        id: id of the node
        """
        self.peerId = peer_id
        self.neighbors = neighbors
        self.objects = []

    def get_peer_id(self):
        """
        :return: the name of the miner.
        """
        return self.peerId

    def add_object(self, object):
        self.objects.append(object);

    def lookup(self, object_id):
        self

    def publish(self, object):
        self

    def move(self, object):
        self

class Object:
    """An object"""
    def __init__(self, object_id, owner_id):
        self.object_id = object_id
        self.owner_id = owner_id
