class Peer:
    """
        A Peer in the network

        Attributes:
            peer_id (str) : peer id
            neighbors (list) : neighbors list
            objects (list) : objects list
        """

    def __init__(self, peer_id, neighbors):
        self.__peer_id = peer_id
        self.__neighbors = neighbors
        self.__objects = []

    def get_peer_id(self):
        return self.__peer_id

    def add_object(self, object):
        self.__objects.append(object)

    def lookup(self, object):
        pass

    def publish(self, object_id):
        pass

    def move(self, object_id):
        pass


class Object:
    """
        Object to move
    """

    def __init__(self, object_id, owner_id):
        self.__object_id = object_id
        self.__object_id = owner_id
