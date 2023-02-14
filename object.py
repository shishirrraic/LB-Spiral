class Object:
    """
    An object in the network

    Attributes:
        id (str): object id
        owner_id (str): object owner id
        path (list): object path in the network
    """

    def __init__(self, id, owner_id):
        self.__id = id
        self.__owner_id = owner_id
        self.__path = None

    def get_id(self):
        return self.__id

    def get_owner(self):
        return self.__owner_id

    def get_path(self):
        return self.__path

    def set_owner(self, owner_id):
        self.__owner_id = owner_id

    def set_path(self, path):
        self.__path = path