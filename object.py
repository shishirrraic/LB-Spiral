class Object:
    """An object in the network"""

    def __init__(self, id, owner_id):
        self.id = id
        self.owner_id = owner_id
        self.path = None

    def set_owner(self, owner_id):
        self.owner_id = owner_id

    def set_path(self, path):
        self.path = path
