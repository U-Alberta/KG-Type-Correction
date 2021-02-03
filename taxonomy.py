import json
from itertools import accumulate


class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.level = 0
        self.children = []
    
    def add_child(self, child):
        self.children.append(child)
        child.parent = self
        child.update_level()
    
    def update_level(self):
        self.level = self.parent.level + 1
        for c in self.children:
            c.update_level()
    
    def get_all_parents(self):
        node = self
        result = []
        while node is not None:
            result.append(node.name)
            node = node.parent
        return list(reversed(result))
    
    def get_type_path(self, level):
        path = self.get_all_parents()[:level]
        if len(path) < level:
            for _ in range(level - len(path)):
                path.append('Unspecified')
        return list(accumulate(path, lambda x, y: x + '/' + y))

    @staticmethod
    def build_taxonomy():
        nodes = {}
        with open('data/dbpedia_taxonomy.json') as f:
            t_to_parent = json.load(f)
        t_to_parent = {k[4:]: v[4:] for k, v in t_to_parent.items()}  # remove prefix
        for child_type, parent_type in t_to_parent.items():
            nodes[parent_type] = Node(parent_type)
            nodes[child_type] = Node(child_type)

        for child_type, parent_type in t_to_parent.items():
            nodes[parent_type].add_child(nodes[child_type])
        return nodes

def get_taxonomy():
    return Node.build_taxonomy()
