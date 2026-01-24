class Node:
    def __init__(self, p_maxn: int, p_d: int, leaf=None):
        self.leaf = leaf
        self.left = None  
        self.right = None  
        self.id = None  
        self.split_feature = None
        self.split_point = None
        self.gh = None
        self.selected_bin = None
        self.is_data = None  
        self.maxn = p_maxn  
        self.d = p_d  
    def is_leaf(self):
        return self.leaf is not None