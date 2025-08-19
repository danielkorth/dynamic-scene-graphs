import numpy as np


class Node:
    def __init__(self, 
                 name: str, 
                 centroid: np.ndarray, 
                 color: np.ndarray = np.array([0, 0, 0]),
                 radius: float = 0.003,
                 label: str = "",
                 pct: np.ndarray | None= None):
        self.name = name
        self.id = int(name.split("_")[-1])
        self.centroid = centroid
        self.color = color
        self.radius = radius
        self.label = label
        self.pct = pct
        self.clip_features = None
    
    @property
    def centroid(self):
        return self._centroid
    
    @centroid.setter
    def centroid(self, value: np.ndarray):
        self._centroid = value
    
    @property
    def color(self):
        return self._color
    
    @color.setter
    def color(self, value: np.ndarray):
        self._color = value
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value: float):
        self._radius = value
    
    @property
    def pct(self):
        return self._pct
    
    @pct.setter
    def pct(self, value: np.ndarray):
        self._pct = value
    
    def add_pct(self, pct: np.ndarray):
        if self.pct is None:
            self.pct = pct
        else:
            self.pct = np.concatenate([self.pct, pct], axis=0)
        
    @property
    def clip_features(self):
        return self._clip_features

    @clip_features.setter
    def clip_features(self, value: np.ndarray):
        self._clip_features = value

class Edge:
    def __init__(self, source: Node, target: Node):
        self.source = source
        self.target = target
