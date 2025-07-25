from scenegraph.node import Node, Edge
import rerun as rr
import numpy as np

class SceneGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []

    def add_node(self, node: Node):
        self.nodes[node.name] = node

    def add_edge(self, edge: Edge):
        self.edges.append(edge)
    
    def __contains__(self, node_name: str):
        return node_name in self.nodes
    
    def log_rerun(self, show_pct: bool = False, max_points: int = 5000):
        edges = rr.LineStrips3D(strips=np.array([[edge.source.centroid, edge.target.centroid] for edge in self.edges]))
        rr.log("centroids", rr.Points3D(positions=np.array([self.nodes[node_name].centroid for node_name in self.nodes]), colors=np.array([0, 0, 0]), radii=0.01))
        for node_name in self.nodes:
            if show_pct:
                rr.log(f"pcts/{node_name}", rr.Points3D(
                    # sample max_points points
                    positions=self.nodes[node_name].pct[np.random.choice(self.nodes[node_name].pct.shape[0], min(max_points, self.nodes[node_name].pct.shape[0]), replace=False)], 
                    colors=self.nodes[node_name].color, 
                    radii=0.003))
        rr.log("edges", edges)

    def log_open3d(self):
        pass

    def __len__(self):
        return len(self.nodes)

if __name__ == "__main__":
    graph = SceneGraph()
    graph.add_node(Node("node1", np.array([0, 0, 0])))
    graph.add_node(Node("node2", np.array([1, 0, 0])))
    graph.add_node(Node("node3", np.array([0, 1, 0])))
    graph.add_node(Node("node4", np.array([1, 1, 0])))
    rr.init("scene_graph", spawn=True)
    graph.log_rerun()