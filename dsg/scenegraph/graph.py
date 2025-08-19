from dsg.scenegraph.node import Node, Edge
import rerun as rr
import numpy as np
import matplotlib.pyplot as plt

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
        # edges = rr.LineStrips3D(strips=np.array([[edge.source.centroid, edge.target.centroid] for edge in self.edges]))
        # rr.log("world/edges", edges)
        for node_name in self.nodes:
            if show_pct:
                rr.log(f"world/points/{node_name}", rr.Points3D(
                    # sample max_points points
                    positions=self.nodes[node_name].pct[np.random.choice(self.nodes[node_name].pct.shape[0], min(max_points, self.nodes[node_name].pct.shape[0]), replace=False)], 
                    radii=0.005,
                    class_ids=np.array([self.nodes[node_name].id] * self.nodes[node_name].pct.shape[0]))
                )
            rr.log(f"world/centroids/{node_name}", rr.Points3D(
                positions=self.nodes[node_name].centroid, 
                radii=0.03,
                class_ids=np.array([self.nodes[node_name].id] * self.nodes[node_name].pct.shape[0]))
            )

    def highlight_clip_feature_simiarlity(self, text: str = "football", max_points: int = 5000):
        from dsg.features.clip_features import CLIPFeatures
        clip_extractor = CLIPFeatures()
        clip_features = clip_extractor.extract_text_features(text).cpu().numpy()

        cmap = plt.get_cmap("plasma")

        # collect nodes with CLIP features
        node_names = [n for n in self.nodes if self.nodes[n].clip_features is not None]

        # compute cosine similarities for all nodes
        text_feat = clip_features.squeeze()
        text_norm = np.linalg.norm(text_feat) + 1e-8
        sims = np.array([
            float(
                np.dot(self.nodes[n].clip_features, text_feat)
                / ((np.linalg.norm(self.nodes[n].clip_features) + 1e-8) * text_norm)
            )
            for n in node_names
        ])

        # robust normalization using percentiles, then gamma shaping to emphasize highs
        p_low, p_high = np.percentile(sims, [5, 95])
        denom = (p_high - p_low) if (p_high - p_low) > 1e-8 else 1.0
        norm = np.clip((sims - p_low) / denom, 0.0, 1.0)
        gamma = 0.6  # <1.0 makes top similarities stand out more
        vals = np.power(norm, gamma)

        # colorize and log
        for idx, node_name in enumerate(node_names):
            color = cmap(vals[idx])[:3]
            positions = self.nodes[node_name].pct[
                np.random.choice(
                    self.nodes[node_name].pct.shape[0],
                    min(max_points, self.nodes[node_name].pct.shape[0]),
                    replace=False,
                )
            ]
            rr.log(
                f"world/points/{node_name}",
                rr.Points3D(
                    positions=positions,
                    radii=0.005,
                    colors=np.array([color] * positions.shape[0]),
                ),
            )


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