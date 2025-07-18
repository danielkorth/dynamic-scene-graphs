import rerun.blueprint as rrb
import rerun as rr
import numpy as np

def setup_blueprint():
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(name="3D", origin="world", contents="world/points", line_grid=rrb.archetypes.LineGrid3D(visible=True, spacing=0.1, plane=rr.components.Plane3D(normal=np.array([0,1,0]), distance=0.5))),
            rrb.Horizontal(
                rrb.Spatial2DView(name="RGB", origin="world/camera/image", contents="world/camera/image/rgb"),
                rrb.Spatial2DView(name="Depth", origin="world/camera/image", contents="world/camera/image/depth"),
                name="2D",
            ),
            row_shares=[7,3]
        )
    )
    return blueprint
