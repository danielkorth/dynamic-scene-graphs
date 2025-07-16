import rerun.blueprint as rrb
import rerun as rr
import numpy as np

def setup_blueprint():
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(name="3D", origin="world", line_grid=rrb.archetypes.LineGrid3D(visible=False)),
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(name="RGB", origin="world/camera/image", contents="world/camera/image/rgb"),
                    rrb.Spatial2DView(name="Depth", origin="world/camera/image", contents="world/camera/image/depth"),
                    name="2D",
                    row_shares=[1,1]
                ),
            ),
    )
    return blueprint
