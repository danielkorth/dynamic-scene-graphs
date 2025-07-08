import numpy as np
import rerun as rr
import rerun.blueprint as rrb

def setup_blueprint():
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial3DView(name="3D", origin="world"),
                    rrb.TextDocumentView(name="Description", origin="/description"),
                    row_shares=[7, 3],
                ),
                rrb.Vertical(
                    # Put the origin for both 2D spaces where the pinhole is logged. Doing so allows them to understand how they're connected to the 3D space.
                    # This enables interactions like clicking on a point in the 3D space to show the corresponding point in the 2D spaces and vice versa.
                    rrb.Spatial2DView(name="RGB", origin="world/camera/image", contents="world/camera/image/rgb"),
                    rrb.Spatial2DView(name="Depth", origin="world/camera/image", contents="world/camera/image/depth"),
                    name="2D",
                    row_shares=[3, 3, 2],
                ),
                column_shares=[2, 1],
            ),
    )
    return blueprint

# rr.init("living_room", spawn=True)
# rr.send_blueprint(setup_blueprint(), make_default=True)
