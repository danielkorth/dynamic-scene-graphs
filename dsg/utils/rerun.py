import rerun.blueprint as rrb
import rerun as rr

def setup_blueprint():
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(name="3D", origin="world", background=[255, 255, 255], line_grid=rrb.archetypes.LineGrid3D(visible=True, spacing=0.8, plane=rr.components.Plane3D(normal=[0,1,0], distance=1.5), color=[99, 99, 99])),
            rrb.Horizontal(
                rrb.Spatial2DView(name="RGB", origin="world/camera/image", contents="world/camera/image/rgb"),
                rrb.Spatial2DView(name="Depth", origin="world/camera/image", contents="world/camera/image/depth"),
                rrb.Spatial2DView(name="Mask", origin="world/camera/image", contents="world/camera/image/mask"),
                name="2D",
            ),
            row_shares=[7,3]
        )
    )
    return blueprint
