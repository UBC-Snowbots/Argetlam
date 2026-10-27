"""Isaac-side rover camera for perception: render RGB (+ depth + instance segmentation) from the pose
and intrinsics in the arm profile `camera:` block, JPEG-encode, and stream it to the app as the
existing CameraFrame message, WITH the sim ground-truth object bottoms (+ per-object 2D boxes from the
segmentation) in the same packet so scripts/perception_eval.py can score the camera path against truth.

UNTESTED IN ISAAC (Thoth holds the shared GPU; no Isaac launch per the brief). It is written against
the Isaac Lab API the rest of sim/ uses. Isaac-only calls (isaaclab.sensors.Camera, the replicator
semantics helper) are lazy-imported inside the Isaac-only functions, so the PURE helpers below
(camera_offset, boxes_from_label_map, encode_rgb_jpeg, encode_depth_png, build_frame) import and are
unit-tested with no Isaac (tests/test_perception_rover.py covers the geometry; see also test_sim_camera
if present). To try it when the GPU is free, see docs/PERCEPTION.md "Running the sim camera".

Isaac Lab reference (isaacsim 6.0.1-rc.7; the old isaacsim.sensors.camera is deprecated):

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    cam = Camera(CameraCfg(
        prim_path="/World/RoverCam",
        update_period=1.0 / rate_hz,                 # 0.2 -> 5 Hz, 0.1 -> 10 Hz
        height=H, width=W,
        data_types=["rgb", "distance_to_image_plane",
                    "semantic_segmentation", "instance_id_segmentation_fast"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=K.flatten().tolist(), width=W, height=H, focus_distance=0.6,
            clipping_range=(0.05, 10.0)),
        offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention=convention)))

and the app must be launched with `--enable_cameras` (AppLauncher). Prims are labelled with
`rep.functional.modify.semantics(prim, {"class": "cup"})` so the segmentation ids map back to object
names. Omniverse's pinhole assumes SQUARE pixels and a CENTRED principal point (fx==fy, cx=W/2,
cy=H/2); a real camera whose K differs must be undistorted/rectified to that before this model applies.
"""
from __future__ import annotations

import base64
import io
import json
import socket
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rover_intent.perception.camera import camera_from_config  # noqa: E402  (pure numpy)


# --- pure helpers (no Isaac) -----------------------------------------------------------------------
def _R_to_quat_wxyz(R: np.ndarray) -> list[float]:
    """3x3 rotation -> quaternion (w, x, y, z), the Isaac convention."""
    m = np.asarray(R, float)
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s; x = 0.25 * s; y = (m[0, 1] + m[1, 0]) / s; z = (m[0, 2] + m[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 - m[0, 0] + m[1, 1] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s; x = (m[0, 1] + m[1, 0]) / s; y = 0.25 * s; z = (m[1, 2] + m[2, 1]) / s
        else:
            s = np.sqrt(1.0 - m[0, 0] - m[1, 1] + m[2, 2]) * 2
            w = (m[1, 0] - m[0, 1]) / s; x = (m[0, 2] + m[2, 0]) / s; y = (m[1, 2] + m[2, 1]) / s; z = 0.25 * s
    q = np.array([w, x, y, z])
    return (q / np.linalg.norm(q)).tolist()


def camera_offset(cam_cfg: dict, z_table_default: float = 0.0):
    """Arm-profile `camera:` block -> (pos [x,y,z], quat_wxyz, K 3x3, width, height, z_table) for the
    Isaac CameraCfg.OffsetCfg (convention='opencv': the offset rot is the OpenCV optical frame's
    orientation in the parent frame, which is exactly PinholeCamera's R_wc = R_cw.T).

    NOTE on convention: 'opencv' is used because it matches PinholeCamera byte-for-byte (+z forward,
    +x right, +y down). The research note suggested convention='ros'; if you switch to 'ros', the
    OffsetCfg rot must be re-derived for the ROS optical frame and verified live - this quaternion is
    the OpenCV one.
    """
    cam, z_table = camera_from_config(cam_cfg, z_table_default=z_table_default)
    pos = cam.center().tolist()
    R_wc = cam.R_cw.T  # camera-optical -> base; columns are the camera axes in the base/world frame
    return pos, _R_to_quat_wxyz(R_wc), cam.K, cam.width, cam.height, z_table


def boxes_from_label_map(label_map: np.ndarray, id_to_name: dict[int, str]) -> dict[str, list[float]]:
    """A HxW integer segmentation map + {segmentation id -> object name} -> {name: [u0,v0,u1,v1]} tight
    2D boxes. Ids absent from the map are skipped; an empty region yields no box."""
    out = {}
    lm = np.asarray(label_map)
    for sid, name in id_to_name.items():
        ys, xs = np.where(lm == sid)
        if xs.size == 0:
            continue
        out[name] = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    return out


def gt_boxes_from_scene(cam, lib, objects_gt: dict, z_table: float) -> dict:
    """Ground-truth 2D boxes {name: [u0,v0,u1,v1]} by projecting each known object's silhouette through
    the camera model - deterministic, needs no Isaac segmentation, so the viewer always has clean truth
    boxes. (The Isaac instance segmentation is still rendered for depth/occlusion realism.)"""
    out = {}
    for name, bottom in (objects_gt or {}).items():
        obj = lib.get(name)
        if obj is None:
            continue
        px = cam.project(obj.silhouette_points((float(bottom[0]), float(bottom[1])), z_table))
        u0, v0, u1, v1 = px[:, 0].min(), px[:, 1].min(), px[:, 0].max(), px[:, 1].max()
        if u1 < 0 or v1 < 0 or u0 > cam.width or v0 > cam.height:
            continue  # fully outside the image
        out[name] = [float(u0), float(v0), float(u1), float(v1)]
    return out


def encode_rgb_jpeg(rgb: np.ndarray, quality: int = 70) -> str:
    """HxWx3 uint8 RGB -> base64 JPEG string (PIL; no cv2 needed)."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, np.uint8)[:, :, :3]).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def encode_depth_png(depth_m: np.ndarray, max_m: float = 10.0) -> str:
    """HxW float metres -> base64 PNG of uint16 MILLIMETRES (clipped to max_m; inf/nan -> 0 = no return)."""
    from PIL import Image
    d = np.asarray(depth_m, float)
    d = np.where(np.isfinite(d), d, 0.0)
    mm = np.clip(d * 1000.0, 0, max_m * 1000.0).astype(np.uint16)
    buf = io.BytesIO()
    Image.fromarray(mm).save(buf, format="PNG")   # uint16 2D -> PIL mode I;16
    return base64.b64encode(buf.getvalue()).decode()


def build_frame(rgb, objects_gt: dict, t: float, seq: int, width: int, height: int,
                depth_m=None, boxes: dict | None = None, quality: int = 70,
                source: str = "sim:rover_front") -> dict:
    """Assemble the CameraFrame-shaped dict the app parses: JPEG + sim ground-truth object bottoms +
    optional depth and per-object 2D boxes, all in one packet. `source` carries the mount (e.g.
    'sim:overhead') so the viewer banner reads "SIM CAMERA (overhead)"."""
    frame = {"kind": "camera_frame", "t": float(t), "jpeg_b64": encode_rgb_jpeg(rgb, quality),
             "width": int(width), "height": int(height), "seq": int(seq), "source": source,
             "objects": {n: [float(v[0]), float(v[1]), float(v[2])] for n, v in (objects_gt or {}).items()}}
    if depth_m is not None:
        frame["depth_b64"] = encode_depth_png(depth_m)
    if boxes:
        frame["boxes"] = boxes
    return frame


# --- Isaac-only (lazy; UNTESTED in Isaac) ----------------------------------------------------------
def resolve_mount(cfg: dict, mount: str | None = None):
    """Resolve a camera config to (pos, quat_wxyz) for a given mount (or cfg's current camera.mount)."""
    c = dict(cfg["camera"])
    if mount is not None:
        c["mount"] = mount
        c.pop("eye", None); c.pop("look_at", None); c.pop("pos", None); c.pop("quat", None)
    pos, rot, _K, _W, _H, _z = camera_offset(c, z_table_default=cfg.get("table", {}).get("top", 0.0))
    return pos, rot


UDP_MAX_FRAME = 60000   # bytes; UDP datagrams are capped at ~65507, keep a margin for headers


def wxyz_to_xyzw(q):
    """(w, x, y, z) -> (x, y, z, w): our pose math returns wxyz; Isaac Lab 3 wants xyzw."""
    w, x, y, z = (float(v) for v in q)
    return (x, y, z, w)


def build_camera(cfg: dict, sim_utils, prim_path: str = "/World/RoverCam"):
    """Create and return an isaaclab.sensors.Camera from the arm profile `camera:` block. Isaac-only."""
    from isaaclab.sensors import Camera, CameraCfg
    c = cfg["camera"]
    pos, rot, K, W, H, _ = camera_offset(c, z_table_default=cfg.get("table", {}).get("top", 0.0))
    rate = float(c.get("rate_hz", 10.0))
    data_types = ["rgb"]
    if c.get("depth", True):
        data_types.append("distance_to_image_plane")
    if c.get("segmentation", True):
        data_types += ["semantic_segmentation", "instance_id_segmentation_fast"]
    cam = Camera(CameraCfg(
        prim_path=prim_path, update_period=1.0 / rate, height=H, width=W, data_types=data_types,
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=K.flatten().tolist(), width=W, height=H, focus_distance=0.6,
            clipping_range=(0.05, 10.0)),
        # Isaac Lab 3: OffsetCfg.rot is (x, y, z, w) and convention must be opengl|ros|world. "ros" = the
        # optical frame +Z forward, -Y up = the OpenCV frame our K/pose math uses. (First live run 2026-10-27:
        # wxyz + "opencv" pointed the camera into empty space: flat grey RGB, 100 % no-return depth.)
        offset=CameraCfg.OffsetCfg(pos=tuple(pos), rot=tuple(wxyz_to_xyzw(rot)), convention="ros")))
    return cam


def label_object_semantics(object_prim_paths: dict[str, str]) -> None:
    """Tag each object prim with a semantic class = its name, so the segmentation ids map back to names.
    `object_prim_paths` = {object_name: prim_path}. Isaac-only (omni.replicator). UNTESTED in Isaac."""
    import omni.replicator.core as rep
    for name, prim in object_prim_paths.items():
        rep.functional.modify.semantics(rep.get.prim_at_path(prim), {"class": name})


class RoverCameraStreamer:
    """Holds the Isaac Camera sensor + a UDP socket; `step()` renders and sends a CameraFrame to the
    app's listen port. Blocking-send only (no polling loop). Isaac-only beyond the pure helpers."""

    def __init__(self, cam_sensor, cfg: dict, id_to_name: dict[int, str] | None = None):
        self.cam = cam_sensor
        self.cfg = cfg
        self.id_to_name = id_to_name or {}
        c = cfg["camera"]
        _, _, _K, W, H, _ = camera_offset(c, z_table_default=cfg.get("table", {}).get("top", 0.0))
        self.width, self.height = W, H
        self.depth_on = bool(c.get("depth", True))
        self.seg_on = bool(c.get("segmentation", True))
        self.period = 1.0 / float(c.get("rate_hz", 10.0))
        self.quality = int(c.get("jpeg_quality", 70))
        self.mount = c.get("mount", "rover_front")
        # camera model + object library for deterministic ground-truth 2D boxes (no Isaac seg needed)
        from rover_intent.perception.objectlib import library_from_scene, set_support
        self._cam, self.z_table = camera_from_config(c, z_table_default=cfg.get("table", {}).get("top", 0.0))
        self._lib = library_from_scene(cfg["scene"])
        set_support(self._lib, cfg.get("table", {}).get("top", 0.0))
        net = cfg["net"]
        # send to the app's own listener as the existing CameraFrame. In a sim run the app uses SPARE
        # ports (never the live 47100); set net.listen_port in the sim app's config, as rover-stack does.
        self.addrs = [(net.get("app_host", net.get("isaac_host", "127.0.0.1")), net["listen_port"])]
        # optionally also stream to a standalone viewer (scripts/sim_cam_viewer.py) if a port is set
        if net.get("sim_cam_view_port"):
            self.addrs.append((net.get("view_host", "127.0.0.1"), int(net["sim_cam_view_port"])))
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq, self._last_t = 0, -1e9
        self._fit_logged = False
        self._moved = False   # set by set_mount: force one re-render + log the pose the sensor now reports

    def set_mount(self, mount: str) -> None:
        """Runtime mount switch: move the camera prim to the new mount's pose (Isaac-only). The prim is
        reused, so no Isaac restart - the next frame renders from the new pose.
        KNOWN BROKEN (live test 2026-10-27): the call raises nothing, but the sensor's pos_w and the render stay on the
        start-up mount. Choosing camera.mount at start-up works; restart the bridge to change mounts."""
        pos, rot = resolve_mount(self.cfg, mount)
        self.mount = mount
        # Rebuild OUR pinhole model for the new mount too, so the ground-truth boxes in each frame follow the view
        # (first live test 2026-10-27: the label changed but boxes and image stayed on the old mount).
        c = dict(self.cfg["camera"])
        c["mount"] = mount
        for k in ("eye", "look_at", "pos", "quat"):
            c.pop(k, None)
        self._cam, self.z_table = camera_from_config(c, z_table_default=self.cfg.get("table", {}).get("top", 0.0))
        m = (self.cfg["camera"].get("mounts") or {}).get(mount, {})
        try:  # Isaac Lab: move the sensor prim (eye/look_at mounts use the look-at API directly)
            import torch
            dev = getattr(self.cam, "device", "cpu")
            if "eye" in m and "look_at" in m:
                self.cam.set_world_poses_from_view(torch.tensor([m["eye"]], dtype=torch.float32, device=dev),
                                                   torch.tensor([m["look_at"]], dtype=torch.float32, device=dev))
            else:
                p = torch.tensor([pos], dtype=torch.float32, device=dev)
                q = torch.tensor([wxyz_to_xyzw(rot)], dtype=torch.float32, device=dev)   # Isaac Lab 3: (x,y,z,w)
                self.cam.set_world_poses(p, q, convention="ros")
        except Exception as e:  # pragma: no cover - Isaac-only
            print(f"[rover_camera] set_mount({mount}) could not move the prim ({e}); "
                  f"restart the bridge with camera.mount: {mount}", flush=True)
        self._moved = True

    def step(self, t_sim: float, objects_gt: dict) -> bool:
        """If the camera period has elapsed, render + send one frame. Returns True if a frame was sent."""
        if t_sim - self._last_t < self.period:
            return False
        self._last_t = t_sim
        self.cam.update(dt=0.0, force_recompute=self._moved)
        if self._moved:
            self._moved = False
            try:
                print(f"[rover_camera] mount {self.mount}: sensor pos_w={self.cam.data.pos_w[0].tolist()}", flush=True)
            except Exception:
                pass
        out = self.cam.data.output
        rgb = out["rgb"][0].cpu().numpy() if hasattr(out["rgb"], "cpu") else np.asarray(out["rgb"])[0]
        depth = None
        if self.depth_on and "distance_to_image_plane" in out:
            d = out["distance_to_image_plane"][0]
            depth = d.cpu().numpy() if hasattr(d, "cpu") else np.asarray(d)
            depth = np.squeeze(depth)
        # ground-truth 2D boxes: deterministic from the known geometry (always), or from the Isaac
        # instance segmentation when an id->name map is supplied (respects real occlusion).
        boxes = gt_boxes_from_scene(self._cam, self._lib, objects_gt, self.z_table)
        if self.seg_on and "instance_id_segmentation_fast" in out and self.id_to_name:
            seg = out["instance_id_segmentation_fast"][0]
            seg = seg.cpu().numpy() if hasattr(seg, "cpu") else np.asarray(seg)
            boxes = boxes_from_label_map(np.squeeze(seg), self.id_to_name)
        frame = build_frame(rgb, objects_gt, t_sim, self.seq, self.width, self.height,
                            depth_m=depth, boxes=boxes, quality=self.quality, source=f"sim:{self.mount}")
        data = json.dumps(frame).encode()
        # UDP caps a datagram at ~64 KB and sendto() fails SILENTLY for bigger ones (first live run 2026-10-27: a
        # real 640x480 image + depth PNG never arrived). Fit the frame: drop depth first, then lower JPEG quality.
        q = self.quality
        while len(data) > UDP_MAX_FRAME and (depth is not None or q > 25):
            if depth is not None:
                depth = None
            else:
                q = max(25, q - 15)
            frame = build_frame(rgb, objects_gt, t_sim, self.seq, self.width, self.height,
                                depth_m=depth, boxes=boxes, quality=q, source=f"sim:{self.mount}")
            data = json.dumps(frame).encode()
        if (q != self.quality or depth is None and self.depth_on) and not self._fit_logged:
            print(f"[rover_camera] frame fitted to UDP: depth {'dropped' if depth is None else 'kept'}, "
                  f"jpeg quality {q}, {len(data)} bytes (limit {UDP_MAX_FRAME})", flush=True)
            self._fit_logged = True
        if len(data) > UDP_MAX_FRAME:
            print(f"[rover_camera] frame still too big for UDP ({len(data)} bytes): lower camera width/height",
                  flush=True)
            return False
        for addr in self.addrs:   # app (perception + HUD forward) and optionally the standalone viewer
            try:
                self.tx.sendto(data, addr)
            except OSError:
                pass  # nothing listening yet: drop, never block the physics loop
        self.seq += 1
        return True
