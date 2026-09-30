"""Isaac Sim side of the IsaacArm backend: rover2026 + gripper + cup/bottle/block, driven over UDP.

    # GUI on :0 (watch via Moonlight):
    DISPLAY=:0 OMNI_KIT_ACCEPT_EULA=YES ~/projects/ubc-rover-arm/isaac/isaacenv6/bin/python sim/isaac_bridge.py --viz kit
    # headless:
    OMNI_KIT_ACCEPT_EULA=YES ~/projects/ubc-rover-arm/isaac/isaacenv6/bin/python sim/isaac_bridge.py --headless
Then, in .venv:  python -m rover_intent.app --backend isaac

Protocol (rover_intent.robot.backends.IsaacArm):
  rx :47110  {"q": [6 arm joint targets, rad], "g": 0..1 gripper close}   or   {"reset": true}
  tx :47111  {"q": [6 measured arm joints], "objects": {name: [x, y, z_bottom]}}
Until the first command arrives the arm holds the config's home pose.

Sim choices (be honest about them in the video):
  * The arm links ignore gravity = perfect gravity compensation. Real drives need a gravity feed-forward or
    integral term; the arm's gravity torques are in docs/rover2026_arm_specs.md.
  * Stiff PD position drives; the app's SafetyLayer shapes the trajectory (vel/acc limits), the sim only tracks.
  * Grasps are physical (friction, finger force limit). No attach/detach.
"""
import argparse
import json
import socket
import time
from pathlib import Path

from isaaclab.app import AppLauncher

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
parser.add_argument("--realtime", type=float, default=1.0, help="1 = real time; 0 = as fast as possible")
parser.add_argument("--arm", default=None, help="arm profile (configs/arms/<name>.yaml); default: ROVER_ARM or config")
parser.add_argument("--exit_after", type=float, default=0.0, help="quit after N sim seconds (0 = run forever)")
parser.add_argument("--dt", type=float, default=1 / 120, help="physics step (s)")
parser.add_argument("--view", default=None,
                    help="camera: behind | operator | high | top | shoulder | side | front. Live: UDP {\"view\": \"top\"}")
parser.add_argument("--render_every", type=int, default=4, help="render once per N physics steps (4 -> 30 fps)")
AppLauncher.add_app_launcher_args(parser)
# One arm: CPU PhysX is far cheaper per step than the GPU pipeline (measured: GUI 1.19x vs 0.22x real time).
parser.set_defaults(device="cpu")
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from rover_intent.config import load_config  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402

COLORS = {"cup": (0.85, 0.25, 0.20), "bottle": (0.20, 0.45, 0.85), "block": (0.95, 0.75, 0.10)}

cfg = load_config(args.config, args.arm)
ARM_JOINTS = cfg["arm"]["joints"]
FINGER_JOINTS = cfg["sim"]["finger_joints"]
STROKE = cfg["sim"]["stroke"]
ROBOT_USD = ROOT / cfg["sim"]["usd"]
print(f"[bridge] arm profile {cfg['arm_profile']}: {ROBOT_USD.name}", flush=True)
net = cfg["net"]
home = cfg["arm"]["home"]

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
    dt=args.dt, render_interval=args.render_every, device=args.device,
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
))
# operator (default, Pranav 2026-09-30): eye level, behind and to the right of the arm, like standing at its shoulder;
# whole arm + gripper + objects stay in view during grasps. Your right = screen right (except front/side).
# behind (default, 2026-09-30): straight behind the base, looking the way you face the webcam -> hand right =
# gripper right on screen; table, box, objects and gripper visible at home and mid-grasp.
VIEWS = {"behind": ((-0.75, -0.1, 1.0), (0.62, 0.0, 0.15)),
         "operator": ((-0.9, -0.6, 0.7), (0.58, 0.0, 0.18)),
         "high": ((-0.35, 0.0, 1.75), (0.65, 0.0, 0.0)),
         "top": ((0.58, 0.0, 1.9), (0.6, 0.0, 0.0)),
         "shoulder": ((-0.85, 0.2, 1.55), (0.6, -0.05, 0.05)),
         "side": ((0.6, 1.4, 0.6), (0.6, 0.0, 0.15)),
         "front": ((1.35, -1.05, 0.85), (0.45, 0.0, 0.2))}


for _name, _v in cfg.get("sim", {}).get("views", {}).items():   # per-arm camera views from the profile
    VIEWS[_name] = (tuple(_v["eye"]), tuple(_v["target"]))


current_cam = {"eye": None, "target": None}


def set_view(v):
    eye, target = (VIEWS[v] if isinstance(v, str) else (v["eye"], v["target"]))
    sim.set_camera_view(eye=eye, target=target)
    current_cam.update(eye=list(map(float, eye)), target=list(map(float, target)))


set_view(args.view or cfg.get("sim", {}).get("default_view", "behind"))

ground = sim_utils.GroundPlaneCfg()
ground.func("/World/ground", ground)
light = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
light.func("/World/Light", light)

robot = Articulation(ArticulationCfg(
    prim_path="/World/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROBOT_USD),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=16, solver_velocity_iteration_count=4),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={**dict(zip(ARM_JOINTS, home)), **{j: 0.0 for j in FINGER_JOINTS}}),
    actuators={
        **{f"group{i}": ImplicitActuatorCfg(joint_names_expr=g["joints"], effort_limit_sim=g["effort"],
                                            velocity_limit_sim=g["velocity"], stiffness=g["stiffness"],
                                            damping=g["damping"]) for i, g in enumerate(cfg["sim"]["actuators"])},
        "gripper": ImplicitActuatorCfg(joint_names_expr=FINGER_JOINTS, effort_limit_sim=40.0,
                                       velocity_limit_sim=0.15, stiffness=2000.0, damping=100.0),
    },
))

# Static table in front of the arm (collider only, no rigid body -> it can't move).
tbl = cfg.get("table")
table_top = tbl["top"] if tbl else 0.0
if tbl and tbl["top"] > 0:  # top 0 = objects stand on the arm's own surface (the ground): no table
    tsize = (tbl["x"][1] - tbl["x"][0], tbl["y"][1] - tbl["y"][0], tbl["top"])
    tcfg = sim_utils.CuboidCfg(size=tsize, collision_props=sim_utils.CollisionPropertiesCfg(),
                               visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.33, 0.22)))
    tcfg.func("/World/Table", tcfg, translation=((tbl["x"][0] + tbl["x"][1]) / 2, (tbl["y"][0] + tbl["y"][1]) / 2,
                                                 tbl["top"] / 2))

objects = {}
for name, o in cfg["scene"].items():
    common = dict(
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=o.get("mass", 0.2)),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=COLORS.get(name, (0.6, 0.6, 0.6))),
    )
    h, r = o["height"], o.get("radius", 0.03)
    spawn = (sim_utils.CuboidCfg(size=(2 * r, 2 * r, h), **common) if o.get("shape") == "box"
             else sim_utils.CylinderCfg(radius=r, height=h, axis="Z", **common))
    x, y, _ = o["pos"]
    objects[name] = RigidObject(RigidObjectCfg(
        prim_path=f"/World/Objects/{name}", spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, table_top + h / 2 + 0.001))))
heights = {n: cfg["scene"][n]["height"] for n in objects}

# Sync cues (visual only): ghost at the target + a line from the gripper to it.
# ok = green (tracking), clamped = amber (out of reach, clamped), lost = red (tracking lost), calib = blue (hold still)
GHOST_STATES = ["ok", "clamped", "lost", "calib"]
_C = {"ok": (0.1, 0.9, 0.3), "clamped": (1.0, 0.65, 0.0), "lost": (0.95, 0.1, 0.1), "calib": (0.2, 0.5, 1.0)}
ghost = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/ghost", markers={
    k: sim_utils.SphereCfg(radius=0.03, visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=_C[k], emissive_color=tuple(0.45 * c for c in _C[k]), opacity=0.55)) for k in GHOST_STATES}))
line = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/ghost_line", markers={
    k: sim_utils.CylinderCfg(radius=0.004, height=1.0, axis="Z", visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=_C[k], emissive_color=tuple(0.45 * c for c in _C[k]), opacity=0.7)) for k in GHOST_STATES}))
ghost_pos, ghost_state, ghost_shown = None, None, None
ghost.set_visibility(False)  # markers are created visible at the origin: hide until there is a target
line.set_visibility(False)


def quat_z_to(d):
    """(x, y, z, w) quaternion rotating +Z onto unit vector d."""
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, d))
    if c < -0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax = np.cross(z, d)
    q = np.array([ax[0], ax[1], ax[2], 1.0 + c])
    return q / np.linalg.norm(q)

# Reach box = the region the controller lets the gripper go (safety workspace, IK-checked reachable top-down),
# drawn as a thin WIREFRAME: a translucent solid renders as frosted glass in RTX real-time and hides the objects.
ws_lo, ws_hi = np.array(cfg["safety"]["workspace_min"]), np.array(cfg["safety"]["workspace_max"])
reach_box = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/reach_box", markers={
    "edge": sim_utils.CylinderCfg(radius=0.0025, height=1.0, axis="Z", visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.35, 0.65, 1.0), emissive_color=(0.1, 0.25, 0.5)))}))

sim.reset()


def reset_scene():
    """Arm to home (teleport, zero velocity), gripper open, every object back to its start pose.
    Also needed at start-up: sim.reset() does NOT apply init_state (the arm would spawn at all-zero joints
    and swing ~4 rad to home at full speed)."""
    q0 = robot.data.default_joint_pos.torch.clone()
    robot.write_joint_position_to_sim_index(position=q0)
    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(q0))
    robot.set_joint_position_target_index(target=q0)
    robot.write_data_to_sim()
    robot.update(0.0)
    for o in objects.values():
        o.write_root_pose_to_sim_index(root_pose=o.data.default_root_pose.torch.clone())
        o.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(o.data.default_root_vel.torch))
        o.reset()
        o.update(0.0)


reset_scene()
# Neutral marker: a white 3-axis cross at the box centre (your relaxed hand maps here after calibration).
neutral_mk = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/neutral", markers={
    "bar": sim_utils.CylinderCfg(radius=0.004, height=0.08, axis="Z", visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(1.0, 1.0, 1.0), emissive_color=(0.6, 0.6, 0.6))),
    "dot": sim_utils.SphereCfg(radius=0.012, visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(1.0, 1.0, 1.0), emissive_color=(0.6, 0.6, 0.6)))}))
_c = (ws_lo + ws_hi) / 2
neutral_mk.visualize(translations=np.array([_c] * 4, dtype=np.float32),
                     orientations=np.array([quat_z_to(np.array(d, float)) for d in
                                            ([1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1])], dtype=np.float32),
                     marker_indices=[0, 0, 0, 1])
# Verification target (magenta): touch it with the gripper during the calibration check.
vtarget = VisualizationMarkers(VisualizationMarkersCfg(prim_path="/Visuals/vtarget", markers={
    "t": sim_utils.SphereCfg(radius=0.025, visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.95, 0.1, 0.85), emissive_color=(0.5, 0.05, 0.45), opacity=0.8))}))
vtarget.set_visibility(False)
vtarget_shown = None
_corners = [np.array([x, y, z]) for x in (ws_lo[0], ws_hi[0]) for y in (ws_lo[1], ws_hi[1]) for z in (ws_lo[2], ws_hi[2])]
_edges = [(a, b) for i, a in enumerate(_corners) for b in _corners[i + 1:] if np.count_nonzero(a != b) == 1]
reach_box.visualize(translations=np.array([(a + b) / 2 for a, b in _edges], dtype=np.float32),
                    orientations=np.array([quat_z_to((b - a) / np.linalg.norm(b - a)) for a, b in _edges], dtype=np.float32),
                    scales=np.array([[1.0, 1.0, np.linalg.norm(b - a)] for a, b in _edges], dtype=np.float32),
                    marker_indices=[0] * len(_edges))
arm_ids = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
finger_ids = robot.find_joints(FINGER_JOINTS, preserve_order=True)[0]
tcp_body = robot.find_bodies(cfg["sim"].get("tcp_body", "tcp"))[0][0]
dev = sim.device
q_target = torch.tensor([home], dtype=torch.float32, device=dev)
g_target = torch.zeros(1, 2, device=dev)

rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
rx.bind(("0.0.0.0", net["isaac_cmd_port"]))
rx.setblocking(False)
tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
state_addr = ("127.0.0.1", net["isaac_state_port"])
print(f"[bridge] ready: cmd :{net['isaac_cmd_port']}  state -> :{net['isaac_state_port']}", flush=True)

dt = sim.get_physics_dt()
t_sim, n_cmd, last_log, n_resets = 0.0, 0, 0.0, 0
wall0, n_step = time.time(), 0
wall_start = wall0  # for the rtf readout (wall0 is re-anchored after hitches; this one isn't)
prof = {"step_r": [0.0, 0], "step_n": [0.0, 0], "io": [0.0, 0]}
while app.is_running():
    t0 = time.time()
    while True:  # latest command wins
        try:
            msg = json.loads(rx.recv(65536))
        except BlockingIOError:
            break
        if "view" in msg:
            try:
                set_view(msg["view"])
                print(f"[bridge] view -> {msg['view']}", flush=True)
            except (KeyError, TypeError):
                print(f"[bridge] unknown view {msg['view']!r}; known: {list(VIEWS)}", flush=True)
            continue
        if msg.get("reset"):
            reset_scene()
            q_target[0] = torch.tensor(home, dtype=torch.float32, device=dev)
            g_target[:] = 0.0
            n_resets += 1
            print(f"[bridge] RESET #{n_resets} at t={t_sim:.1f}s", flush=True)
            continue
        q_target[0] = torch.tensor(msg["q"], dtype=torch.float32, device=dev)
        g_target[:] = float(msg.get("g", 0.0)) * STROKE
        vt = msg.get("vtarget")
        if (tuple(vt) if vt else None) != vtarget_shown:
            vtarget_shown = tuple(vt) if vt else None
            if vt:
                vtarget.set_visibility(True)
                vtarget.visualize(translations=np.array([vt], dtype=np.float32))
            else:
                vtarget.set_visibility(False)
        ghost_pos = msg.get("ghost")
        ghost_state = msg.get("ghost_state") or ("clamped" if msg.get("ghost_bad") else "ok")
        n_cmd += 1
    robot.set_joint_position_target_index(target=q_target, joint_ids=arm_ids)
    robot.set_joint_position_target_index(target=g_target, joint_ids=finger_ids)
    robot.write_data_to_sim()
    tcp_now = robot.data.body_pos_w.torch[0, tcp_body].cpu().numpy() if ghost_pos else None
    state = ((tuple(round(v, 3) for v in ghost_pos), ghost_state, tuple(np.round(tcp_now, 3)))
             if ghost_pos else None)
    if state != ghost_shown:  # only touch the markers when something changed
        ghost_shown = state
        if state is None:
            ghost.set_visibility(False)
            line.set_visibility(False)
        else:
            idx = GHOST_STATES.index(ghost_state) if ghost_state in GHOST_STATES else 0
            gp = np.asarray(ghost_pos, float)
            ghost.set_visibility(True)
            ghost.visualize(translations=np.array([gp], dtype=np.float32), marker_indices=[idx])
            d = gp - tcp_now
            length = float(np.linalg.norm(d))
            if length > 0.01:
                line.set_visibility(True)
                line.visualize(translations=np.array([(gp + tcp_now) / 2], dtype=np.float32),
                               orientations=np.array([quat_z_to(d / length)], dtype=np.float32),
                               scales=np.array([[1.0, 1.0, length]], dtype=np.float32), marker_indices=[idx])
            else:
                line.set_visibility(False)
    n_step += 1
    rend = n_step % args.render_every == 0
    ts = time.time()
    sim.step(render=rend)  # step() ignores render_interval; gate it here
    k = "step_r" if rend else "step_n"
    prof[k][0] += time.time() - ts; prof[k][1] += 1
    t_sim += dt
    ts = time.time()
    robot.update(dt)
    for o in objects.values():
        o.update(dt)

    q = robot.data.joint_pos.torch[0, arm_ids].tolist()
    objs = {}
    for n, o in objects.items():
        p = o.data.root_pos_w.torch[0].tolist()
        objs[n] = [p[0], p[1], p[2] - heights[n] / 2]  # bottom (on the table: table_top)
    tx.sendto(json.dumps({"q": q, "objects": objs, "cam": current_cam}).encode(), state_addr)
    prof["io"][0] += time.time() - ts; prof["io"][1] += 1

    if t_sim - last_log >= 2.0:
        last_log = t_sim
        fingers = robot.data.joint_pos.torch[0, finger_ids].tolist()
        rtf = t_sim / max(time.time() - wall_start, 1e-6)
        print("[prof] " + " ".join(f"{k}={1e3 * v[0] / max(v[1], 1):.1f}ms" for k, v in prof.items()), flush=True)
        print(f"[bridge] t={t_sim:6.1f}s rtf={rtf:.2f} cmds={n_cmd} q={[round(v, 2) for v in q]} "
              f"fingers={[round(v * 1000, 1) for v in fingers]}mm "
              + " ".join(f"{n}=({v[0]:.3f},{v[1]:.3f},{v[2]:.3f})" for n, v in objs.items()), flush=True)
    if args.exit_after and t_sim >= args.exit_after:
        break
    if args.realtime > 0:
        # pace against a GLOBAL clock: a slow (rendered) step is caught up by the fast ones. Per-step pacing
        # ("sleep until dt after this step began") lost every render overshoot for good: rtf 0.82 with 36 % idle CPU.
        lag = wall0 + t_sim / args.realtime - time.time()
        if lag > 0:
            time.sleep(lag)
        elif lag < -0.25:
            wall0 -= lag + 0.25  # far behind (e.g. a hitch): don't sprint to catch up, just re-anchor

app.close()
