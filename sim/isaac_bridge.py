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
parser.add_argument("--exit_after", type=float, default=0.0, help="quit after N sim seconds (0 = run forever)")
parser.add_argument("--dt", type=float, default=1 / 120, help="physics step (s)")
parser.add_argument("--render_every", type=int, default=4, help="render once per N physics steps (4 -> 30 fps)")
AppLauncher.add_app_launcher_args(parser)
# One arm: CPU PhysX is far cheaper per step than the GPU pipeline (measured: GUI 1.19x vs 0.22x real time).
parser.set_defaults(device="cpu")
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402
import yaml  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg  # noqa: E402

ARM_JOINTS = ["shoulder_joint", "link_1_joint", "link1_link2", "a4_rotation", "a5_rotation", "a6_rotation"]
FINGER_JOINTS = ["finger_left_joint", "finger_right_joint"]
STROKE = 0.0425
ROBOT_USD = ROOT / "assets/rover2026/usd/rover2026_gripper/rover2026_gripper.usda"
COLORS = {"cup": (0.85, 0.25, 0.20), "bottle": (0.20, 0.45, 0.85), "block": (0.95, 0.75, 0.10)}

cfg = yaml.safe_load(open(args.config))
net = cfg["net"]
home = cfg["arm"]["home"]

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
    dt=args.dt, render_interval=args.render_every, device=args.device,
    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
))
sim.set_camera_view(eye=(1.35, -1.05, 0.85), target=(0.45, 0.0, 0.2))

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
        "arm": ImplicitActuatorCfg(joint_names_expr=ARM_JOINTS[:3], effort_limit_sim=200.0,
                                   velocity_limit_sim=1.5, stiffness=2000.0, damping=200.0),
        "wrist": ImplicitActuatorCfg(joint_names_expr=ARM_JOINTS[3:], effort_limit_sim=40.0,
                                     velocity_limit_sim=3.0, stiffness=400.0, damping=40.0),
        "gripper": ImplicitActuatorCfg(joint_names_expr=FINGER_JOINTS, effort_limit_sim=40.0,
                                       velocity_limit_sim=0.15, stiffness=2000.0, damping=100.0),
    },
))

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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(x, y, h / 2 + 0.001))))
heights = {n: cfg["scene"][n]["height"] for n in objects}

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
arm_ids = robot.find_joints(ARM_JOINTS, preserve_order=True)[0]
finger_ids = robot.find_joints(FINGER_JOINTS, preserve_order=True)[0]
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
prof = {"step_r": [0.0, 0], "step_n": [0.0, 0], "io": [0.0, 0]}
while app.is_running():
    t0 = time.time()
    while True:  # latest command wins
        try:
            msg = json.loads(rx.recv(65536))
        except BlockingIOError:
            break
        if msg.get("reset"):
            reset_scene()
            q_target[0] = torch.tensor(home, dtype=torch.float32, device=dev)
            g_target[:] = 0.0
            n_resets += 1
            print(f"[bridge] RESET #{n_resets} at t={t_sim:.1f}s", flush=True)
            continue
        q_target[0] = torch.tensor(msg["q"], dtype=torch.float32, device=dev)
        g_target[:] = float(msg.get("g", 0.0)) * STROKE
        n_cmd += 1
    robot.set_joint_position_target_index(target=q_target, joint_ids=arm_ids)
    robot.set_joint_position_target_index(target=g_target, joint_ids=finger_ids)
    robot.write_data_to_sim()
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
        objs[n] = [p[0], p[1], p[2] - heights[n] / 2]
    tx.sendto(json.dumps({"q": q, "objects": objs}).encode(), state_addr)
    prof["io"][0] += time.time() - ts; prof["io"][1] += 1

    if t_sim - last_log >= 2.0:
        last_log = t_sim
        fingers = robot.data.joint_pos.torch[0, finger_ids].tolist()
        rtf = t_sim / max(time.time() - wall0, 1e-6)
        print("[prof] " + " ".join(f"{k}={1e3 * v[0] / max(v[1], 1):.1f}ms" for k, v in prof.items()), flush=True)
        print(f"[bridge] t={t_sim:6.1f}s rtf={rtf:.2f} cmds={n_cmd} q={[round(v, 2) for v in q]} "
              f"fingers={[round(v * 1000, 1) for v in fingers]}mm "
              + " ".join(f"{n}=({v[0]:.3f},{v[1]:.3f},{v[2]:.3f})" for n, v in objs.items()), flush=True)
    if args.exit_after and t_sim >= args.exit_after:
        break
    if args.realtime > 0:
        time.sleep(max(0.0, dt / args.realtime - (time.time() - t0)))

app.close()
