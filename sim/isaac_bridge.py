"""VM (Isaac, ~/isaacenv6): rover2026 arm + gripper + table/cup/bottle, driven by joint targets over UDP.

    OMNI_KIT_ACCEPT_EULA=YES ~/isaacenv6/bin/python sim/isaac_bridge.py [--headless]

Protocol (matches rover_intent.robot.backends.IsaacArm):
  rx :47110  {"q": [6 joint targets, rad], "g": 0..1 gripper}
  tx :47111  {"q": [6 measured joint positions]}

TODO (milestone 1), reusing what worked in LearnFlake/isaac_migration:
  1. Build the gripper URDF: rover2026.urdf + a 2-finger gripper on the +x tool face of a6_EE_holder
     (pattern: sim/make_ee_urdf.py; geometry reference: docs/GRIPPER.md), then URDF -> USD with
     sim/import_ee.py (Isaac 6.0.1 API: URDFImporter(URDFImporterConfig(...)).import_urdf()).
     a6_rotation must be imported as a real continuous joint (exporter wrote lower=upper=0 -> locked).
  2. Scene: ground, table (top z ~ 0.0 in arm base frame, or shift config scene/workspace), rigid cup + bottle
     with mass/friction so grasps are physical (no attach/detach cheats in the demo).
  3. Articulation with stiff PD position drives; apply the incoming targets each physics step (60-120 Hz),
     publish joint state every step. Isaac Lab's UR10e_ROBOTIQ_2F_85_CFG shows how to drive a
     mimic-joint gripper.
  4. Viewport camera for the demo video (record_v4.py pattern: --headless --enable_cameras).
Remember: GPU jobs through `tsp gpu-run`; the playback on display :0 shares the GPU.
"""

raise SystemExit("sim/isaac_bridge.py is a stub - see the TODO in its docstring")
