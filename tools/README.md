# Bench launchers

Shell tools that run the real-arm bench on the VM. They expect the workspace layout
`~/projects/ubc-rover-arm/{rover-intent,RoverFlake2,tools,logs}` and are usually symlinked into `~/bin`.

| Tool | What it does |
|---|---|
| `rover-up` | One command for the bench: sim stack, arm link, then the follower (never homes the arm) |
| `rover-stack on\|off\|status\|restart` | Isaac bridge + app + EEG hub |
| `arm-link up\|down\|home\|status\|watch\|logs` | Laptop serial bridge tunnel -> socat pty -> Rowan's ROS 2 `stepper_arm_driver` |
| `rover-follow on\|off\|align\|mirror\|status [--ee]` | Real arm follows the sim (`scripts/sim_to_real.py`), plus the raw recorder |
| `ee_timecheck.py` | Bench timing check for the open-loop gripper |
