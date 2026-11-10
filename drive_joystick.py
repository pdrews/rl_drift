"""Drive a car in MuJoCo with a PlayStation (PS4/PS5) controller.

Controls (same for every car):
    Left stick X   steering
    R2             forward throttle
    L2             reverse throttle
    Options        reset the car
    Circle         quit

Each trigger and the stick are scaled to the actuator's ctrlrange, so full R2
is the actuator's top ctrl and full L2 its bottom one:
    mushr     throttle is a torque motor (ctrl x gear), steering in rad
    neoracer  throttle is the motor's voltage (DC motor model), steering in rad

Run (macOS needs mjpython for the live viewer):
    .venv/bin/mjpython drive_joystick.py                  # MuSHR (default)
    .venv/bin/mjpython drive_joystick.py --car neoracer
    .venv/bin/mjpython drive_joystick.py --car mushr --model mushr_mujoco_ros/models/two_cars.xml
"""

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import mujoco
import mujoco.viewer

HERE = Path(__file__).resolve().parent

# Per-car settings: default scene, and the actuator names (exact, or a suffix
# such as "steering_pos" that matches "buddy_steering_pos").
CARS = {
    "mushr": {
        "model": HERE / "mushr_mujoco_ros/models/one_car.xml",
        "steer": "steering_pos",
        "throttle": "throttle_motor",
    },
    "neoracer": {
        # Completed by neoracer-mujoco's write_car (the files in its assets/
        # are templates that do not compile on their own):
        #   cd neoracer-mujoco && PYTHONPATH=src python3.14 -m neoracer_mujoco.write_car ../neoracer_cars
        "model": HERE / "neoracer_cars/neoracer.xml",
        "steer": "steer_servo",
        "throttle": "motor",
    },
}

# SDL2 axis/button layout for PS4 and PS5 controllers.
AXIS_LEFT_X = 0
AXIS_L2 = 4
AXIS_R2 = 5
BUTTON_CIRCLE = 1
BUTTON_OPTIONS = 6

MAX_STEER = 0.38  # rad, used when the steering actuator has no ctrlrange (MuSHR)
DEADZONE = 0.08


def deadzone(v: float, dz: float = DEADZONE) -> float:
    if abs(v) < dz:
        return 0.0
    return (v - dz * (1 if v > 0 else -1)) / (1 - dz)


def trigger(v: float) -> float:
    """Map a trigger axis from [-1 (released), 1 (pressed)] to [0, 1]."""
    return max(0.0, min(1.0, (v + 1.0) / 2.0))


# Shared-state slots written by the controller process.
STEER, L2, R2, CIRCLE, OPTIONS, READY = range(6)


def controller_process(state, stop) -> None:
    """Poll the controller with pygame in its own process.

    On macOS, SDL must pump events on the process's main thread, but mjpython
    runs this script off the main thread, so pygame lives in a child process.
    """
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        state[READY] = -1
        return
    js = pygame.joystick.Joystick(0)
    js.init()
    print(f"Using controller: {js.get_name()}", flush=True)
    state[READY] = 1
    parent = os.getppid()
    while not stop.is_set() and os.getppid() == parent:  # also exit if the sim is killed
        pygame.event.pump()
        state[STEER] = js.get_axis(AXIS_LEFT_X)
        state[L2] = js.get_axis(AXIS_L2)
        state[R2] = js.get_axis(AXIS_R2)
        state[CIRCLE] = js.get_button(BUTTON_CIRCLE)
        state[OPTIONS] = js.get_button(BUTTON_OPTIONS)
        time.sleep(0.005)
    pygame.quit()


def start_controller():
    state = mp.Array("d", [0.0, -1.0, -1.0, 0.0, 0.0, 0.0], lock=False)
    stop = mp.Event()
    proc = mp.Process(target=controller_process, args=(state, stop), daemon=True)
    proc.start()
    deadline = time.time() + 10
    while state[READY] == 0 and proc.is_alive() and time.time() < deadline:
        time.sleep(0.05)
    if state[READY] != 1:
        stop.set()
        raise SystemExit("No controller found. Connect it via USB or Bluetooth and retry.")
    return state, stop, proc


def find_actuator(model: mujoco.MjModel, name: str) -> int:
    """The actuator called name, else the first whose name ends in _name."""
    names = [model.actuator(i).name for i in range(model.nu)]
    if name in names:
        return names.index(name)
    for i, n in enumerate(names):
        if n.endswith("_" + name):
            return i
    raise SystemExit(f"No actuator '{name}' in model (has: {', '.join(names)})")


def ctrl_limits(model: mujoco.MjModel, act: int, default: float) -> tuple[float, float]:
    """(low, high) ctrl of an actuator: its ctrlrange, or +/-default if unlimited."""
    lo, hi = model.actuator_ctrlrange[act]
    if model.actuator_ctrllimited[act] and hi > lo:
        return float(lo), float(hi)
    return -default, default


def scale(cmd: float, limits: tuple[float, float]) -> float:
    """Map cmd in [-1, 1] to ctrl: positive onto [0, high], negative onto [low, 0]."""
    lo, hi = limits
    return cmd * hi if cmd >= 0 else -cmd * lo


def free_body(model: mujoco.MjModel) -> int:
    """The body of the first free joint: the car's chassis, for the camera to follow."""
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_bodyid[j])
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--car", choices=CARS, default="mushr", help="which car to drive (default: mushr)")
    parser.add_argument("--model", type=Path, help="MJCF scene to load instead of the car's default")
    args = parser.parse_args()

    car = CARS[args.car]
    model_path = (args.model or car["model"]).resolve()
    if not model_path.exists():
        hint = f"\n{car['model'].name} is generated; see the comment in CARS['{args.car}']." if args.model is None else ""
        raise SystemExit(f"Model not found: {model_path}{hint}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    steer_id = find_actuator(model, car["steer"])
    throttle_id = find_actuator(model, car["throttle"])
    steer_limits = ctrl_limits(model, steer_id, MAX_STEER)
    throttle_limits = ctrl_limits(model, throttle_id, 1.0)
    print(f"Driving {args.car}: {model_path.name}, steer ctrl {steer_limits}, throttle ctrl {throttle_limits}")

    state, stop, proc = start_controller()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Follow the car with the free camera.
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = free_body(model)
        viewer.cam.distance = 2.0
        viewer.cam.elevation = -20
        viewer.cam.azimuth = 180

        wall_start = time.perf_counter()
        sim_start = data.time

        while viewer.is_running():
            if state[CIRCLE]:
                break
            if state[OPTIONS]:
                mujoco.mj_resetData(model, data)
                wall_start, sim_start = time.perf_counter(), data.time

            steer = scale(-deadzone(state[STEER]), steer_limits)  # stick left = + = steer left
            throttle = scale(trigger(state[R2]) - trigger(state[L2]), throttle_limits)

            with viewer.lock():
                data.ctrl[steer_id] = steer
                data.ctrl[throttle_id] = throttle
                mujoco.mj_step(model, data)
            viewer.sync()

            # Keep the sim in step with wall-clock time.
            ahead = (data.time - sim_start) - (time.perf_counter() - wall_start)
            if ahead > 0:
                time.sleep(ahead)
            elif ahead < -0.25:  # fell far behind (e.g. window dragged): resync
                wall_start, sim_start = time.perf_counter(), data.time

    stop.set()
    proc.join(timeout=2)


if __name__ == "__main__":
    main()
