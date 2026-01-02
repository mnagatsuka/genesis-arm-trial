Below is a **hands-on, end-to-end tutorial** for **Embodied AI with Genesis + LLM**, focused on **robot-arm manipulation** using **current best practices** (LLM as planner, Genesis as executor, classical control underneath).

This is intentionally **minimal but correct**, so you can extend it toward VLA later.


## What you’ll build

A system where:

* You give a **natural-language instruction**
* An **LLM plans actions** using tools
* **Genesis executes** those actions via IK + PD control
* The system **replans** based on state feedback

Example:

> “Pick up the cube and place it at (0.4, 0.2, 0.3)”

---

## Architecture (best practice, 2025)

```
User
 ↓
LLM (Planner / Reasoner)
 ↓   (tool calls)
Action Executor (Python)
 ↓
Genesis (IK + PD control)
 ↓
World State → feedback → LLM
```

Key principle:
LLM **never** outputs torques or joint velocities
LLM **only** outputs symbolic actions

---

## Step 0 — Prerequisites

* Python 3.12+
* Genesis installed
* Any LLM API (OpenAI, Gemini, local LLM)

You already finished:

```
Genesis/examples/*
```

So we’ll build *on top* of that.

---

## Step 1 — Create a minimal Genesis scene

### 1.1 Scene + robot + object

```python
import genesis as gs
import numpy as np

gs.init(backend=gs.cpu)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(dt=1/240),
    viewer_options=gs.options.ViewerOptions(camera_pos=(2, 2, 2))
)

robot = scene.add_entity(
    gs.morphs.MJCF(file="assets/franka_panda.xml"),
    pos=(0, 0, 0)
)

cube = scene.add_entity(
    gs.morphs.Box(size=(0.04, 0.04, 0.04)),
    pos=(0.5, 0.0, 0.02)
)

scene.build()
```

We’ll assume:

* End effector link name: `"panda_hand"`
* Gripper joints exist

---

## Step 2 — Low-level motion primitive (IK + PD)

This is the **most important part**.

### 2.1 Move end-effector to a pose

```python
def move_ee_to(pos, quat, steps=240):
    qpos = robot.inverse_kinematics(
        link="panda_hand",
        pos=pos,
        quat=quat
    )

    arm_dofs = robot.get_dofs_idx("arm")
    robot.control_dofs_position(qpos[arm_dofs], arm_dofs)

    for _ in range(steps):
        scene.step()
```

Why this is best practice:

* IK gives feasibility
* PD control is stable
* Deterministic and debuggable

---

## Step 3 — Gripper primitives

```python
def open_gripper():
    gripper_dofs = robot.get_dofs_idx("gripper")
    robot.control_dofs_position([0.04, 0.04], gripper_dofs)
    for _ in range(120):
        scene.step()

def close_gripper():
    gripper_dofs = robot.get_dofs_idx("gripper")
    robot.control_dofs_position([0.0, 0.0], gripper_dofs)
    for _ in range(120):
        scene.step()
```

---

## Step 4 — Define **high-level tools** for the LLM

This is the **interface contract**.

```python
TOOLS = {
    "move_ee_to": move_ee_to,
    "open_gripper": open_gripper,
    "close_gripper": close_gripper,
}
```

LLM is allowed to call **only these**.

---

## Step 5 — World state abstraction (critical)

LLMs need **structured state**, not raw simulator data.

```python
def get_state():
    ee_pos = robot.get_link("panda_hand").get_pos()
    cube_pos = cube.get_pos()

    return {
        "end_effector": np.round(ee_pos, 3).tolist(),
        "cube_position": np.round(cube_pos, 3).tolist(),
        "gripper_open": True
    }
```

This enables:

* Replanning
* Failure recovery
* Debugging

---

## Step 6 — LLM prompt (planner role)

### 6.1 Tool-only system prompt

```text
You are a robot arm planner.

You may ONLY respond using tool calls.
Do not explain.
Do not output text.

Available tools:
- move_ee_to(pos, quat)
- open_gripper()
- close_gripper()

Goal: complete the task safely and step by step.
```

### 6.2 User instruction

```text
Pick up the cube and place it at position (0.4, 0.2, 0.3)
```

---

## Step 7 — Example LLM output (ideal)

```json
[
  {"tool": "open_gripper"},
  {
    "tool": "move_ee_to",
    "args": {
      "pos": [0.5, 0.0, 0.1],
      "quat": [1, 0, 0, 0]
    }
  },
  {
    "tool": "move_ee_to",
    "args": {
      "pos": [0.5, 0.0, 0.03],
      "quat": [1, 0, 0, 0]
    }
  },
  {"tool": "close_gripper"},
  {
    "tool": "move_ee_to",
    "args": {
      "pos": [0.4, 0.2, 0.3],
      "quat": [1, 0, 0, 0]
    }
  },
  {"tool": "open_gripper"}
]
```

---

## Step 8 — Tool execution loop

```python
for action in llm_plan:
    tool = action["tool"]
    args = action.get("args", {})
    TOOLS[tool](**args)
```

Add **validators** here:

* workspace bounds
* IK success
* max step distance

---

## Step 9 — Replanning loop (real embodied AI)

```python
for _ in range(5):
    state = get_state()
    plan = call_llm(state)
    success = execute(plan)
    if success:
        break
```

This loop is what makes it **embodied**, not scripted.

---

## Why this is current best practice

| Approach                        | Status              |
| ------------------------------- | ------------------- |
| LLM → joint torques             | ❌ unstable          |
| VLA end-to-end                  | ⚠️ needs large data |
| LLM planner + classical control | ✅ industry standard |
| Hybrid (LLM + VLA)              | 🔥 cutting edge     |

Genesis fits **perfectly** into this architecture.

---

## How to evolve this next

1. Add **camera observations**
2. Replace planner with **Gemini Robotics-ER**
3. Replace executor with **OpenVLA / diffusion policy**
4. Add **collision-aware motion planning**
5. Train policies in Genesis → deploy on real arm

---

## Mental model to keep

* LLM = *brain*
* Genesis = *body*
* IK / PD = *muscles*
* State feedback = *senses*

---

## Calling an LLM API (OpenAI example)

Use the helper script `tutorials/llm/llm_robot_arm.py`:

1. Install deps: `pip install openai python-dotenv`
2. Create `.env` in the repo root with `OPENAI_API_KEY=...`
3. Run: `python tutorials/llm/llm_robot_arm.py`

Notes:

* Uses `gpt-5-mini` with tool-calling; only the defined tools are allowed.
* The script auto-loads `.env` (repo root or CWD) before reading env vars via `python-dotenv`.
* If the key/SDK is missing or the API errors, it falls back to a deterministic plan.
* Tools exposed: `move_ee_to`, `open_gripper`, `close_gripper`.

If you want, next I can:

* Convert this into a **full repo layout**
* Show **Genesis → OpenVLA integration**
* Add **vision-language planning**
* Compare **Genesis vs Isaac Gym vs MuJoCo** for embodied AI
