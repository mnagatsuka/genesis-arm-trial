import json
import os
from typing import Any, Dict, List, Optional

import genesis as gs
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


SYSTEM_PROMPT = """You are a robot arm planner.
You MUST respond ONLY with tool calls.
Do not output plain text.

Available tools:
- move_ee_to(pos, quat)
- open_gripper()
- close_gripper()

Plan step by step, safe, collision-aware, and within reachable workspace."""


# Load .env files (current working directory by default).
load_dotenv(override=False)


class OpenAIClient:
    """Thin wrapper around OpenAI tool-calling for planning."""

    def __init__(self, model: str = "gpt-5-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model = model
        self.tools_schema = self._build_tools_schema()

    @staticmethod
    def _build_tools_schema() -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "move_ee_to",
                    "description": "Move end-effector to position and orientation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pos": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3,
                            },
                            "quat": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 4,
                                "maxItems": 4,
                            },
                        },
                        "required": ["pos", "quat"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "open_gripper",
                    "description": "Open the gripper fingers.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "close_gripper",
                    "description": "Close the gripper fingers.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    @staticmethod
    def _fallback_plan(state: Dict[str, Any]) -> List[Dict[str, Any]]:
        cube_pos = state["cube_position"]
        target_pos = state["target_position"]
        return [
            {"tool": "open_gripper"},
            {
                "tool": "move_ee_to",
                "args": {"pos": [cube_pos[0], cube_pos[1], cube_pos[2] + 0.1], "quat": [1, 0, 0, 0]},
            },
            {
                "tool": "move_ee_to",
                "args": {"pos": [cube_pos[0], cube_pos[1], cube_pos[2] + 0.03], "quat": [1, 0, 0, 0]},
            },
            {"tool": "close_gripper"},
            {"tool": "move_ee_to", "args": {"pos": target_pos, "quat": [1, 0, 0, 0]}},
            {"tool": "open_gripper"},
        ]

    def _build_messages(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"State: {state}. Task: pick the cube and place it at {state['target_position']}.",
            },
        ]

    def plan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call OpenAI with tool-calls; fallback to deterministic plan on failure."""
        if self.client is None:
            print("[LLM] OPENAI_API_KEY not set; using deterministic fallback plan.")
            return self._fallback_plan(state)

        messages = self._build_messages(state)
        response = None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools_schema,
                tool_choice="auto",
            )
        except Exception as exc:
            print(f"[LLM] API error; using deterministic fallback plan: {exc}")
            return self._fallback_plan(state)

        if response is None or not getattr(response, "choices", None):
            print("[LLM] Empty response; using deterministic fallback plan.")
            return self._fallback_plan(state)

        tool_calls = response.choices[0].message.tool_calls or []
        if not tool_calls:
            print("[LLM] No tool calls; using deterministic fallback plan.")
            return self._fallback_plan(state)

        plan: List[Dict[str, Any]] = []
        for call in tool_calls:
            name = call.function.name
            raw_args = call.function.arguments if hasattr(call.function, "arguments") else {}
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except Exception:
                args = {}
            plan.append({"tool": name, "args": args})

        return plan


class FrankaWrapper:
    """Wraps common Franka operations and state queries."""

    def __init__(self, scene: gs.Scene):
        self.scene = scene
        self.franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
        self.end_effector = self.franka.get_link("hand")
        self.arm_dofs = np.arange(7)
        self.gripper_dofs = np.array([7, 8])

    def configure(self) -> None:
        self.franka.set_dofs_kp(
            np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0]),
        )
        self.franka.set_dofs_kv(
            np.array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0]),
        )

    def move_ee_to(self, pos, quat, steps: int = 240) -> None:
        qpos = self.franka.inverse_kinematics(
            link=self.end_effector,
            pos=np.array(pos),
            quat=np.array(quat),
        )
        self.franka.control_dofs_position(qpos[:-2], self.arm_dofs)
        for _ in range(steps):
            self.scene.step()

    def open_gripper(self, steps: int = 120) -> None:
        self.franka.control_dofs_position([0.04, 0.04], self.gripper_dofs)
        for _ in range(steps):
            self.scene.step()

    def close_gripper(self, steps: int = 120) -> None:
        self.franka.control_dofs_position([0.0, 0.0], self.gripper_dofs)
        for _ in range(steps):
            self.scene.step()

    def get_state(self, cube) -> Dict[str, Any]:
        ee_pos = self.end_effector.get_pos()
        cube_pos = cube.get_pos()
        target_pos = [0.4, 0.2, 0.3]
        return {
            "end_effector": np.round(ee_pos, 3).tolist(),
            "cube_position": np.round(cube_pos, 3).tolist(),
            "target_position": target_pos,
            "gripper_open": True,
        }


def main():
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=1 / 240),
        viewer_options=gs.options.ViewerOptions(camera_pos=(2, 2, 2)),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())
    cube = scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.5, 0.0, 0.02)))

    franka = FrankaWrapper(scene=scene)

    scene.build()
    franka.configure()

    client = OpenAIClient()

    tools = {
        "move_ee_to": franka.move_ee_to,
        "open_gripper": franka.open_gripper,
        "close_gripper": franka.close_gripper,
    }

    max_iters = 5
    success = False

    for _ in range(max_iters):
        state = franka.get_state(cube)
        cube_pos = np.array(state["cube_position"])
        target_pos = np.array(state["target_position"])

        if np.linalg.norm(cube_pos - target_pos) < 0.02:
            print("[Loop] Goal reached; stopping.")
            success = True
            break

        plan = client.plan(state)
        for action in plan:
            tool = action["tool"]
            args = action.get("args", {})
            if tool not in tools:
                print(f"[LLM] Unknown tool '{tool}', skipping.")
                continue
            tools[tool](**args)

    if not success:
        print("[Loop] Max iterations reached without achieving target.")


if __name__ == "__main__":
    main()
