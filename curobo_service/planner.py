"""Adapter around CuroboPlanner.

Wraps the upstream planner with:
  - Lazy warmup at first request
  - Thread-safe access (cuRobo internals aren't safe under concurrent calls)
  - Per-env world swap logic (uses WorldStore)

Plan/IK methods accept the same shapes the sim already uses, so
forwarding from sim is a 1:1 substitution.
"""

from threading import Lock
from typing import Optional

import numpy as np

from .planner_core import CuroboPlanner
from .world import WorldStore


class PlannerAdapter:
    def __init__(self, device: str, world_store: WorldStore):
        self._planner = CuroboPlanner(device=device)
        self._world = world_store
        self._lock = Lock()
        self._warmed_up = False

    # ------------------------------------------------------------------
    # Warmup / health
    # ------------------------------------------------------------------
    def warmup(self) -> dict:
        with self._lock:
            if self._warmed_up:
                return {"status": "already_warmed_up"}
            self._planner.warmup()
            self._warmed_up = True
            return {"status": "warmed_up"}

    def is_warmed_up(self) -> bool:
        return self._warmed_up

    # ------------------------------------------------------------------
    # World state
    # ------------------------------------------------------------------
    def _ensure_world(self, env_id: str) -> None:
        """Load env_id's cuboids into cuRobo if it's not already active."""
        if self._world.active() == env_id:
            return
        ws = self._world.get(env_id)
        if ws is None:
            # No world set for this env — nothing to load. Caller should
            # POST /world/cuboids first.
            return
        self._planner.set_collision_world(
            ws.cuboids,
            robot_pos=ws.robot_pos,
        )
        self._world.mark_active(env_id)

    # ------------------------------------------------------------------
    # Plan APIs (mirrors CuroboPlanner)
    # ------------------------------------------------------------------
    def plan_pose(self, env_id: str, current_q: list[float],
                  target_pos: list[float], target_quat: Optional[list[float]],
                  lock_base: bool = False) -> dict:
        with self._lock:
            if not self._warmed_up:
                self._planner.warmup()
                self._warmed_up = True
            self._ensure_world(env_id)
            traj = self._planner.plan_pose(
                np.asarray(current_q, dtype=float),
                np.asarray(target_pos, dtype=float),
                np.asarray(target_quat, dtype=float) if target_quat is not None else None,
                lock_base=lock_base,
            )
            if traj is None:
                return {"status": "failed", "trajectory": None}
            return {"status": "success", "trajectory": traj.tolist()}

    def plan_joints(self, env_id: str, current_q: list[float],
                    target_q: list[float]) -> dict:
        with self._lock:
            if not self._warmed_up:
                self._planner.warmup()
                self._warmed_up = True
            self._ensure_world(env_id)
            traj = self._planner.plan_joints(
                np.asarray(current_q, dtype=float),
                np.asarray(target_q, dtype=float),
            )
            if traj is None:
                return {"status": "failed", "trajectory": None}
            return {"status": "success", "trajectory": traj.tolist()}

    def solve_ik(self, env_id: str, target_pos: list[float],
                 target_quat: Optional[list[float]],
                 lock_base: bool = False,
                 seed_q: Optional[list[float]] = None) -> dict:
        with self._lock:
            if not self._warmed_up:
                self._planner.warmup()
                self._warmed_up = True
            self._ensure_world(env_id)
            qpos = self._planner.solve_ik(
                np.asarray(target_pos, dtype=float),
                np.asarray(target_quat, dtype=float) if target_quat is not None else None,
                seed_q=np.asarray(seed_q, dtype=float) if seed_q is not None else None,
                lock_base=lock_base,
            )
            if qpos is None:
                return {"status": "failed", "qpos": None}
            return {"status": "success", "qpos": qpos.tolist()}

    def validate_base_path(self, env_id: str, base_positions: list[list[float]],
                           target_pos: list[float],
                           base_box: dict) -> dict:
        """Forward to CuroboPlanner.validate_base_path.

        Returns {collision: bool, waypoint_idx: int, fixture_name: str}.
        """
        with self._lock:
            if not self._warmed_up:
                self._planner.warmup()
                self._warmed_up = True
            self._ensure_world(env_id)
            collision, idx, name = self._planner.validate_base_path(
                np.asarray(base_positions, dtype=float),
                target_pos=np.asarray(target_pos, dtype=float),
                base_box=base_box,
            )
            return {
                "collision": bool(collision),
                "waypoint_idx": int(idx),
                "fixture_name": str(name),
            }

    # ------------------------------------------------------------------
    # World hookup
    # ------------------------------------------------------------------
    def push_world(self, env_id: str, cuboids: list[dict],
                   robot_pos: Optional[list[float]] = None) -> dict:
        self._world.set(env_id, cuboids, robot_pos)
        # Invalidate active so next plan call reloads
        if self._world.active() == env_id:
            self._world.mark_active(None)
        return {
            "status": "ok",
            "env_id": env_id,
            "cuboid_count": len(cuboids),
        }
