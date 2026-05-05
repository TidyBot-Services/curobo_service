"""cuRobo v0.8.0 (cuRoboV2) motion planner for Franka Panda on mobile base.

Public API identical to v0.7.6 planner_core.py — server.py / world.py untouched.

Internal v1 → v0.8 mapping:
  MotionGen, MotionGenConfig          → MotionPlanner, MotionPlannerCfg.create
  WorldConfig                          → Scene
  Cuboid                               → same constructor
  motion_gen.plan_single(s, g, cfg)   → planner.plan_pose(GoalToolPose, s, max_attempts)
  motion_gen.plan_single_js           → planner.plan_cspace
  motion_gen.solve_ik                  → planner.ik_solver.solve_pose
  motion_gen.update_world(WC)         → planner.update_world(Scene)
  Pose.from_list([7])                 → Pose(position=[1,3]_tensor, quaternion=[1,4]_tensor)
  Goal wrap                            → GoalToolPose.from_poses({tool: Pose}, num_goalset=1)
  update_locked_joints({...}, cfg)    → planner.kinematics.update_kinematics_config(new_KinematicsCfg)
"""

import numpy as np
import torch
import time
from typing import Optional


class CuroboPlanner:
    """GPU-accelerated motion planner using NVIDIA cuRobo v0.8 (cuRoboV2)."""

    _FINGER_LOCKS = {"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04}
    _ROBOT_CFG_YML = "franka_tidyverse.yml"

    def __init__(self, device: str = "cuda:0"):
        self._device = torch.device(device)
        self._planner = None         # whole-body (base free)
        self._planner_arm = None     # arm-only (base_x/y/z locked)
        self._arm_only_cfg_dict = None
        self._tool_frame = None      # cached EE link name (single-tool robot)
        self._tool_frame_arm = None
        self._warmed_up = False
        self._world_cuboids = []
        self._last_arm_base_lock = None  # (x, y, z) of last lock update

    # ------------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------------

    def warmup(self):
        if self._warmed_up:
            return

        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.scene import Scene, Cuboid

        t0 = time.time()
        print("[curobo v0.8] Loading franka_tidyverse config...")

        # cuRobo requires a non-empty initial scene for cfg construction
        dummy_scene = Scene(cuboid=[
            Cuboid(name="ground", pose=[0, 0, -0.5, 1, 0, 0, 0], dims=[10, 10, 0.01])
        ])

        # Whole-body planner (base free)
        cfg = MotionPlannerCfg.create(
            robot=self._ROBOT_CFG_YML,
            scene_model=dummy_scene,
            collision_cache={"cuboid": 100, "mesh": 10},
            optimizer_collision_activation_distance=0.01,
            use_cuda_graph=False,  # allow runtime obstacle / lock changes
        )
        self._planner = MotionPlanner(cfg)
        self._tool_frame = self._planner.tool_frames[0]

        print("[curobo v0.8] Warming up CUDA kernels (whole_body)...")
        self._planner.warmup(enable_graph=True, num_warmup_iterations=3)

        # Arm-only planner (base joints locked).  Same lock-set cardinality
        # is used at init so update_kinematics_config can swap VALUES later.
        print("[curobo v0.8] Loading arm-only planner (base joints locked)...")
        from curobo._src.util_file import (
            get_robot_configs_path, join_path, load_yaml,
        )
        cfg_dict = load_yaml(join_path(get_robot_configs_path(), self._ROBOT_CFG_YML))
        arm_locks = dict(self._FINGER_LOCKS)
        arm_locks.update({"base_x": 0.0, "base_y": 0.0, "base_z": 0.0})
        cfg_dict["robot_cfg"]["kinematics"]["lock_joints"] = arm_locks
        self._arm_only_cfg_dict = cfg_dict

        cfg_arm = MotionPlannerCfg.create(
            robot=cfg_dict["robot_cfg"],     # MotionPlannerCfg.create accepts dict
            scene_model=dummy_scene,
            collision_cache={"cuboid": 100, "mesh": 10},
            optimizer_collision_activation_distance=0.01,
            use_cuda_graph=False,
        )
        self._planner_arm = MotionPlanner(cfg_arm)
        self._tool_frame_arm = self._planner_arm.tool_frames[0]
        self._planner_arm.warmup(enable_graph=True, num_warmup_iterations=3)

        dt = time.time() - t0
        print(f"[curobo v0.8] Ready ({dt:.1f}s, whole_body + arm_only)")
        self._warmed_up = True

    # ------------------------------------------------------------------
    # Collision world
    # ------------------------------------------------------------------

    def set_collision_world(self, cuboids: list[dict],
                            robot_pos: Optional[np.ndarray] = None,
                            max_distance: float = 3.0):
        """Update collision world with kitchen fixture cuboids."""
        from curobo.scene import Scene, Cuboid as CuCuboid

        self._world_cuboids = cuboids

        cu_cuboids = []
        skipped_overlap = 0
        skipped_far = 0
        for c in cuboids:
            cx, cy, cz = c["center"]
            hx, hy, hz = c["half_size"]
            quat = c.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0])

            if robot_pos is not None:
                rx, ry = robot_pos[0], robot_pos[1]

                # Skip cuboids that overlap with robot position (cause start-state collision)
                margin = 0.15  # 15cm margin
                if (cx - hx - margin < rx < cx + hx + margin and
                    cy - hy - margin < ry < cy + hy + margin):
                    skipped_overlap += 1
                    continue
                dist = np.sqrt((cx - rx) ** 2 + (cy - ry) ** 2)
                if dist > max_distance:
                    skipped_far += 1
                    continue

            cu_cuboids.append(CuCuboid(
                name=c["name"],
                pose=[cx, cy, cz, quat[0], quat[1], quat[2], quat[3]],
                dims=[hx * 2, hy * 2, hz * 2],
            ))

        scene = Scene(cuboid=cu_cuboids)
        self._planner.update_world(scene)
        if self._planner_arm is not None:
            self._planner_arm.update_world(scene)
        print(f"[curobo v0.8] Updated collision world: {len(cu_cuboids)} cuboids "
              f"(skipped {skipped_overlap} overlap, {skipped_far} far)")

    # ------------------------------------------------------------------
    # validate_base_path — UNCHANGED (pure numpy SAT)
    # ------------------------------------------------------------------

    def validate_base_path(self, base_positions: np.ndarray,
                           target_pos: Optional[np.ndarray] = None,
                           base_radius: float = 0.20,
                           target_exclusion_radius: float = 0.15,
                           base_box: Optional[dict] = None) -> tuple:
        if not self._world_cuboids or len(base_positions) <= 1:
            return False, -1, ""

        use_obb = base_box is not None
        if use_obb:
            box_offset = np.array(base_box.get("center_xy", [0.0, 0.0]))
            box_half = np.array(base_box["half_extents"])

        start_x = float(base_positions[0, 0])
        start_y = float(base_positions[0, 1])
        start_yaw = float(base_positions[0, 2]) if base_positions.shape[1] > 2 else 0.0

        if use_obb:
            cos_s, sin_s = np.cos(start_yaw), np.sin(start_yaw)
            start_ox = start_x + cos_s * box_offset[0] - sin_s * box_offset[1]
            start_oy = start_y + sin_s * box_offset[0] + cos_s * box_offset[1]
            start_x_axis = (cos_s, sin_s)
            start_y_axis = (-sin_s, cos_s)
            start_margin = 0.05

        start_inside = set()
        for c in self._world_cuboids:
            cx, cy = c["center"][0], c["center"][1]
            hx, hy = c["half_size"][0], c["half_size"][1]
            cz, hz = c["center"][2], c["half_size"][2]
            if cz + hz < 0.0 or cz - hz > 0.5:
                continue
            if use_obb:
                hx_m = hx + start_margin
                hy_m = hy + start_margin
                dx_c = cx - start_ox
                dy_c = cy - start_oy
                overlap = True
                for ax, ay in [(1.0, 0.0), (0.0, 1.0), start_x_axis, start_y_axis]:
                    dist = abs(dx_c * ax + dy_c * ay)
                    aabb_proj = hx_m * abs(ax) + hy_m * abs(ay)
                    obb_proj = (box_half[0] * abs(start_x_axis[0] * ax + start_x_axis[1] * ay) +
                                box_half[1] * abs(start_y_axis[0] * ax + start_y_axis[1] * ay))
                    if dist > obb_proj + aabb_proj:
                        overlap = False
                        break
                if overlap:
                    start_inside.add(c["name"])
            else:
                if (cx - hx - base_radius <= start_x <= cx + hx + base_radius and
                    cy - hy - base_radius <= start_y <= cy + hy + base_radius):
                    start_inside.add(c["name"])

        active_cuboids = []
        for c in self._world_cuboids:
            name_lower = c["name"].lower()
            if "wall" in name_lower or "floor" in name_lower:
                continue
            if c["name"] in start_inside:
                continue
            if target_pos is not None:
                cx, cy = c["center"][0], c["center"][1]
                tdx = cx - float(target_pos[0])
                tdy = cy - float(target_pos[1])
                if tdx * tdx + tdy * tdy < target_exclusion_radius ** 2:
                    continue
            active_cuboids.append(c)

        r2 = base_radius * base_radius

        for idx in range(1, len(base_positions)):
            x, y = float(base_positions[idx, 0]), float(base_positions[idx, 1])
            yaw = float(base_positions[idx, 2]) if base_positions.shape[1] > 2 else 0.0

            if use_obb:
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                ox = x + cos_y * box_offset[0] - sin_y * box_offset[1]
                oy = y + sin_y * box_offset[0] + cos_y * box_offset[1]
                obb_x_axis = (cos_y, sin_y)
                obb_y_axis = (-sin_y, cos_y)

            for c in active_cuboids:
                cx, cy, cz = c["center"]
                hx, hy, hz = c["half_size"]
                if cz + hz < 0.0 or cz - hz > 0.5:
                    continue
                if use_obb:
                    dx_c = cx - ox
                    dy_c = cy - oy
                    overlap = True
                    for ax, ay in [(1.0, 0.0), (0.0, 1.0), obb_x_axis, obb_y_axis]:
                        dist = abs(dx_c * ax + dy_c * ay)
                        aabb_proj = hx * abs(ax) + hy * abs(ay)
                        obb_proj = (box_half[0] * abs(obb_x_axis[0] * ax + obb_x_axis[1] * ay) +
                                    box_half[1] * abs(obb_y_axis[0] * ax + obb_y_axis[1] * ay))
                        if dist > obb_proj + aabb_proj:
                            overlap = False
                            break
                    if overlap:
                        return True, idx, c["name"]
                else:
                    dx_c = max(cx - hx - x, 0.0, x - (cx + hx))
                    dy_c = max(cy - hy - y, 0.0, y - (cy + hy))
                    if dx_c * dx_c + dy_c * dy_c < r2:
                        return True, idx, c["name"]

        return False, -1, ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_collision_for_target(self, target_pos: np.ndarray,
                                     robot_pos: Optional[np.ndarray] = None):
        """Update collision world, excluding cuboids that contain the target."""
        from curobo.scene import Scene, Cuboid as CuCuboid

        if not self._world_cuboids:
            return

        margin = 0.05
        cu_cuboids = []
        skipped_target = 0
        skipped_overlap = 0
        skipped_far = 0
        for c in self._world_cuboids:
            cx, cy, cz = c["center"]
            hx, hy, hz = c["half_size"]
            tx, ty, tz = target_pos
            target_inside = (
                abs(cx - tx) < hx + margin and
                abs(cy - ty) < hy + margin and
                abs(cz - tz) < hz + margin
            )
            if target_inside:
                skipped_target += 1
                continue

            if robot_pos is not None:
                rx, ry = robot_pos[0], robot_pos[1]
                rm = 0.15
                if (cx - hx - rm < rx < cx + hx + rm and
                    cy - hy - rm < ry < cy + hy + rm):
                    skipped_overlap += 1
                    continue
                dist = np.sqrt((cx - rx) ** 2 + (cy - ry) ** 2)
                if dist > 3.0:
                    skipped_far += 1
                    continue

            quat = c.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0])
            cu_cuboids.append(CuCuboid(
                name=c["name"],
                pose=[cx, cy, cz, quat[0], quat[1], quat[2], quat[3]],
                dims=[hx * 2, hy * 2, hz * 2],
            ))

        scene = Scene(cuboid=cu_cuboids)
        self._planner.update_world(scene)
        if self._planner_arm is not None:
            self._planner_arm.update_world(scene)
        print(f"[curobo v0.8] Collision world for target: {len(cu_cuboids)} cuboids "
              f"(excl {skipped_target} at target, {skipped_overlap} overlap, {skipped_far} far)")

    def _update_arm_base_lock(self, base_x: float, base_y: float, base_z: float):
        """Swap base_x/y/z lock values on arm-only planner (no rebuild).

        v0.8 API:  Kinematics.update_kinematics_config takes KinematicsParams,
        which lives at KinematicsCfg.kinematics_config.  The .copy_() inside
        does in-place lock_jointstate copy + validate_shapes() (same num_dof
        + same lock-set cardinality required, only values change).
        """
        from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
        from curobo._src.robot.loader.kinematics_loader_cfg import KinematicsLoaderCfg

        kin_dict = dict(self._arm_only_cfg_dict["robot_cfg"]["kinematics"])
        arm_locks = dict(self._FINGER_LOCKS)
        arm_locks["base_x"] = base_x
        arm_locks["base_y"] = base_y
        arm_locks["base_z"] = base_z
        kin_dict["lock_joints"] = arm_locks

        device_cfg = self._planner_arm.config.device_cfg
        loader_cfg = KinematicsLoaderCfg(**kin_dict, device_cfg=device_cfg)
        new_full_cfg = KinematicsCfg.from_config(loader_cfg)
        # Inner KinematicsParams is what update_kinematics_config expects
        self._planner_arm.kinematics.update_kinematics_config(new_full_cfg.kinematics_config)

    def _select_planner(self, lock_base: bool, current_q: np.ndarray):
        """Returns (planner, q_active, tool_frame). Updates arm_only base lock if needed."""
        if lock_base:
            if self._planner_arm is None:
                raise RuntimeError("lock_base=True requires warmup() to have completed")
            base_key = (float(current_q[0]), float(current_q[1]), float(current_q[2]))
            if base_key != self._last_arm_base_lock:
                self._update_arm_base_lock(*base_key)
                self._last_arm_base_lock = base_key
            return self._planner_arm, np.asarray(current_q[3:10], dtype=np.float32), self._tool_frame_arm
        return self._planner, np.asarray(current_q[:10], dtype=np.float32), self._tool_frame

    def _widen(self, partial: np.ndarray, current_q: np.ndarray,
               lock_base: bool) -> np.ndarray:
        if not lock_base:
            return partial
        if partial.ndim == 1:
            full = np.zeros(10, dtype=partial.dtype)
            full[:3] = current_q[:3]
            full[3:] = partial
            return full
        T = partial.shape[0]
        full = np.zeros((T, 10), dtype=partial.dtype)
        full[:, :3] = current_q[:3]
        full[:, 3:] = partial
        return full

    # ------------------------------------------------------------------
    # plan_pose
    # ------------------------------------------------------------------

    def plan_pose(self, current_q: np.ndarray, target_pos: np.ndarray,
                  target_quat: np.ndarray, lock_base: bool = False,
                  max_attempts: int = 10) -> Optional[np.ndarray]:
        from curobo.types import Pose, JointState, GoalToolPose

        if not self._warmed_up:
            self.warmup()

        robot_pos = current_q[:2] if len(current_q) >= 2 else None
        self._update_collision_for_target(target_pos, robot_pos=robot_pos)

        planner, q_active, tool_frame = self._select_planner(lock_base, current_q)
        joint_names = planner.joint_names

        q_tensor = torch.as_tensor(q_active, dtype=torch.float32,
                                   device=self._device).unsqueeze(0)
        current_state = JointState.from_position(q_tensor, joint_names=joint_names)

        pos_t = torch.as_tensor(np.asarray(target_pos, dtype=np.float32).reshape(1, 3),
                                device=self._device)
        quat_t = torch.as_tensor(np.asarray(target_quat, dtype=np.float32).reshape(1, 4),
                                 device=self._device)
        goal_pose = Pose(position=pos_t, quaternion=quat_t)
        goal = GoalToolPose.from_poses({tool_frame: goal_pose}, num_goalset=1)

        t0 = time.time()
        try:
            result = planner.plan_pose(goal, current_state, max_attempts=max_attempts)
            # Mirror v0.7.6 fallback: if start state in collision, retry with cleared world
            if result is None or not bool(result.success.any()):
                from curobo.scene import Scene as _Sc, Cuboid as _Cb
                print("[curobo v0.8] Plan failed, retrying with cleared world")
                planner.update_world(_Sc(cuboid=[
                    _Cb(name="ground", pose=[0, 0, -1, 1, 0, 0, 0], dims=[0.01, 0.01, 0.01])
                ]))
                result = planner.plan_pose(goal, current_state, max_attempts=max_attempts)
                # Restore the real collision world so future calls see obstacles
                if self._world_cuboids:
                    self._update_collision_for_target(target_pos, robot_pos=robot_pos)
        except Exception as e:
            dt = time.time() - t0
            print(f"[curobo v0.8] Plan ERROR ({dt:.2f}s): {e}")
            import traceback; traceback.print_exc()
            return None
        dt = time.time() - t0

        if result is None or not bool(result.success.any()):
            print(f"[curobo v0.8] Plan FAILED ({dt:.2f}s)")
            return None

        traj = result.get_interpolated_plan()
        # v0.8 trajectory shape: (batch, num_seeds, horizon, full_dof_with_locks)
        # Pull single solution: index [0, 0] → (horizon, full_dof)
        # Slice to active DOF (drop locked finger columns) using planner.joint_names size
        pos_full = traj.position
        while pos_full.ndim > 2:
            pos_full = pos_full[0]
        positions = pos_full.cpu().numpy()
        if positions.shape[1] > planner.action_dim:
            positions = positions[:, :planner.action_dim]
        positions = self._widen(positions, current_q, lock_base)
        print(f"[curobo v0.8] Plan OK ({dt:.2f}s, {'arm_only' if lock_base else 'whole_body'}): "
              f"{positions.shape[0]} waypoints")
        return positions

    # ------------------------------------------------------------------
    # plan_joints
    # ------------------------------------------------------------------

    def plan_joints(self, current_q: np.ndarray,
                    target_q: np.ndarray,
                    lock_base: bool = False) -> Optional[np.ndarray]:
        from curobo.types import JointState

        if not self._warmed_up:
            self.warmup()

        planner, q_active, _ = self._select_planner(lock_base, current_q)
        joint_names = planner.joint_names
        target_active = np.asarray(target_q[3:10] if lock_base else target_q[:10],
                                   dtype=np.float32)

        current_state = JointState.from_position(
            torch.as_tensor(q_active, dtype=torch.float32,
                            device=self._device).unsqueeze(0),
            joint_names=joint_names,
        )
        goal_state = JointState.from_position(
            torch.as_tensor(target_active, dtype=torch.float32,
                            device=self._device).unsqueeze(0),
            joint_names=joint_names,
        )

        t0 = time.time()
        try:
            result = planner.plan_cspace(goal_state, current_state, max_attempts=10)
            if result is None or not bool(result.success.any()):
                from curobo.scene import Scene as _Sc, Cuboid as _Cb
                print("[curobo v0.8] Joint plan failed, retrying with cleared world")
                planner.update_world(_Sc(cuboid=[
                    _Cb(name="ground", pose=[0, 0, -1, 1, 0, 0, 0], dims=[0.01, 0.01, 0.01])
                ]))
                result = planner.plan_cspace(goal_state, current_state, max_attempts=10)
                # Restore real world
                if self._world_cuboids:
                    robot_pos = (np.asarray(current_q[:2], dtype=np.float32)
                                 if len(current_q) >= 2 else None)
                    self.set_collision_world(self._world_cuboids, robot_pos=robot_pos)
        except Exception as e:
            dt = time.time() - t0
            print(f"[curobo v0.8] Joint plan ERROR ({dt:.2f}s): {e}")
            import traceback; traceback.print_exc()
            return None
        dt = time.time() - t0

        if result is None or not bool(result.success.any()):
            print(f"[curobo v0.8] Joint plan FAILED ({dt:.2f}s)")
            return None

        traj = result.get_interpolated_plan()
        pos_full = traj.position
        while pos_full.ndim > 2:
            pos_full = pos_full[0]
        positions = pos_full.cpu().numpy()
        if positions.shape[1] > planner.action_dim:
            positions = positions[:, :planner.action_dim]
        positions = self._widen(positions, current_q, lock_base)
        print(f"[curobo v0.8] Joint plan OK ({dt:.2f}s, {'arm_only' if lock_base else 'whole_body'}): "
              f"{positions.shape[0]} waypoints")
        return positions

    # ------------------------------------------------------------------
    # solve_ik
    # ------------------------------------------------------------------

    def solve_ik(self, target_pos: np.ndarray, target_quat: np.ndarray,
                 current_q: Optional[np.ndarray] = None,
                 lock_base: bool = False,
                 num_seeds: int = 40,
                 return_closest: bool = True) -> Optional[np.ndarray]:
        from curobo.types import Pose, GoalToolPose, JointState

        if not self._warmed_up:
            self.warmup()

        if lock_base and current_q is None:
            raise ValueError("lock_base=True requires current_q to pin base values")

        if self._world_cuboids:
            robot_pos = current_q[:2] if current_q is not None and len(current_q) >= 2 else None
            self._update_collision_for_target(np.asarray(target_pos), robot_pos=robot_pos)

        if lock_base:
            planner, _, tool_frame = self._select_planner(True, current_q)
        else:
            planner, tool_frame = self._planner, self._tool_frame

        pos_t = torch.as_tensor(np.asarray(target_pos, dtype=np.float32).reshape(1, 3),
                                device=self._device)
        quat_t = torch.as_tensor(np.asarray(target_quat, dtype=np.float32).reshape(1, 4),
                                 device=self._device)
        goal_pose = Pose(position=pos_t, quaternion=quat_t)
        goal = GoalToolPose.from_poses({tool_frame: goal_pose}, num_goalset=1)

        want_multi = bool(return_closest and current_q is not None)
        return_seeds = min(num_seeds, 8) if want_multi else 1

        if current_q is not None:
            q_active = (np.asarray(current_q[3:10], dtype=np.float32) if lock_base
                        else np.asarray(current_q[:10], dtype=np.float32))
            current_state = JointState.from_position(
                torch.as_tensor(q_active, device=self._device).unsqueeze(0),
                joint_names=planner.joint_names,
            )
        else:
            current_state = None

        t0 = time.time()
        try:
            result = planner.ik_solver.solve_pose(
                goal,
                return_seeds=return_seeds,
                current_state=current_state,
            )
        except Exception as e:
            print(f"[curobo v0.8] IK ERROR ({time.time()-t0:.2f}s): {e}")
            return None
        dt = time.time() - t0

        if not bool(result.success.any()):
            print(f"[curobo v0.8] IK FAILED ({dt:.2f}s, num_seeds={num_seeds})")
            return None

        # result.solution shape: (batch, return_seeds, dof) → squeeze batch
        sol = result.solution.squeeze(0).cpu().numpy()
        succ = result.success.squeeze(0).cpu().numpy().astype(bool)
        sol_ok = sol[succ]
        if sol_ok.shape[0] == 0:
            return None

        if want_multi:
            ref = current_q[3:10] if lock_base else current_q[:10]
            ref = np.asarray(ref, dtype=sol_ok.dtype)
            dists = np.linalg.norm(sol_ok - ref[None, :], axis=1)
            chosen = sol_ok[int(np.argmin(dists))]
        else:
            chosen = sol_ok[0]

        print(f"[curobo v0.8] IK OK ({dt:.2f}s, num_seeds={num_seeds}, "
              f"{int(succ.sum())}/{len(succ)} succeeded)")

        return self._widen(chosen, current_q if current_q is not None else np.zeros(10),
                           lock_base)


# ---------------------------------------------------------------------------
# build_collision_cuboids_from_fixtures — UNCHANGED (numpy + sapien only)
# ---------------------------------------------------------------------------

def build_collision_cuboids_from_fixtures(scene, fixtures_dict) -> list[dict]:
    """Extract kitchen fixture AABBs and convert to cuboid format."""
    from maniskill_tidyverse.planning_utils import _compute_fixture_aabb

    cuboids = []
    for fname, fix in fixtures_dict.items():
        try:
            from robocasa.models.fixtures import Floor, Wall
            if isinstance(fix, (Floor, Wall)):
                continue
        except ImportError:
            pass

        if not hasattr(fix, 'pos'):
            continue

        bbox_min, bbox_max = _compute_fixture_aabb(scene, fname)

        if bbox_min is None and hasattr(fix, 'size') and fix.size is not None:
            pos = np.array(fix.pos)
            size = np.array(fix.size)
            if len(size) == 3 and np.all(size > 0.01):
                fix_half = size / 2.0
                bbox_min = pos - fix_half
                bbox_max = pos + fix_half

        if bbox_min is None:
            continue

        center = (bbox_min + bbox_max) / 2.0
        half_size = (bbox_max - bbox_min) / 2.0

        if np.any(half_size < 0.005) or np.any(half_size > 5.0):
            continue

        cuboids.append({
            "name": f"fixture_{fname}",
            "center": center.tolist(),
            "half_size": half_size.tolist(),
        })

    print(f"[curobo v0.8] Extracted {len(cuboids)} fixture cuboids")
    return cuboids
