"""FastAPI app exposing the planner over HTTP.

Endpoints
---------
GET  /health                  — readiness + warmup state
POST /warmup                  — explicit warmup trigger (idempotent)
POST /world/cuboids           — push collision cuboids for an env_id
GET  /world/envs              — list known env_ids
POST /plan                    — plan to EE pose
POST /plan/joint              — plan to joint config
POST /plan/ik                 — IK solve

env_id is passed via JSON body (default = CUROBO_DEFAULT_ENV from config).
"""

from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Config
from .planner import PlannerAdapter
from .world import WorldStore


# ---- request models -------------------------------------------------------

class WorldRequest(BaseModel):
    env_id: Optional[str] = None
    cuboids: list[dict] = Field(..., description="list of cuboid dicts")
    robot_pos: Optional[list[float]] = Field(
        None, description="[x, y] of base in world frame; needed for arm-only mode")


class PlanRequest(BaseModel):
    env_id: Optional[str] = None
    current_q: list[float]
    target_pose: list[float] = Field(..., description="[x, y, z]")
    target_quat: Optional[list[float]] = Field(
        None, description="[w, x, y, z]; defaults to top-down")
    mask: str = Field("whole_body", pattern="^(whole_body|arm_only)$")


class JointPlanRequest(BaseModel):
    env_id: Optional[str] = None
    current_q: list[float]
    target_qpos: list[float]


class IKRequest(BaseModel):
    env_id: Optional[str] = None
    target_pose: list[float]
    target_quat: Optional[list[float]] = None
    mask: str = Field("whole_body", pattern="^(whole_body|arm_only)$")
    seed_q: Optional[list[float]] = None


class ValidateBaseRequest(BaseModel):
    env_id: Optional[str] = None
    base_positions: list[list[float]]
    target_pos: list[float]
    base_box: dict = Field(..., description="{center_xy, half_extents}")


# ---- app factory ----------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="cuRobo Service",
        description="GPU-accelerated motion planning over HTTP",
        version="0.1.0",
    )

    world = WorldStore(max_envs=Config.MAX_ENVS)
    planner = PlannerAdapter(device=Config.DEVICE, world_store=world)

    def _env(req_env: Optional[str]) -> str:
        return req_env or Config.DEFAULT_ENV

    # ------------------ health / warmup ------------------------------------

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "warmed_up": planner.is_warmed_up(),
            "device": Config.DEVICE,
            "robot_cfg": Config.ROBOT_CFG,
            "envs": world.list_envs(),
            "active_env": world.active(),
        }

    @app.post("/warmup")
    def warmup():
        return planner.warmup()

    # ------------------ world ----------------------------------------------

    @app.post("/world/cuboids")
    def push_world(req: WorldRequest):
        return planner.push_world(_env(req.env_id), req.cuboids, req.robot_pos)

    @app.get("/world/envs")
    def list_envs():
        return {"envs": world.list_envs(), "active": world.active()}

    # ------------------ plan -----------------------------------------------

    @app.post("/plan")
    def plan_pose(req: PlanRequest):
        try:
            return planner.plan_pose(
                env_id=_env(req.env_id),
                current_q=req.current_q,
                target_pos=req.target_pose,
                target_quat=req.target_quat,
                lock_base=(req.mask == "arm_only"),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    @app.post("/plan/joint")
    def plan_joints(req: JointPlanRequest):
        try:
            return planner.plan_joints(
                env_id=_env(req.env_id),
                current_q=req.current_q,
                target_q=req.target_qpos,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    @app.post("/plan/ik")
    def plan_ik(req: IKRequest):
        try:
            return planner.solve_ik(
                env_id=_env(req.env_id),
                target_pos=req.target_pose,
                target_quat=req.target_quat,
                lock_base=(req.mask == "arm_only"),
                seed_q=req.seed_q,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    @app.post("/plan/validate_base_path")
    def validate_base_path(req: ValidateBaseRequest):
        try:
            return planner.validate_base_path(
                env_id=_env(req.env_id),
                base_positions=req.base_positions,
                target_pos=req.target_pos,
                base_box=req.base_box,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return app


app = create_app()
