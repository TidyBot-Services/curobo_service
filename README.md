# curobo_service

Thin HTTP service that wraps NVIDIA cuRobo for GPU-accelerated motion planning.
Used by the TidyBot sim and hardware servers so all callers share a single
warmed-up planner instead of each loading their own copy.

## Why a service (vs. inlining the call)

- **Warmup amortization** — cuRobo CUDA JIT takes ~30s on first call. A persistent
  service warms up once; sim/hardware restarts don't pay it again.
- **GPU memory savings** — N sim slots × 1 cuRobo each is wasteful on small GPUs.
  Multi-target setups can share a single planner with per-env world state.
- **Crash isolation** — a cuRobo OOM or kernel error doesn't bring down the sim
  or robot driver.
- **Hardware/sim symmetry** — same planner endpoint works for both, so plan
  outputs are byte-identical.

## Quick start

```bash
git clone https://github.com/TidyBot-Services/curobo_service
cd curobo_service

# Use an env that already has a working cuRobo install. On Blackwell GPUs that
# means torch 2.10+cu128, driver 570, cuRobo built from source.
conda activate maniskill   # or whatever env has cuRobo

pip install -r requirements.txt   # adds fastapi/uvicorn only — no cuRobo here

cp .env.example .env              # tweak port/device if you need to

set -a && source .env && set +a
python -m curobo_service
# Listening on http://127.0.0.1:7000
```

Verify:

```bash
curl http://localhost:7000/health
# {"status":"ok","warmed_up":false, ...}

curl -X POST http://localhost:7000/warmup
# {"status":"warmed_up"}     # ~30s the first time
```

## API

All POST bodies are JSON. `env_id` is optional and defaults to
`CUROBO_DEFAULT_ENV`. Use it to keep separate worlds for separate sim
instances or hardware setups.

### `GET /health`

```json
{
  "status": "ok",
  "warmed_up": true,
  "device": "cuda:0",
  "robot_cfg": "franka_tidyverse.yml",
  "envs": ["sim-0", "sim-1"],
  "active_env": "sim-0"
}
```

### `POST /warmup`

Idempotent. Triggers cuRobo CUDA kernel compilation. Block ~30s on first call.

### `POST /world/cuboids`

Push collision cuboids for an env. Call this **before** any plan request.

```bash
curl -X POST http://localhost:7000/world/cuboids \
  -H "Content-Type: application/json" \
  -d '{
    "env_id": "sim-0",
    "cuboids": [
      {"name": "counter", "size": [1.0, 0.6, 1.0], "pose": [0.5, 0.0, 0.5, 1, 0, 0, 0]}
    ],
    "robot_pos": [0.0, 0.0]
  }'
```

`cuboids[i]` follows the upstream `CuroboPlanner.set_collision_world` shape —
typically `{name, size: [x, y, z], pose: [x, y, z, qw, qx, qy, qz]}`.

### `POST /plan`

Plan from `current_q` to a target EE pose.

```bash
curl -X POST http://localhost:7000/plan \
  -H "Content-Type: application/json" \
  -d '{
    "env_id": "sim-0",
    "current_q": [0, 0, 0, 0, -0.785, 0, -2.356, 0, 1.571, 0.785],
    "target_pose": [0.5, -0.2, 1.0],
    "target_quat": [0, 1, 0, 0],
    "mask": "whole_body"
  }'
```

Returns `{"status": "success", "trajectory": [[...], [...], ...]}` or
`{"status": "failed", "trajectory": null}`.

### `POST /plan/joint`

Plan to a joint configuration target.

```json
{ "env_id": "sim-0", "current_q": [...], "target_qpos": [...] }
```

### `POST /plan/ik`

IK only — no trajectory, just the goal qpos.

```json
{
  "env_id": "sim-0",
  "target_pose": [0.5, -0.2, 1.0],
  "target_quat": [0, 1, 0, 0],
  "mask": "whole_body",
  "seed_q": [...]    // optional
}
```

## Config

All via env (see `.env.example`). Sensible defaults; `CUROBO_PORT` and
`CUROBO_DEVICE` are the two you'll touch most.

| Env var | Default | Notes |
|---|---|---|
| `CUROBO_HOST` | `127.0.0.1` | Bind address. Use `0.0.0.0` for remote callers. |
| `CUROBO_PORT` | `7000` | HTTP port. |
| `CUROBO_DEVICE` | `cuda:0` | GPU for cuRobo kernels. |
| `CUROBO_ROBOT_CFG` | `franka_tidyverse.yml` | Robot YML in `assets/`. |
| `CUROBO_ASSETS_DIR` | `<package>/assets` | Where the YML + sphere configs live. |
| `CUROBO_DEFAULT_ENV` | `default` | env_id when caller doesn't supply one. |
| `CUROBO_MAX_ENVS` | `8` | Cap on concurrent worlds. |

## Architecture

```
sim-0 ─┐                           ┌─ planner CUDA kernels (warmed once)
sim-1 ─┼─► curobo_service:7000 ────┤
hw    ─┘    fastapi router         └─ per-env WorldStore
                ↓                       (cuboids + robot pose)
         PlannerAdapter
         (single CuroboPlanner,
          ensures world before plan)
```

Phase 1 keeps a single underlying `CuroboPlanner` instance. Switching
between env_ids re-calls `set_collision_world()` — cheap when one env
dominates traffic, more expensive under round-robin. Phase 2 will hold N
motion-gen configs (one per env_id) if profiling shows that matters.

## Caveats

- Only Franka Panda is wired up. Multi-robot needs config fan-out (Phase 3).
- The planner is **not** safe under concurrent requests — calls are serialized
  with a process-wide lock. Throughput is bounded by your GPU.
- This service does not validate that `current_q` matches the robot's actual
  state — that's the caller's responsibility.
