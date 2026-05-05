"""Apply curobo_service-shipped configuration patches to the local cuRobo
installation. Runs once on service startup. Idempotent.

Why this exists
---------------
Two cuRobo v0.8.0 default configs cause real-pipeline breakage that
synthetic benchmarks don't surface:

1. `task/metrics_base.yml`: the PRM feasibility check uses
   `cspace_cfg.activation_distance: [0.0]*5`, so a joint exactly at its
   limit (which IK solvers often produce, e.g. panda_joint4 == -0.07)
   trips constraint validation. The error message reads "Start or End
   state in collision" but is a joint-limit issue, not collision.
   Also uses `scene_collision_cfg.activation_distance: 0.0` which
   allows arm to brush cuboids at zero margin.

2. `robot/spheres/franka_tidyverse_mesh.yml`: the `base_link_z`
   collision-sphere set has a redundant top layer at z=0.5 with radius
   0.1m, modeling volume the actual Tidybot base doesn't occupy
   (base body ends at z=0.472). This is cosmetic — physics still
   works without it — but the smaller radius reduces false-positive
   safety filtering near cabinet doors.

Both fixes ship in `curobo_service/assets/` and get copied into the
cuRobo install at startup. Originals are backed up to `.curobo_service_orig`.

Tested against cuRobo 0.8.0 (commit … irrelevant to this version pin).
If a different cuRobo version is detected we WARN but still apply —
patches are line/key changes, not hash-keyed.
"""
from __future__ import annotations

import filecmp
import shutil
import sys
from pathlib import Path

# Map of (relative path inside cuRobo content/configs/) → (asset filename)
# Both halves resolved at runtime so we work regardless of where curobo is
# installed.
PATCH_FILES = {
    "task/metrics_base.yml": "task/metrics_base.yml",
    "robot/spheres/franka_tidyverse_mesh.yml": "spheres/franka_tidyverse_mesh.yml",
}

# cuRobo version prefixes we've tested these patches against. A detected
# version that doesn't startswith any of these triggers a WARN (we still
# apply — patches are content rewrites, not hash-keyed). The 0.8 prefix
# matches "0.8.0", "0.8.0.post1.dev0+dirty", "0.8.1", etc.
SUPPORTED_VERSION_PREFIXES = ("0.8",)


def _curobo_configs_root() -> Path | None:
    """Locate cuRobo's content/configs/ directory by importing it."""
    try:
        import curobo
    except ImportError:
        return None
    pkg_dir = Path(curobo.__file__).resolve().parent
    candidates = [
        pkg_dir / "content" / "configs",
        pkg_dir.parent / "src" / "curobo" / "content" / "configs",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return None


def _assets_root() -> Path:
    return Path(__file__).resolve().parent / "assets"


def _curobo_version() -> str:
    try:
        import curobo
        return getattr(curobo, "__version__", "unknown")
    except ImportError:
        return "missing"


def apply(verbose: bool = True) -> dict:
    """Apply all known patches to the active cuRobo install.

    Returns a summary dict suitable for logging.
    """
    summary: dict = {
        "curobo_version": _curobo_version(),
        "patches_applied": [],
        "patches_skipped_idempotent": [],
        "errors": [],
    }

    v = summary["curobo_version"]
    if not any(v.startswith(p) for p in SUPPORTED_VERSION_PREFIXES):
        summary["errors"].append(
            f"curobo {v} doesn't start with any tested prefix "
            f"{SUPPORTED_VERSION_PREFIXES}; applying anyway")
        if verbose:
            print(f"[apply_patches] WARN {summary['errors'][-1]}")

    configs_root = _curobo_configs_root()
    if configs_root is None:
        summary["errors"].append("could not locate cuRobo install")
        if verbose:
            print(f"[apply_patches] ERROR: {summary['errors'][-1]}")
        return summary

    assets_root = _assets_root()

    for rel_in_curobo, rel_in_assets in PATCH_FILES.items():
        src = assets_root / rel_in_assets
        dst = configs_root / rel_in_curobo
        if not src.exists():
            summary["errors"].append(f"asset missing: {src}")
            continue
        if not dst.parent.exists():
            summary["errors"].append(f"cuRobo dest missing: {dst.parent}")
            continue

        # Idempotent: skip if already byte-identical
        if dst.exists() and filecmp.cmp(str(src), str(dst), shallow=False):
            summary["patches_skipped_idempotent"].append(rel_in_curobo)
            continue

        # Backup pristine original on first apply (don't overwrite an
        # existing backup — preserve the very first version we saw).
        backup = dst.with_suffix(dst.suffix + ".curobo_service_orig")
        if dst.exists() and not backup.exists():
            shutil.copy2(dst, backup)

        shutil.copy2(src, dst)
        summary["patches_applied"].append(rel_in_curobo)
        if verbose:
            print(f"[apply_patches] {rel_in_curobo}: applied (backup at "
                  f"{backup.name if backup.exists() else 'n/a'})")

    if verbose:
        if summary["patches_applied"]:
            print(f"[apply_patches] applied {len(summary['patches_applied'])} patch(es) "
                  f"to cuRobo {summary['curobo_version']}")
        elif summary["patches_skipped_idempotent"]:
            print(f"[apply_patches] all {len(summary['patches_skipped_idempotent'])} "
                  f"patch(es) already up to date")

    return summary


def revert(verbose: bool = True) -> dict:
    """Restore pristine cuRobo configs from .curobo_service_orig backups."""
    summary: dict = {"reverted": [], "no_backup": [], "errors": []}
    configs_root = _curobo_configs_root()
    if configs_root is None:
        summary["errors"].append("could not locate cuRobo install")
        return summary

    for rel_in_curobo in PATCH_FILES:
        dst = configs_root / rel_in_curobo
        backup = dst.with_suffix(dst.suffix + ".curobo_service_orig")
        if backup.exists():
            shutil.copy2(backup, dst)
            summary["reverted"].append(rel_in_curobo)
            if verbose:
                print(f"[apply_patches] reverted {rel_in_curobo}")
        else:
            summary["no_backup"].append(rel_in_curobo)
            if verbose:
                print(f"[apply_patches] no backup for {rel_in_curobo}, skipped")
    return summary


if __name__ == "__main__":
    if "--revert" in sys.argv:
        out = revert()
    else:
        out = apply()
    if out.get("errors"):
        sys.exit(1)
