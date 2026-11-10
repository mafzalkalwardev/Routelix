"""One Google Voice account, several Chrome profiles.

Each profile is one live call. The account is not duplicated. Line 1 reuses
chrome_profiles/tony when that session already exists. Extra profiles are
tony_2, tony_3, and so on, cloned from tony the first time a run prepares
them. Displays stay on :98 (laneora owns :99). Separate displays, if turned
on, start at :100. VNC ports start at 5911/6091 so they miss 5901, 5902,
6080, and 6081.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from src.capacity import configured_line_count, sustainable_lines
from src.paths import runtime_base

MAX_LINES = 7
FORBIDDEN_DISPLAYS = {":99"}
VNC_PORT_BASE = 5910
NOVNC_PORT_BASE = 6090
RESERVED_VNC = {5901, 5902, 6080, 6081}


def line_count() -> int:
    return configured_line_count()


def set_line_count(count: int) -> int:
    count = max(1, min(MAX_LINES, int(count)))
    path = runtime_base() / "data" / "lines.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"lines": count}, indent=2), encoding="utf-8")
    os.environ["ROUTELIX_LINES"] = str(count)
    return count


def capacity_warning(count: int | None = None) -> str:
    count = line_count() if count is None else max(1, min(MAX_LINES, int(count)))
    limit = sustainable_lines()
    if count > limit:
        return (
            f"{count} lines is above the measured capacity of {limit} on this host. "
            "Calls can still be armed, and audio will contend for the CPUs."
        )
    return ""


def line_display(index: int) -> str:
    separate = os.getenv("ROUTELIX_SEPARATE_DISPLAYS", "").strip().lower() in {"1", "true", "yes", "on"}
    if not separate:
        base = os.getenv("ROUTELIX_DISPLAY", "").strip() or os.getenv("DISPLAY", "").strip() or ":98"
        if base in FORBIDDEN_DISPLAYS:
            base = ":98"
        return base
    number = 99 + max(1, index)
    if number == 99:
        number = 100
    display = f":{number}"
    if display in FORBIDDEN_DISPLAYS:
        display = ":100"
    return display


def line_ports(index: int) -> tuple[int, int]:
    vnc = VNC_PORT_BASE + index
    novnc = NOVNC_PORT_BASE + index
    if vnc in RESERVED_VNC or novnc in RESERVED_VNC:
        raise RuntimeError(f"Line {index} picked a reserved VNC port")
    return vnc, novnc


def profile_name(index: int) -> str:
    if index < 1:
        raise ValueError("lines are 1-based")
    return "tony" if index == 1 else f"tony_{index}"


def _profiles_root() -> Path:
    return runtime_base() / "chrome_profiles"


def session_ready(name: str) -> bool:
    return (_profiles_root() / name / ".routelix_session").is_file()


def ensure_profile(index: int) -> str:
    """Reuse tony. Clone it into tony_N when that folder is missing."""
    name = profile_name(index)
    root = _profiles_root()
    dest = root / name
    if dest.exists():
        return name
    source = root / "tony"
    if index > 1 and source.is_dir():
        shutil.copytree(
            source,
            dest,
            ignore=shutil.ignore_patterns("SingletonLock", "SingletonCookie", "SingletonSocket"),
        )
    return name


def line_plan(count: int | None = None) -> list[dict]:
    count = line_count() if count is None else max(1, min(MAX_LINES, int(count)))
    rows = []
    for index in range(1, count + 1):
        vnc, novnc = line_ports(index)
        name = profile_name(index)
        signed = session_ready(name) or (index == 1 and session_ready("tony"))
        rows.append({
            "index": index,
            "profile_name": name,
            "gv_account": "Main line",
            "display": line_display(index),
            "window_position": f"{60 * (index - 1)},{40 * (index - 1)}",
            "vnc_port": vnc,
            "novnc_port": novnc,
            "loopback_device": f"routelix_to_gv_{index}",
            "capture_device": f"routelix_from_gv_{index}.monitor",
            "pulse_sink": f"routelix_from_gv_{index}",
            "pulse_source": f"routelix_to_gv_{index}.monitor",
            "signed_in": signed,
            "status": "idle" if signed else "needs_signin",
            "warning": capacity_warning(count),
        })
    return rows


def parallel_profiles(count: int | None = None) -> list[dict]:
    return [
        {
            "profile_name": row["profile_name"],
            "loopback_device": row["loopback_device"],
            "capture_device": row["capture_device"],
            "pulse_sink": row["pulse_sink"],
            "pulse_source": row["pulse_source"],
        }
        for row in line_plan(count)
    ]
