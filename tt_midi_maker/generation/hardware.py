import subprocess
import json


def detect_tt_devices() -> list[int]:
    """Return list of available TT device indices. Empty list if none found.

    A device is considered available when tt-smi reports it in device_info with
    board_info.dram_status == True.  The older snapshot format had a top-level
    'status' field; the current format uses nested board_info instead.
    """
    try:
        result = subprocess.run(
            ["tt-smi", "-s"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
        devices = []
        for i, d in enumerate(data.get("device_info", [])):
            board = d.get("board_info", {})
            # Current tt-smi format: board_info.dram_status is bool True when healthy
            if board.get("dram_status") is True:
                devices.append(i)
            # Legacy format fallback
            elif d.get("status") == "available":
                devices.append(i)
        return devices
    except (FileNotFoundError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return []


def hardware_status() -> dict:
    """Return a status dict for the midi://hardware/status resource."""
    try:
        result = subprocess.run(
            ["tt-smi", "-s"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            devices = data.get("device_info", [])
            return {
                "tt_smi_available": True,
                "device_count": len(devices),
                "devices": devices,
            }
    except (FileNotFoundError, json.JSONDecodeError, subprocess.TimeoutExpired):
        pass
    return {"tt_smi_available": False, "device_count": 0, "devices": []}
