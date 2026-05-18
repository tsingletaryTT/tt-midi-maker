import subprocess
import json


def detect_tt_devices() -> list[int]:
    """Return list of available TT device indices. Empty list if none found."""
    try:
        result = subprocess.run(
            ["tt-smi", "-s"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        data = json.loads(result.stdout)
        return [i for i, d in enumerate(data.get("device_info", []))
                if d.get("status") == "available"]
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
