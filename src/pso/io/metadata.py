import subprocess
import platform
import os

def get_git_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"

def get_hardware_info() -> dict:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
    }
