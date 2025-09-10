import yaml
import uuid
from datetime import datetime
from typing import Any, Dict
from pathlib import Path
import json
import os
sessions: Dict[str, Dict] = {}
results_storage: Dict[str, Dict] = {}

def load_yaml_config(file_path: str) -> dict:
    """Load YAML configuration file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML config from {file_path}: {e}")
        return {}
    
# Related to Fastapi 

def create_session() -> str:
    """Create a new session ID"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "created_at": datetime.now().isoformat(),
        "status": "active",
        "steps_completed": [],
    }
    return session_id


def get_session(session_id: str) -> Dict:
    """Get session data"""
    if session_id not in sessions:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Session not found")
    return sessions[session_id]


def update_session(session_id: str, step: str, data: Any):
    """Update session with step completion"""
    session = get_session(session_id)
    session["steps_completed"].append(step)
    session["last_updated"] = datetime.now().isoformat()
    results_storage[f"{session_id}_{step}"] = data


def convert_pydantic_to_dict(data):
    """Convert Pydantic model to dictionary for JSON serialization"""
    if hasattr(data, "model_dump"):
        return data.model_dump()
    elif hasattr(data, "dict"):
        return data.dict()
    else:
        return data


OUTPUTS_DIR = Path(__file__).resolve().parents[3] / "Outputs"

def get_session_output_dir(session_id: str) -> Path:
    """Ensure and return the output directory for a session."""
    session_dir = OUTPUTS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def save_json_output(session_id: str, step: str, data: Any) -> Path:
    """Save step output as JSON under Outputs/<session_id>/<step>.json.

    Returns the path to the saved file.
    """
    session_dir = get_session_output_dir(session_id)
    file_path = session_dir / f"{step}.json"

    # Convert Pydantic or other objects to serializable dicts
    serializable = convert_pydantic_to_dict(data)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    return file_path


def load_json_output(session_id: str, step: str) -> Dict:
    """Load step output JSON from Outputs/<session_id>/<step>.json."""
    session_dir = get_session_output_dir(session_id)
    file_path = session_dir / f"{step}.json"
    if not file_path.exists():
        raise FileNotFoundError(
            f"No output found for step '{step}' in session '{session_id}'"
        )
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
