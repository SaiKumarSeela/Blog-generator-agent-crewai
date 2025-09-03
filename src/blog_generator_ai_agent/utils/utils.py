import yaml
import uuid
from datetime import datetime
from typing import Any, Dict


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


