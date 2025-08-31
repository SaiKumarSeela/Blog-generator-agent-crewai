import yaml

def load_yaml_config(file_path: str) -> dict:
    """Load YAML configuration file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML config from {file_path}: {e}")
        return {}