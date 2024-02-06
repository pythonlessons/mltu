import re
import yaml

def yaml_load(file: str) -> dict:
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File path.

    Returns:
        (dict): YAML data.
    """
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        # Add YAML filename to dict and return
        data = yaml.safe_load(s)  # always return a dict (yaml.safe_load() may return None for empty files)
        if data:
            return data
        else:
            raise ValueError(f"YAML file '{file}' is empty")
        
def guess_model_scale(model_path):
    """
    Guess model scale from model path.

    Args:
        model_path (str): Model path.

    Returns:
        (int): Model scale.
    """
    if model_path[6] in ["n", "s", "m", "l", "x"]:
        return model_path[6]
    else:
        raise ValueError(f"Model scale not found in '{model_path}'")