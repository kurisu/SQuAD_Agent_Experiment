# Import all prompts as both constants and in a PROMPTS dictionary,
# from all files in the prompts directory that aren't __init__.py 

import os

def load_constants(constants_dir):
    """Loads constants from .py files in the specified directory."""

    constants = {}

    for filename in os.listdir(constants_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = filename[:-3]  # Remove .py extension
            module = __import__(f"{constants_dir}.{module_name}", fromlist=[module_name])

            for name, value in vars(module).items():
                if name.isupper():  # Convention for constants
                    constants[name] = value

    return constants

PROMPTS = load_constants("prompts")

# Import all prompts locally as well, for code completion
from prompts.default import DEFAULT_SQUAD_REACT_CODE_SYSTEM_PROMPT
