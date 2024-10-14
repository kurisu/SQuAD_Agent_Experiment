from transformers import ReactCodeAgent, HfApiEngine
from prompts import *
from tools.squad_tools import SquadRetrieverTool, SquadQueryTool

DEFAULT_TASK_SOLVING_TOOLBOX = [SquadRetrieverTool(), SquadQueryTool()]

def get_agent(
    model_name=None,
    system_prompt=DEFAULT_SQUAD_REACT_CODE_SYSTEM_PROMPT,
    toolbox=DEFAULT_TASK_SOLVING_TOOLBOX,
):
    DEFAULT_MODEL_NAME = "http://localhost:1234/v1"
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME

    llm_engine = HfApiEngine(model_name)

    # Initialize the agent with both tools
    agent = ReactCodeAgent(
        tools=toolbox,
        llm_engine=llm_engine,
        system_prompt=system_prompt,
    )

    return agent
