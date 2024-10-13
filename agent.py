from transformers import ReactCodeAgent, HfApiEngine
from prompts import SQUAD_REACT_CODE_SYSTEM_PROMPT
from tools.squad_tools import SquadRetrieverTool, SquadQueryTool
from tools.text_to_image import TextToImageTool

def get_agent():
    # model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    model_name = "http://localhost:1234/v1"

    llm_engine = HfApiEngine(model_name)

    TASK_SOLVING_TOOLBOX = [
        SquadRetrieverTool(),
        SquadQueryTool(),
        TextToImageTool(),
    ]

    # Initialize the agent with both tools
    agent = ReactCodeAgent(
        tools=TASK_SOLVING_TOOLBOX,
        llm_engine=llm_engine,
        system_prompt=SQUAD_REACT_CODE_SYSTEM_PROMPT,
    )

    return agent

