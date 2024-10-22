from transformers import ReactCodeAgent, HfApiEngine
from prompts import *
from tools.squad_tools import SquadRetrieverTool, SquadQueryTool
from transformers.agents.llm_engine import MessageRole, get_clean_message_list
from openai import OpenAI

DEFAULT_TASK_SOLVING_TOOLBOX = [SquadRetrieverTool()] # , SquadQueryTool()

openai_role_conversions = {
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}

class OpenAIModel:
    def __init__(self, model_name="gpt-4o-mini-2024-07-18"):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def __call__(self, messages, stop_sequences=[]):
        messages = get_clean_message_list(messages, role_conversions=openai_role_conversions)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stop=stop_sequences,
            temperature=0.5
        )
        return response.choices[0].message.content

def get_agent(
    model_name=None,
    system_prompt=DEFAULT_SQUAD_REACT_CODE_SYSTEM_PROMPT,
    toolbox=DEFAULT_TASK_SOLVING_TOOLBOX,
    use_openai=False,
    openai_model_name="gpt-4o-mini-2024-07-18",
):
    DEFAULT_MODEL_NAME = "http://localhost:1234/v1"
    if model_name is None:
        model_name = DEFAULT_MODEL_NAME

    llm_engine = HfApiEngine(model_name) if not use_openai else OpenAIModel(openai_model_name)

    # Initialize the agent with both tools
    agent = ReactCodeAgent(
        tools=toolbox,
        llm_engine=llm_engine,
        system_prompt=system_prompt,
        additional_authorized_imports=["PIL"],
    )

    return agent
