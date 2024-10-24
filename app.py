import gradio as gr
from gradio import ChatMessage
from utils import stream_from_transformers_agent
from gradio.context import Context
from gradio import Request
import pickle
import os
from dotenv import load_dotenv
from agent import get_agent, DEFAULT_TASK_SOLVING_TOOLBOX
from transformers.agents import (
    DuckDuckGoSearchTool,
    ImageQuestionAnsweringTool,
    VisitWebpageTool,
)
from tools.text_to_image import TextToImageTool
from PIL import Image
from transformers import load_tool
from prompts import (
    DEFAULT_SQUAD_REACT_CODE_SYSTEM_PROMPT,
    FOCUSED_SQUAD_REACT_CODE_SYSTEM_PROMPT,
)
from pygments.formatters import HtmlFormatter


load_dotenv()

SESSION_PERSISTENCE_ENABLED = os.getenv("SESSION_PERSISTENCE_ENABLED", False)

sessions_path = "sessions.pkl"
sessions = (
    pickle.load(open(sessions_path, "rb"))
    if SESSION_PERSISTENCE_ENABLED and os.path.exists(sessions_path)
    else {}
)

# If currently hosted on HuggingFace Spaces, use the default model, otherwise use the local model
model_name = (
    "meta-llama/Meta-Llama-3.1-8B-Instruct"
    if os.getenv("SPACE_ID") is not None
    else "http://localhost:1234/v1"
)

class FixImageQuestionAnsweringTool(ImageQuestionAnsweringTool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, image: "Image | str", question: str):
        if isinstance(image, str):
            image = Image.open(image)
        return super().encode(image, question)

ADDITIONAL_TOOLS = [
    DuckDuckGoSearchTool(),
    VisitWebpageTool(),
    FixImageQuestionAnsweringTool(),
    load_tool("speech_to_text"),
    load_tool("text_to_speech"),
    load_tool("translation"),
    TextToImageTool(),
]

# Add image tools to the default task solving toolbox, for a more visually interactive experience
TASK_SOLVING_TOOLBOX = DEFAULT_TASK_SOLVING_TOOLBOX + ADDITIONAL_TOOLS

# system_prompt = DEFAULT_SQUAD_REACT_CODE_SYSTEM_PROMPT
system_prompt = FOCUSED_SQUAD_REACT_CODE_SYSTEM_PROMPT

agent = get_agent(
    model_name=model_name,
    toolbox=TASK_SOLVING_TOOLBOX,
    system_prompt=system_prompt,
    use_openai=True,  # Use OpenAI instead of a local or HF model as the base LLM engine
)

app = None


def append_example_message(x: gr.SelectData, messages):
    if x.value["text"] is not None:
        message = x.value["text"]
    if "files" in x.value:
        if isinstance(x.value["files"], list):
            message = "Here are the files: "
            for file in x.value["files"]:
                message += f"{file}, "
        else:
            message = x.value["files"]
    messages.append(ChatMessage(role="user", content=message))
    return messages


def add_message(message, messages):
    messages.append(ChatMessage(role="user", content=message))
    return messages


def interact_with_agent(messages, request: Request):
    session_hash = request.session_hash
    prompt = messages[-1]["content"]
    agent.logs = sessions.get(session_hash + "_logs", [])
    yield messages, gr.update(
        value="<center><h1>Thinking...</h1></center>", visible=True
    )
    for msg in stream_from_transformers_agent(agent, prompt):
        if isinstance(msg, ChatMessage):
            messages.append(msg)
            yield messages, gr.update(visible=True)
        else:
            yield messages, gr.update(
                value=f"<center><h1>{msg}</h1></center>", visible=True
            )
    yield messages, gr.update(value="<center><h1>Idle</h1></center>", visible=False)


def persist(component):

    def resume_session(value, request: Request):
        session_hash = request.session_hash
        print(f"Resuming session for {session_hash}")
        state = sessions.get(session_hash, value)
        agent.logs = sessions.get(session_hash + "_logs", [])
        return state

    def update_session(value, request: Request):
        session_hash = request.session_hash
        print(f"Updating persisted session state for {session_hash}")
        sessions[session_hash] = value
        sessions[session_hash + "_logs"] = agent.logs
        if SESSION_PERSISTENCE_ENABLED:
            pickle.dump(sessions, open(sessions_path, "wb"))

    Context.root_block.load(resume_session, inputs=[component], outputs=component)
    component.change(update_session, inputs=[component], outputs=None)

    return component


from gradio.components import (
    Component as GradioComponent,
)
from gradio.components.chatbot import (
    Chatbot,
    FileDataDict,
    FileData,
    ComponentMessage,
    FileMessage,
)


class CleanChatBot(Chatbot):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _postprocess_content(
        self,
        chat_message: (
            str | tuple | list | FileDataDict | FileData | GradioComponent | None
        ),
    ) -> str | FileMessage | ComponentMessage | None:
        response = super()._postprocess_content(chat_message)
        print(f"Post processing content: {response}")
        if isinstance(response, ComponentMessage):
            print(f"Setting open to False for {response}")
            response.props["open"] = False
        return response


with gr.Blocks(
    fill_height=True,
    css=".gradio-container .message .content {text-align: left;}"
    + HtmlFormatter().get_style_defs(".highlight"),
) as demo:
    state = gr.State()
    inner_monologue_component = gr.Markdown(
        """<h2>Inner Monologue</h2>""", visible=False
    )
    chatbot = persist(
        gr.Chatbot(
            value=[],
            label="SQuAD Agent",
            type="messages",
            avatar_images=(
                None,
                "SQuAD.png",
            ),
            scale=1,
            autoscroll=True,
            show_copy_all_button=True,
            show_copy_button=True,
            placeholder="""<h1>SQuAD Agent</h1>
            <h2>I am your friendly guide to the Stanford Question and Answer Dataset (SQuAD).</h2>
            <h2>You can ask me questions about the dataset. You can also ask me to create images 
            to help illustrate the topics under discussion, or expand the discussion beyond the dataset.</h2>
        """,
            examples=[
                {
                    "text": "What is on top of the Notre Dame building?",
                },
                {
                    "text": "Tell me what's on top of the Notre Dame building, and draw a picture of it.",
                },
                {
                    "text": "Draw a picture of whatever is on top of the Notre Dame building.",
                },
            ],
        )
    )
    text_input = gr.Textbox(lines=1, label="Chat Message", scale=0)
    chat_msg = text_input.submit(add_message, [text_input, chatbot], [chatbot])
    bot_msg = chat_msg.then(
        interact_with_agent, [chatbot], [chatbot, inner_monologue_component]
    )
    text_input.submit(lambda: "", None, text_input)
    chatbot.example_select(append_example_message, [chatbot], [chatbot]).then(
        interact_with_agent, [chatbot], [chatbot, inner_monologue_component]
    )

if __name__ == "__main__":
    demo.launch()
