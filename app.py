import gradio as gr
from gradio import ChatMessage
from transformers import ReactCodeAgent, HfApiEngine
from utils import stream_from_transformers_agent
from prompts import SQUAD_REACT_CODE_SYSTEM_PROMPT
from tools.squad_retriever import SquadRetrieverTool
from tools.text_to_image import TextToImageTool
from dotenv import load_dotenv

load_dotenv()

TASK_SOLVING_TOOLBOX = [
    SquadRetrieverTool(),
    TextToImageTool(),
]

model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_name = "http://localhost:1234/v1"

llm_engine = HfApiEngine(model_name)

# Initialize the agent with both tools
agent = ReactCodeAgent(
    tools=TASK_SOLVING_TOOLBOX,
    llm_engine=llm_engine,
    system_prompt=SQUAD_REACT_CODE_SYSTEM_PROMPT,
)

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

def interact_with_agent(messages):
    prompt = messages[-1]['content']
    for msg in stream_from_transformers_agent(agent, prompt):
        messages.append(msg)
        yield messages
    yield messages

with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.Chatbot(
        label="SQuAD Agent",
        type="messages",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png",
        ),
        scale=1,
        bubble_full_width=False,
        autoscroll=True,
        show_copy_all_button=True,
        show_copy_button=True,
        placeholder="Enter a message",
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
    text_input = gr.Textbox(lines=1, label="Chat Message", scale=0)
    chat_msg = text_input.submit(add_message, [text_input, chatbot], [chatbot])
    bot_msg = chat_msg.then(interact_with_agent, [chatbot], [chatbot])
    text_input.submit(lambda: "", None, text_input)
    chatbot.example_select(append_example_message, [chatbot], [chatbot]).then(
        interact_with_agent, [chatbot], [chatbot]
    )

if __name__ == "__main__":
    demo.launch()
