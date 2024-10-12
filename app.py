import gradio as gr
from gradio import ChatMessage
from transformers import ReactCodeAgent, HfApiEngine
from utils import stream_from_transformers_agent
from prompts import SQUAD_REACT_CODE_SYSTEM_PROMPT
from tools.squad_tools import SquadRetrieverTool, SquadQueryTool
from tools.text_to_image import TextToImageTool
from gradio.context import Context
from gradio import Request
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

TASK_SOLVING_TOOLBOX = [
    SquadRetrieverTool(),
    SquadQueryTool(),
    TextToImageTool(),
]

sessions_path = "sessions.pkl"
sessions = pickle.load(open(sessions_path, "rb")) if os.path.exists(sessions_path) else {}

app = None

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

def interact_with_agent(messages, request: Request):
    username = request.username
    prompt = messages[-1]['content']
    agent.logs = sessions.get(username + "_logs", [])
    for msg in stream_from_transformers_agent(agent, prompt):
        messages.append(msg)
        yield messages
    yield messages

def persist(component):

    def resume_session(value, request: Request):
        username = request.username
        print(f"Resuming session for {username}")
        state = sessions.get(username, value)
        agent.logs = sessions.get(username + "_logs", [])
        return state

    def update_session(value, request: Request):
        username = request.username
        print(f"Updating persisted session state for {username}")
        sessions[username] = value
        sessions[username + "_logs"] = agent.logs
        pickle.dump(sessions, open(sessions_path, "wb"))
        return

    Context.root_block.load(resume_session, inputs=[component], outputs=component)
    component.change(update_session, inputs=[component], outputs=[])

    return component

def welcome_message(request: Request):
    return f"<h2>Welcome, {request.username}</h2>"

with gr.Blocks(fill_height=True) as demo:
    # put the welcome message and logout button in a row
    with gr.Row() as row:
        welcome_msg = gr.Markdown(f"Welcome")
        logout_button = gr.Button("Logout", link="/logout")
        demo.load(welcome_message, None, welcome_msg)

    chatbot = persist(gr.Chatbot(
        value=[],
        label="SQuAD Agent",
        type="messages",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/53/robot-face_1f916.png",
        ),
        scale=1,
        autoscroll=True,
        show_copy_all_button=True,
        show_copy_button=True,
        placeholder="""<h1>SQuAD Agent</h1>
            <h2>I am your friendly guide to the Stanford Question and Answer Dataset (SQuAD).</h2>
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
    ))
    text_input = gr.Textbox(lines=1, label="Chat Message", scale=0)
    chat_msg = text_input.submit(add_message, [text_input, chatbot], [chatbot])
    bot_msg = chat_msg.then(interact_with_agent, [chatbot], [chatbot])
    text_input.submit(lambda: "", None, text_input)
    chatbot.example_select(append_example_message, [chatbot], [chatbot]).then(
        interact_with_agent, [chatbot], [chatbot]
    )

def honor_system_auth(username, password):
    return password == "happy"

if __name__ == "__main__":
    demo.launch(
        auth=honor_system_auth,
        auth_message="""<h3>Honor System Authentication:</h3>
            <p>Log in with <strong>any username</strong> and the password <strong>"happy"</strong>.</p>
            <br/><br/>
            <i>Note:Your chat history will be saved to your username, but take care because others 
            may log in with the same username and see your chat history.</i>
        """,
    )
