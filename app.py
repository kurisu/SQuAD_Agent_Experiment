import gradio as gr
from gradio import ChatMessage
from transformers import load_tool, ReactCodeAgent, HfApiEngine
from utils import stream_from_transformers_agent
import os
# Import tool from Hub
#image_generation_tool = load_tool("m-ric/text-to-image")
from transformers.agents.tools import Tool
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()

from data import Data

class SquadRetrieverTool(Tool):
    name = "squad_retriever"
    description = "Answers questions from the Stanford Question Answering Dataset (SQuAD)."
    inputs = {
        "query": {
            "type": "string",
            "description": "The question. This should be the literal question being asked, only modified to be informed by chat history.",
        },
    }
    output_type = "string"
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = Data()
        self.data.load_data()
        self.query_engine = self.data.index.as_query_engine()

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        response = self.query_engine.query(query)
        # docs = self.data.index.similarity_search(query, k=3)

        if len(response.response) == 0:
            return "No answer found for this query."
        return "Retrieved answer:\n\n" + "\n===Answer===\n".join(
            [response.response]
        )

class TextToImageTool(Tool):
    description = "This is a tool that creates an image according to a prompt, which is a text description."
    name = "image_generator"
    inputs = {"prompt": {"type": "string", "description": "The image generator prompt. Don't hesitate to add details in the prompt to make the image look better, like 'high-res, photorealistic', etc."}}
    output_type = "image"
    model_sdxl = "stabilityai/stable-diffusion-xl-base-1.0"
    client = InferenceClient(model_sdxl)

    def forward(self, prompt):
        return self.client.text_to_image(prompt)


# image_generation_tool = TextToImageTool()
# squad_retriever_tool = SquadRetrieverTool()

TASK_SOLVING_TOOLBOX = [
    SquadRetrieverTool(),
    TextToImageTool(),
    # SearchTool(),
    # VisualQATool(),
    # SpeechToTextTool(),
    # TextInspectorTool(),
]

#llm_engine = HfApiEngine("meta-llama/Meta-Llama-3.1-8B-Instruct")
llm_engine = HfApiEngine(model="http://localhost:1234/v1")
# Initialize the agent with both tools
agent = ReactCodeAgent(tools=TASK_SOLVING_TOOLBOX, llm_engine=llm_engine)
# print(agent.toolbox().tools)

def interact_with_agent(prompt, messages):
    messages.append(ChatMessage(role="user", content=prompt))
    yield messages
    for msg in stream_from_transformers_agent(agent, prompt):
        messages.append(msg)
        yield messages
    yield messages


with gr.Blocks(fill_height=True) as demo:
    chatbot = gr.Chatbot(
        label="Agent",
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
            }
        ]
    )
    text_input = gr.Textbox(lines=1, label="Chat Message", scale=0)
    text_input.submit(interact_with_agent, [text_input, chatbot], [chatbot])
    text_input.submit(lambda x: "", None, text_input)

if __name__ == "__main__":
    demo.launch()
