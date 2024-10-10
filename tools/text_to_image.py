from transformers.agents.tools import Tool
from huggingface_hub import InferenceClient

class TextToImageTool(Tool):
    description = "This is a tool that creates an image according to a prompt, which is a text description."
    name = "image_generator"
    inputs = {"prompt": {"type": "string", "description": "The image generator prompt. Don't hesitate to add details in the prompt to make the image look better, like 'high-res, photorealistic', etc."}}
    output_type = "image"
    model_sdxl = "stabilityai/stable-diffusion-xl-base-1.0"
    client = InferenceClient(model_sdxl)

    def forward(self, prompt):
        return self.client.text_to_image(prompt)