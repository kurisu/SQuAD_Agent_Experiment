---
title: SQuAD_Agent_Experiment
app_file: app.py
sdk: gradio
sdk_version: 5.0.1
python_version: 3.11.9
---

# SQuAD_Agent_Experiment

## Overview

The project is built using Transformers Agents 2.0, and uses the Stanford SQuAD dataset for training. The chatbot is designed to answer questions about the dataset, while also incorporating conversational context and various tools to provide a more natural and engaging conversational experience.

At the time of writing, the project is available on [Hugging Face Spaces](https://huggingface.co/spaces/kaiokendall/SQuAD_Agent_Experiment).

## Getting Started

1. Install dependencies:

* Requires Python >= 3.11.9

```bash
pip install -r pre-requirements.txt
pip install -r requirements.txt
```

2. Set up required keys:

Create a `.env` file and set the following environment variables:

```bash
HF_TOKEN=<your token>
OPENAI_API_KEY=<your key>
```

3. Run the app:

```bash
python app.py
```

## Methods Used

1. SQuAD Dataset: The dataset used for training the chatbot is the Stanford SQuAD dataset, which contains over 100,000 questions and answers extracted from 500+ articles.
2. RAG: RAG is a technique used to improve the accuracy of chatbots by using a custom knowledge base. In this project, the Stanford SQuAD dataset is used as the knowledge base.
3. Transformers Agents 2.0: Transformers Agents 2.0 is a framework for building conversational AI systems. It is used in this project to build the chatbot.
4. Created a SquadRetrieverTool to integrate a fine-tuned BERT model into the agent, along with a TextToImageTool for a playful way to engage with the question-answering agent.
5. Gradio: Gradio is used to create the chatbot interface, in `app.py`.

## Evaluation

SemScore is used in this project to evaluate the chatbot's responses in the notebook `benchmarking.ipynb`.

See [SemScore: Automated Evaluation of Instruction-Tuned LLMs based on Semantic Textual Similarity](https://doi.org/10.48550/arXiv.2401.17072)

In this experiment, the agent is evaluated with 3 different system prompting approaches:

1. The default prompting approach, which is just the default system prompt used in Hugging Face Transformers Agents 2.0, with only an example of using the `squad_retriever` tool added.
2. A succinct prompting approach, which guides the agent to be concise if possible while still answering the question.
3. A focused prompting approach, which reframes the entire chatbots purpose to focus more on the specific task of answering questions about the SQuAD dataset, while still being open to exploring other topics.

## Results



## Limitations

* This experiment is not designed for multiple users. While it has in-session memory, simply refreshing the browser will reset the chat history, which is convenient for experimentation.
* Some of the agent's underlying engines, models, and tools use keys that have usage limits, so the app may not work if those limits have been reached.
  * It is recommended to clone the repo and run the code using your own keys, to avoid running into those limits.

## Acknowledgments

* [Hugging Face Transformers Agents 2.0](https://huggingface.co/docs/transformers/en/main_classes/agent)
* [SemScore: Automated Evaluation of Instruction-Tuned LLMs based on Semantic Textual Similarity](https://arxiv.org/abs/2401.17072)
* `semscore.py` from [geronimi73/semscore](https://github.com/geronimi73/semscore/blob/main/semscore.py)
* [SemScore](https://huggingface.co/blog/g-ronimo/semscore)
* [Stanford SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
* [Gradio](https://www.gradio.app/)
