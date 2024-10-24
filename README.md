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

## Getting Started

1. Install dependencies:

* Requires Python >= 3.11.9

```bash
pip install -r pre-requirements.txt
pip install -r requirements.txt
```

1. Set up required keys:

```bash
HF_TOKEN=<your token>
```

1. Run the app:

```bash
python app.py
```

## Methods Used

1. SQuAD Dataset: The dataset used for training the chatbot is the Stanford SQuAD dataset, which contains over 100,000 questions and answers extracted from 500+ articles.
2. RAG: RAG is a technique used to improve the accuracy of chatbots by using a custom knowledge base. In this project, the Stanford SQuAD dataset is used as the knowledge base.
3. Llama 3.1: Llama 3.1 is a large language model used to generate responses to user questions. It is used in this project to generate responses to user questions, while also incorporating conversational context.
4. Transformers Agents 2.0: Transformers Agents 2.0 is a framework for building conversational AI systems. It is used in this project to build the chatbot.
5. Created a SquadRetrieverTool to integrate a fine-tuned BERT model into the agent, along with a TextToImageTool for a playful way to engage with the question-answering agent.

## Evaluation

* [Agent Reasoning Benchmark](https://github.com/aymeric-roucher/agent_reasoning_benchmark)
* [Hugging Face Blog: Open Source LLMs as Agents](https://huggingface.co/blog/open-source-llms-as-agents)
* [Benchmarking Transformers Agents](https://github.com/aymeric-roucher/agent_reasoning_benchmark/blob/main/benchmark_transformers_agents.ipynb)

## Results

TBD

## Limitations

TBD

## Related Research

* [Retro: A Generalist Agent for Science](https://arxiv.org/abs/2112.04426)
* [RETRO-pytorch](https://github.com/lucidrains/RETRO-pytorch)
* [Why isn't Retro mainstream? State-of-the-art within reach](https://www.reddit.com/r/MachineLearning/comments/1cffgkt/d_why_isnt_retro_mainstream_stateoftheart_within/)

TBD

## Acknowledgments

* [Agents 2.0](https://github.com/huggingface/transformers/tree/main/src/transformers/agents)
* [SemScore: Automated Evaluation of Instruction-Tuned LLMs based on Semantic Textual Similarity](https://arxiv.org/abs/2401.17072)
* [SemScore](https://huggingface.co/blog/g-ronimo/semscore)
* [Stanford SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
* [llama 3.1](https://github.com/meta-llama/Meta-Llama)
* [Gradio](https://www.gradio.app/)
