# memgpt_rag_experiment

## Overview

The project is built using MemGPT, a framework for building conversational AI systems, and uses the Stanford SQuAD dataset for training. The chatbot is designed to answer questions about the dataset, while also incorporating conversational context to provide a more natural and engaging conversation experience.

## Getting Started

```bash
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

## Methods Used

1. SQuAD Dataset: The dataset used for training the chatbot is the Stanford SQuAD dataset, which contains over 100,000 questions and answers extracted from 500+ articles.
2. MemGPT: MemGPT is a framework for building conversational AI systems. It uses a combination of retrieval and generation to create a chatbot that can answer questions about the dataset, while also incorporating conversational context.
3. RAG: RAG is a technique used to improve the accuracy of chatbots by using a custom knowledge base. In this project, the Stanford SQuAD dataset is used as the knowledge base.
4. Llama 3.1: Llama 3.1 is a large language model used to generate responses to user questions. It is used in this project to generate responses to user questions, while also incorporating conversational context.
5. Transformers Agents 2.0: Transformers Agents 2.0 is a framework for building conversational AI systems. It is used in this project to build the chatbot.
6. Created a SquadRetrieverTool to integrate a fine-tuned BERT model into the agent, along with a TextToImageTool for a playful way to engage with the question-answering agent.

## Results

TBD

## Limitations

TBD

## Future Work

TBD

## Acknowledgments

* [MemGPT](https://github.com/cpacker/MemGPT)
* [Stanford SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)
* [GPT-4](https://openai.com/gpt-4/)
