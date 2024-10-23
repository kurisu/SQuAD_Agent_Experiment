import pytest
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
import pandas as pd
import os
from agent import get_agent
from semscore import EmbeddingModelWrapper
import logging
from tqdm import tqdm
from transformers.agents import agent_types

def test_case():
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output from your LLM application
        actual_output="We offer a 30-day full refund at no extra costs.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."]
    )
    assert_test(test_case, [answer_relevancy_metric])


def test_default_agent():
    SAMPLES_DIR = "samples"
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    dfSample = pd.read_pickle(os.path.join(SAMPLES_DIR, f"samples.pkl"))
    agent = get_agent()
    # Suppress logging from the agent, which can be quite verbose
    agent.logger.setLevel(logging.CRITICAL)
    answers_ref = []
    answers_pred = []
    for title, context, question, answer, synthesized_question in tqdm(dfSample.values):
        class Output:
            output: agent_types.AgentType | str = None

        prompt = synthesized_question
        answers_ref.append(answer)
        final_answer = agent.run(prompt, stream=False, reset=True)
        answers_pred.append(final_answer)

    answers_ref = [str(answer) for answer in answers_ref]
    answers_pred = [str(answer) for answer in answers_pred]

    em = EmbeddingModelWrapper()
    similarities = em.get_similarities(
        em.get_embeddings( answers_pred ),
        em.get_embeddings( answers_ref ),
    )
    mean_similarity = similarities.mean()

    assert(mean_similarity >= 0.5, f"Mean similarity is too low: {mean_similarity}")
        