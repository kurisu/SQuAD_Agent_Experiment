from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from agent import get_agent
from prompts import FOCUSED_SQUAD_REACT_CODE_SYSTEM_PROMPT
import logging

"""
Test the chatbot's ability to carry out multi-turn conversations, 
adapt to context, and handle a variety of topics.
"""
def test_chatbot_goals():
    user_messages = [
        "What is on top of the Notre Dame building?",
        "When did the United States purchase Alaska from Russia?",
        "What year did Bern join the Swiss Confederacy?",
        "Are there any other statues nearby the first one you mentioned?",
    ]
    minimum_acceptable_answers = [
        "golden statue of the Virgin Mary",
        "1867",
        "1353",
        "copper statue of Christ"
    ]
    agent = get_agent(system_prompt=FOCUSED_SQUAD_REACT_CODE_SYSTEM_PROMPT)
    agent.logger.setLevel(logging.CRITICAL)
    for i, (user_message, minimum_acceptable_answer) in enumerate(zip(user_messages, minimum_acceptable_answers)):
        answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
        reset = (i == 0) # Reset the agent for the first message only
        print(f"Running with reset={reset}")
        answer = agent.run(user_message, stream=False, reset=reset)
        print(f"User message: {user_message}")
        print(f"Minimum acceptable answer: {minimum_acceptable_answer}")
        print(f"Answer: {answer}")
        test_case = LLMTestCase(
            input=user_message,
            actual_output=answer,
        )
        assert_test(test_case, [answer_relevancy_metric])
