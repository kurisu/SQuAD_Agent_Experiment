from __future__ import annotations

import gradio as gr
from gradio import ChatMessage
from transformers.agents import ReactCodeAgent, agent_types
from typing import Generator
from termcolor import colored
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
from pygments.formatters import TerminalFormatter

def highlight_code_terminal(text):
    return highlight(text, PythonLexer(), TerminalFormatter())

def highlight_code_html(code):
    return highlight(code, PythonLexer(), HtmlFormatter())


def pull_message(step_log: dict):
    if step_log.get("rationale"):
        yield "🧠 Thinking...", f"{step_log["rationale"]}"
    if step_log.get("tool_call"):
        used_code = step_log["tool_call"]["tool_name"] == "code interpreter"
        content = step_log["tool_call"]["tool_arguments"]
        yield f"🛠️ Using tool {step_log['tool_call']['tool_name']}...", content
    if step_log.get("observation"):
        yield "👀 Observing...", step_log['observation']
    if step_log.get("error"):
        yield "💥 Coping with an Error...", step_log['error'].message

def stream_from_transformers_agent(
    agent: ReactCodeAgent, prompt: str
) -> Generator[ChatMessage, None, ChatMessage | None]:
    """Runs an agent with the given prompt and streams the messages from the agent as ChatMessages."""

    class Output:
        output: agent_types.AgentType | str = None

    inner_monologue = ChatMessage(
        role="assistant", 
        metadata={"title": "🧠 Thinking..."},
        content="",
    )

    step_log = None
    for step_log in agent.run(prompt, stream=True, reset=len(agent.logs) == 0): # Reset=False misbehaves if the agent has not yet been run
        if isinstance(step_log, dict):
            for title, message in pull_message(step_log):
                terminal_message = message
                if ("Using tool" in title) or ("Error" in title):
                    terminal_message = highlight_code_terminal(message)
                    message = highlight_code_html(message)
                if "Observing" in title:
                    message = f"<div style='border:1px solid black; background-color: var(--code-background-fill); padding: 10px;'>{message.replace("\n", "<br/>")}</div>"
                print(colored("=== Inner Monologue Message:\n", "blue", attrs=["bold"]), f"{title}\n{terminal_message}")
                inner_monologue.content += f"<h2>{title}</h2><p>{message}</p>"
                yield title

    if inner_monologue is not None:
        inner_monologue.metadata = {"title": "Inner Monologue (click to expand)"}
        yield inner_monologue

    Output.output = step_log
    if isinstance(Output.output, agent_types.AgentText):
        yield ChatMessage(
            role="assistant", content=f"{Output.output.to_string()}\n")  # type: ignore
    elif isinstance(Output.output, agent_types.AgentImage):
        yield ChatMessage(
            role="assistant",
            content={"path": Output.output.to_string(), "mime_type": "image/png"},  # type: ignore
        )
    elif isinstance(Output.output, agent_types.AgentAudio):
        yield ChatMessage(
            role="assistant",
            content={"path": Output.output.to_string(), "mime_type": "audio/wav"},  # type: ignore
        )
    else:
        return ChatMessage(role="assistant", content=Output.output)
