"""
Working with LLMs in LangChain V.1
Multiple providers, configuration, streaming, and cost optimization
"""

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console

load_dotenv()

console = Console()
MODEL_COLORS = ["cyan", "green", "magenta", "yellow", "blue"]

ROLE_STYLE = {
    SystemMessage: ("System", "magenta"),
    HumanMessage: ("Human", "blue"),
}


def print_messages(messages):
    if isinstance(messages, str):
        console.rule(style="blue")
        console.print(f"[bold blue]Human:[/bold blue] {messages}")
        console.print()
        return
    if messages and isinstance(messages[0], SystemMessage):
        console.rule(style="blue")
    for msg in messages:
        label, color = ROLE_STYLE.get(type(msg), ("AI", "green"))
        console.print(f"[bold {color}]{label}:[/bold {color}] {msg.content}")
    console.print()


def print_response(source: str, content: str, color: str = "cyan"):
    console.rule(f"[bold {color}]{source}[/bold {color}]")
    console.print(content)
    console.print()


def make_model(name: str, provider: str = None, streaming: bool = False):
    return init_chat_model(
        model=name,
        model_provider=provider,
        temperature=0.7,
        streaming=streaming,
        max_retries=3,
    )


def demo_init_chat_model():
    question = "What is the capital of France? Answer in one word."
    print_messages(question)
    response = make_model("gpt-4o-mini", streaming=True).invoke(question)
    print_response("gpt-4o-mini", response.content, "cyan")

    if os.getenv("ANTHROPIC_API_KEY"):
        print_messages(question)
        response = make_model("claude-sonnet-4-5-20250929", provider="anthropic", streaming=True).invoke(question)
        print_response("Anthropic", response.content, "green")


def demo_model_comparison():
    prompt = "Explain recursion in one sentence."
    print_messages(prompt)

    model_names = ["gpt-4o-mini", "gpt-4o"]
    if os.getenv("ANTHROPIC_API_KEY"):
        model_names.append("claude-sonnet-4-5-20250929")

    for i, name in enumerate(model_names):
        provider = "anthropic" if "claude" in name else None
        response = make_model(name, provider=provider).invoke(prompt)
        print_response(name, response.content, MODEL_COLORS[i % len(MODEL_COLORS)])


def demo_message():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    messages = [
        SystemMessage(content="You are a pirate. Always answer like a pirate."),
        HumanMessage(content="What's the weather like today?"),
    ]
    print_messages(messages)
    print_response("The Pirate", model.invoke(messages).content, "yellow")

    messages.append(model.invoke(messages))
    messages.append(HumanMessage(content="What about tomorrow?"))
    console.print("[bold]Multi-turn conversation:[/bold]")
    print_messages([messages[-1]])
    print_response("The Pirate (Follow-up)", model.invoke(messages).content, "yellow")


def exercise_multi_model():
    """
    EXERCISE: Create a function that:
    1. Takes a question and a list of model names
    2. Gets responses from all models
    3. Returns a dict of {model_name: response}

    Test with: question="What is AI?", models=["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-5-20250929"]
    """

    def get_responses(question: str, model_names: list[str], system_prompt: str = None) -> dict[str, str]:
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=question))
        print_messages(messages)
        return {name: make_model(name).invoke(messages).content for name in model_names}

    results = get_responses(
        "What is AI?",
        ["gpt-4o-mini", "gpt-4o", "claude-sonnet-4-5-20250929"],
        system_prompt="请用中文回答所有问题。",
    )
    for i, (model, answer) in enumerate(results.items()):
        print_response(model, answer, MODEL_COLORS[i % len(MODEL_COLORS)])


# Recursive function to calculate factorial
def factorial(n):
    # Base Case (終止條件)
    if n == 0 or n == 1:
        return 1

    # Recursive Case (遞迴)
    return n * factorial(n - 1)

if __name__ == "__main__":
    demo_init_chat_model()
    demo_model_comparison()
    demo_message()
    exercise_multi_model()
    # print(factorial(5))  # Output: 120
