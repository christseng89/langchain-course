import operator

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

load_dotenv()


LLM = init_chat_model("gpt-4o-mini", temperature=0.7)
print(f"\033[93mUsing LLM: {LLM.model_name}\033[0m")


def save_graph_png(app, png_file: str) -> None:
  png_bytes = app.get_graph().draw_mermaid_png()
  with open(png_file, "wb") as f:
    f.write(png_bytes)
  print(f"\033[93mGraph saved to {png_file}\033[0m")


class ConversationState(TypedDict):
  messages: Annotated[list, operator.add]
  sentiment: str
  response_count: int


def create_conversation_graph():
  # Define node function
  def analyze_sentiment(state: ConversationState) -> dict:
    """Analyze the sentiment of the last message."""
    last_message = state["messages"][-1]

    response = LLM.invoke(
      [
        SystemMessage(
          content="Classify sentiment as: positive, negative, or neutral. Reply with just the word."
        ),
        HumanMessage(content=last_message),
      ]
    )

    return {"sentiment": response.content.lower().strip()}

  def generate_response(state: ConversationState) -> dict:
    """Generate appropriate response based on sentiment."""
    sentiment = state["sentiment"]
    last_message = state["messages"][-1]
    # print(f"\033[38;5;208mLast message: {last_message}\033[0m")

    system_prompts = {
      "positive": "Respond enthusiastically and build on their positive energy.",
      "negative": "Respond empathetically and offer support.",
      "neutral": "Respond helpfully and informatively.",
    }

    prompt = system_prompts.get(sentiment, system_prompts["neutral"])
    # print(f"\033[38;5;208mPrompt: {prompt}\033[0m")
    response = LLM.invoke([SystemMessage(content=prompt), HumanMessage(content=last_message)])

    return {"messages": [f"AI: {response.content}"], "response_count": 1}

  # Create graph
  graph = StateGraph(ConversationState)

  # Add nodes
  graph.add_node("analyze_sentiment", analyze_sentiment)
  graph.add_node("generate_response", generate_response)

  # Add edges
  graph.add_edge(START, "analyze_sentiment")
  graph.add_edge("analyze_sentiment", "generate_response")
  graph.add_edge("generate_response", END)

  app = graph.compile()

  save_graph_png(app, "graph8_conversation.png")

  return app


def demo_conversation():
  app = create_conversation_graph()

  # Simulate a conversation

  queries = [
    "I just got promoted at work! I'm so excited!",
    "My computer crashed and I lost all my work...",
    "What's the weather like today?",
    "I am going to work overtime today.",
  ]

  print("\n\033[94mConversation Graph Demo:\n\033[0m")

  for query in queries:
    result = app.invoke({"messages": [f"Human: {query}"], "sentiment": "", "response_count": 0})

    print(f"\033[92mQuery: {query}\033[0m\n")

    print("\033[93mResponse Result:\033[0m")
    print(f"Sentiment: {result['sentiment']}")

    result_msg = result["messages"]
    print(f"Response: length = {len(result_msg)}")
    for i, msg in enumerate(result_msg):
      print(f"{i + 1}. {msg}")

    # print("\n")
    # print("-" * 40)


if __name__ == "__main__":
  demo_conversation()
