from langgraph.graph import StateGraph, END
from typing import TypedDict, NotRequired

from langchain_core.runnables import Runnable
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agent.tool_config import TOOLS

import time
from pathlib import Path


class AgentState(TypedDict):
    input_text: str
    context: NotRequired[str]
    response: NotRequired[str]


# 1. LangChain LLM setup
llm = ChatOllama(model="mistral", base_url="http://192.168.1.158:11434")

# 2. Tool selection prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant for NebulaForge. You have access to tools for answering questions about either the company Code of Conduct or Tax Guidelines."),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# 3. Agent setup
agent_runnable: Runnable = create_tool_calling_agent(llm, TOOLS, prompt=prompt)
agent_executor = AgentExecutor(agent=agent_runnable, tools=TOOLS, verbose=True)


# 4. Node: Run LLM + pick + call tool
def tool_router_node(state: AgentState) -> AgentState:
    result = agent_executor.invoke({"input": state["input_text"]})
    return {**state, "response": result["output"]}


# 5. Build LangGraph
def build_flow_graph():
    g = StateGraph(AgentState)

    g.add_node("route_tool", tool_router_node)
    g.set_entry_point("route_tool")
    g.add_edge("route_tool", END)

    return g.compile()


INBOX_DIR = Path("inbox")
OUTBOX_DIR = Path("outbox")


def run_inbox_loop():
    app = build_flow_graph()
    print("[Agent] Monitoring inbox...")

    while True:
        txt_files = sorted(INBOX_DIR.glob("*.txt"))
        if not txt_files:
            time.sleep(1)
            continue

        input_file = txt_files[0]
        print(f"[Agent] Processing: {input_file.name}")

        input_text = input_file.read_text()
        result = app.invoke({"input_text": input_text})

        output_file = OUTBOX_DIR / input_file.name
        output_file.write_text(result["response"])

        input_file.unlink()  # optionally delete inbox file
        print(f"[Agent] Wrote response to {output_file}")

        time.sleep(0.5)
