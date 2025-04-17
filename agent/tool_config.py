from langchain.tools import tool
from agent.tools import fetch_from_coc, fetch_from_tax


@tool
def coc_tool(question: str) -> str:
    """Search the Code of Conduct for guidance on behavioral, workplace, or cultural scenarios."""
    result = fetch_from_coc(question)
    context = result["context"]
    return f"[Code of Conduct]\n{context}"


@tool
def tax_tool(question: str) -> str:
    """Search the Tax Handbook for information on employee or contractor tax obligations."""
    result = fetch_from_tax(question)
    context = result["context"]
    return f"[Tax Handbook]\n{context}"


TOOLS = [coc_tool, tax_tool]
