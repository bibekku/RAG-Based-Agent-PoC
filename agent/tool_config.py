from langchain.tools import tool
from agent.tools import fetch_from_coc, fetch_from_tax


@tool
def coc_tool(question: str) -> str:
    """Search the Code of Conduct for guidance on behavioral, workplace, or cultural scenarios."""
    return fetch_from_coc(question)


@tool
def tax_tool(question: str) -> str:
    """Search the Tax Handbook for information on employee or contractor tax obligations."""
    return fetch_from_tax(question)


TOOLS = [coc_tool, tax_tool]
