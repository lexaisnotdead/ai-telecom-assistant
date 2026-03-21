from typing import Annotated
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.prompts.templates import AGENT_SYSTEM_PROMPT
from src.rag.retriever import TelecomRetriever


# =============================================================================
# State: what is stored between graph steps
# =============================================================================

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # add_messages reducer appends new messages instead of replacing old ones


# =============================================================================
# Agent tools
# =============================================================================

def build_tools(retriever: TelecomRetriever) -> list:

    @tool
    def search_knowledge_base(query: str) -> str:
        """Search for information about company plans, services, and technical issues.
        Use it for any product-related question. Input: a question in English."""
        return retriever.get_context(query, k=3)

    @tool
    def calculate_cost(params: str) -> str:
        """Calculate service cost.
        Input format: 'plan: Name, phone_numbers: N, months: M'"""
        plan_prices = {"start": 5.50, "business": 13.50, "team": 28.00}
        try:
            parts = dict(p.strip().split(": ", 1) for p in params.split(","))
            plan = parts.get("plan", "").lower()
            count = int(parts.get("phone_numbers", 1))
            months = int(parts.get("months", 1))
            price = plan_prices.get(plan, 0)
            if not price:
                return f"Plan '{plan}' was not found. Available plans: Start, Business, Team."
            return (
                f"Plan '{plan.title()}': {price} USD/month per number\n"
                f"Phone numbers: {count}, period: {months} month(s)\n"
                f"Monthly: {price * count} USD | Total: {price * count * months} USD"
            )
        except Exception:
            return "Format: 'plan: Business, phone_numbers: 3, months: 12'"

    @tool
    def check_compatibility(service: str) -> str:
        """Check compatibility with external systems such as CRM, booking, and accounting tools.
        Input: the system name, for example 'yclients', 'amoCRM', or '1C'."""
        integrations = {
            "yclients": "Compatible with the 'Restaurant' package. Setup is available via API.",
            "reservio": "Listed as a supported booking system for the 'Restaurant' package.",
            "amocrm": "Listed as a CRM integration for the 'Shop' package.",
            "bitrix24": "Listed as a CRM integration for the 'Shop' package.",
            "pms": "The 'Hotel' package includes integration with PMS systems.",
        }
        key = service.lower().strip()
        return integrations.get(
            key,
            f"No compatibility information for '{service}' was found in the knowledge base."
        )

    @tool
    def get_support_contacts(issue_type: str) -> str:
        """Return the contact details for the right support team.
        Input: issue type such as 'technical', 'billing', 'pbx', or 'connection'."""
        contacts = {
            "technical": "Technical support: 8-800-XXX-XX-XX (24/7 for 'Team', 9:00-21:00 for others)",
            "billing": "No dedicated billing contact is listed in the knowledge base. Use the client portal or general support: 8-800-XXX-XX-XX.",
            "pbx": "PBX technical support: pbx-support@carrier.com",
            "connection": "Connection lead times depend on the service: same day for mobile, 1-2 business days for virtual PBX, 3-5 business days for 8-800 numbers.",
        }
        for k, v in contacts.items():
            if k in issue_type.lower():
                return v
        return "General support: 8-800-XXX-XX-XX"

    return [search_knowledge_base, calculate_cost, check_compatibility, get_support_contacts]


# =============================================================================
# Agent graph
# =============================================================================

def create_telecom_agent(retriever: TelecomRetriever):
    """
    Build a ReAct agent as a StateGraph.

    Graph structure:
        START
          │
          ▼
        agent  ◄──────────────┐
          │                   │
          ├── tool_calls? ──► tools
          │
          └── no calls ──► END
    """
    tools = build_tools(retriever)
    llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0).bind_tools(tools)

    system_message = SystemMessage(content=AGENT_SYSTEM_PROMPT)

    # Node 1: call the LLM
    def agent_node(state: AgentState) -> AgentState:
        """The LLM inspects messages and either answers or calls a tool."""
        messages = [system_message] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    # Node 2: execute tools
    tool_node = ToolNode(tools)  # Calls the requested @tool and writes the result into messages

    # Conditional transition
    def should_continue(state: AgentState) -> str:
        """Go to tools if the last message contains tool_calls, otherwise finish."""
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return END

    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")  # After a tool call, return to the LLM

    return graph.compile()


def run_agent(retriever: TelecomRetriever, question: str) -> str:
    """Run the agent and return the final answer."""
    agent = create_telecom_agent(retriever)
    result = agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content
