from src.agents.react_agent import build_tools

class StubRetriever:
    def get_context(self, query: str, k: int = 3) -> str:
        return f"stub context for {query} ({k})"


def get_tool_by_name(name: str):
    tools = build_tools(StubRetriever())
    for tool in tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"Tool {name} not found")


def test_calculate_cost_returns_known_plan_price():
    tool = get_tool_by_name("calculate_cost")
    result = tool.invoke("plan: Business, phone_numbers: 3, months: 12")

    assert "Plan 'Business': 13.5 USD/month per number" in result
    assert "Monthly: 40.5 USD | Total: 486.0 USD" in result


def test_check_compatibility_uses_knowledge_base_facts():
    tool = get_tool_by_name("check_compatibility")

    assert "Restaurant" in tool.invoke("yclients")
    assert "Shop" in tool.invoke("Bitrix24")
    assert "knowledge base" in tool.invoke("1C")
