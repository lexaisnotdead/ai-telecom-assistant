"""
Telecom LLM Assistant entry point.
Run with: python3 main.py [--mode chat|agent|eval|demo]
"""
import os
import argparse
from dotenv import load_dotenv
from colorama import Fore, Style, init

load_dotenv()
init(autoreset=True)


def ensure_indexed(persist_dir: str = "./chroma_db", data_dir: str = "./data"):
    """Index the knowledge base if it has not been indexed yet."""
    from src.rag.retriever import TelecomIndexer, TelecomRetriever

    persist_dir = os.getenv("PERSIST_DIR", persist_dir)
    data_dir = os.getenv("DATA_DIR", data_dir)
    retriever = TelecomRetriever(persist_dir=persist_dir)
    if not retriever.is_ready():
        print(f"{Fore.YELLOW}Indexing the knowledge base...{Style.RESET_ALL}")
        indexer = TelecomIndexer(persist_dir=persist_dir)
        count = indexer.index_directory(data_dir)
        print(f"{Fore.GREEN}Done: {count} chunk(s){Style.RESET_ALL}")
    return retriever


def run_chat_mode():
    """Regular chat mode with RAG."""
    print(f"\n{Fore.CYAN}=== Telecom Assistant (RAG + Multi-turn Dialogue) ==={Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Type 'exit' to quit or 'new' to start a new session{Style.RESET_ALL}\n")

    retriever = ensure_indexed()

    from src.dialog.manager import DialogManager
    manager = DialogManager(retriever=retriever)

    session_id = "main_session"

    while True:
        try:
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            break
        if user_input.lower() == "new":
            manager.reset_session(session_id)
            print(f"{Fore.YELLOW}Session reset{Style.RESET_ALL}")
            continue

        result = manager.chat(session_id=session_id, user_message=user_input)

        print(f"\n{Fore.BLUE}Alice [{result['intent']}]:{Style.RESET_ALL}")
        print(result["reply"])
        print(f"{Fore.WHITE}[turn {result['turn']}, tokens: {result['tokens_used']}]{Style.RESET_ALL}\n")


def run_agent_mode():
    """ReAct agent mode with tools."""
    print(f"\n{Fore.CYAN}=== ReAct Agent with Tools ==={Style.RESET_ALL}")
    print(f"{Fore.YELLOW}The agent can search the knowledge base, calculate pricing, and check compatibility{Style.RESET_ALL}\n")

    retriever = ensure_indexed()

    from src.agents.react_agent import create_telecom_agent
    agent = create_telecom_agent(retriever=retriever)

    while True:
        try:
            user_input = input(f"{Fore.GREEN}Query: {Style.RESET_ALL}").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input or user_input.lower() == "exit":
            break

        print(f"\n{Fore.YELLOW}--- Reasoning Process ---{Style.RESET_ALL}")
        result = agent.invoke({"messages": [("user", user_input)]})
        print(f"\n{Fore.BLUE}Answer:{Style.RESET_ALL} {result['messages'][-1].content}\n")


def run_eval_mode():
    """Quality evaluation mode using a set of test cases."""
    print(f"\n{Fore.CYAN}=== Answer Quality Evaluation ==={Style.RESET_ALL}\n")

    retriever = ensure_indexed()

    from src.dialog.manager import DialogManager
    from src.metrics.evaluator import LLMEvaluator

    manager = DialogManager(retriever=retriever)
    evaluator = LLMEvaluator()

    test_cases_input = [
        "Which plan is suitable for a cafe with 5 employees?",
        "What should I do if the internet is not working?",
        "How much does it cost to connect an 8-800 number?",
        "What documents are required for a sole proprietor?",
    ]

    print("Generating answers and evaluating them...")
    eval_cases = []

    for question in test_cases_input:
        result = manager.chat(session_id=f"eval_{hash(question)}", user_message=question)
        context = retriever.get_context(question)
        eval_cases.append({
            "question": question,
            "answer": result["reply"],
            "context": context,
        })
        print(f"  ✓ {question[:50]}...")

    print("\nRunning the LLM-as-judge evaluation...")
    metrics = evaluator.evaluate_batch(eval_cases)

    print(f"\n{Fore.CYAN}=== Results ==={Style.RESET_ALL}")
    print(f"Answers evaluated: {metrics['total_evaluated']}")
    print(f"Faithfulness:      {Fore.GREEN}{metrics['avg_faithfulness']:.1%}{Style.RESET_ALL}")
    print(f"Relevance:         {Fore.GREEN}{metrics['avg_relevance']:.1%}{Style.RESET_ALL}")
    print(f"Completeness:      {Fore.GREEN}{metrics['avg_completeness']:.1%}{Style.RESET_ALL}")
    print(f"Appropriate tone:  {Fore.GREEN}{metrics['tone_appropriate_rate']:.1%}{Style.RESET_ALL}")
    print(f"Overall score:     {Fore.YELLOW}{metrics['overall_score']:.1%}{Style.RESET_ALL}")


def run_demo():
    """Run a non-interactive demo scenario for screenshots or presentations."""
    print(f"\n{Fore.CYAN}=== Demo: HoReCa Scenario ==={Style.RESET_ALL}\n")

    retriever = ensure_indexed()

    from src.dialog.manager import DialogManager
    manager = DialogManager(retriever=retriever)

    demo_dialog = [
        "Hello! We have a small restaurant with 8 employees and we are looking for a telecom solution.",
        "It is important for us to receive calls for table reservations. Is there anything tailored for HoReCa?",
        "Can it be integrated with YCLIENTS?",
        "What would the approximate monthly cost be?",
    ]

    session_id = "demo_horeca"
    for message in demo_dialog:
        print(f"{Fore.GREEN}Client:{Style.RESET_ALL} {message}")
        result = manager.chat(session_id=session_id, user_message=message)
        print(f"{Fore.BLUE}Alice:{Style.RESET_ALL} {result['reply']}")
        print(f"{Fore.WHITE}[intent: {result['intent']}]{Style.RESET_ALL}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telecom LLM Assistant")
    parser.add_argument(
        "--mode",
        choices=["chat", "agent", "eval", "demo"],
        default="demo",
        help="Run mode"
    )
    args = parser.parse_args()

    if not os.getenv("GOOGLE_API_KEY"):
        print(f"{Fore.RED}Error: GOOGLE_API_KEY is not set in .env{Style.RESET_ALL}")
        exit(1)

    modes = {
        "chat": run_chat_mode,
        "agent": run_agent_mode,
        "eval": run_eval_mode,
        "demo": run_demo,
    }
    modes[args.mode]()
