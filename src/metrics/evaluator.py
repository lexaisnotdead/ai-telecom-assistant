"""
LLM answer quality evaluation.
Covers faithfulness, relevance, completeness, tone, and A/B prompt testing.
"""
import json
from dataclasses import dataclass, asdict
from typing import List
from openai import OpenAI


@dataclass
class EvalResult:
    question: str
    answer: str
    context: str
    faithfulness: float       # 0-1: whether the model avoids inventing facts
    relevance: float          # 0-1: whether the answer addresses the question
    completeness: float       # 0-1: whether the answer is complete
    tone_appropriate: bool    # whether the tone fits a business client
    notes: str = ""


@dataclass
class ABTestResult:
    prompt_a_name: str
    prompt_b_name: str
    question: str
    response_a: str
    response_b: str
    winner: str               # "A" | "B" | "tie"
    reasoning: str


class LLMEvaluator:
    """
    Uses the LLM-as-judge pattern to evaluate answer quality.
    This is a standard approach in production LLM systems.
    """

    def __init__(self, model: str = "gpt-5.4-mini"):
        self.client = OpenAI()
        self.model = model

    def evaluate_response(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> EvalResult:
        """Evaluate a single answer across multiple metrics."""

        eval_prompt = f"""Evaluate a telecom support assistant answer.

CLIENT QUESTION:
{question}

KNOWLEDGE BASE CONTEXT (what the system knows):
{context}

ASSISTANT ANSWER:
{answer}

Score the answer from 0.0 to 1.0 using these criteria:

1. faithfulness: Are all facts in the answer supported by the context?
   - 1.0 = everything comes from the context, nothing extra
   - 0.0 = includes facts not found in the context (hallucinations)

2. relevance: Does the answer address the client question?
   - 1.0 = fully answers the question
   - 0.0 = off topic

3. completeness: Does the client get enough information?
   - 1.0 = the client gets everything needed
   - 0.0 = incomplete answer or requires too many follow-up questions

4. tone_appropriate: Is the tone appropriate for a business client? (true/false)

Reply strictly in JSON:
{{
  "faithfulness": 0.0-1.0,
  "relevance": 0.0-1.0,
  "completeness": 0.0-1.0,
  "tone_appropriate": true/false,
  "notes": "short comment"
}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": eval_prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        scores = json.loads(response.choices[0].message.content)

        return EvalResult(
            question=question,
            answer=answer,
            context=context,
            faithfulness=scores.get("faithfulness", 0),
            relevance=scores.get("relevance", 0),
            completeness=scores.get("completeness", 0),
            tone_appropriate=scores.get("tone_appropriate", False),
            notes=scores.get("notes", ""),
        )

    def evaluate_batch(self, test_cases: List[dict]) -> dict:
        """
        Evaluate a set of test cases and return aggregate metrics.
        test_cases: [{"question": ..., "answer": ..., "context": ...}, ...]
        """
        results = []
        for case in test_cases:
            result = self.evaluate_response(
                question=case["question"],
                answer=case["answer"],
                context=case["context"],
            )
            results.append(result)

        if not results:
            return {}

        avg_faithfulness = sum(r.faithfulness for r in results) / len(results)
        avg_relevance = sum(r.relevance for r in results) / len(results)
        avg_completeness = sum(r.completeness for r in results) / len(results)
        tone_ok = sum(1 for r in results if r.tone_appropriate) / len(results)

        return {
            "total_evaluated": len(results),
            "avg_faithfulness": round(avg_faithfulness, 3),
            "avg_relevance": round(avg_relevance, 3),
            "avg_completeness": round(avg_completeness, 3),
            "tone_appropriate_rate": round(tone_ok, 3),
            "overall_score": round(
                (avg_faithfulness + avg_relevance + avg_completeness) / 3, 3
            ),
            "details": [asdict(r) for r in results],
        }

    def ab_test_prompts(
        self,
        question: str,
        context: str,
        response_a: str,
        response_b: str,
        prompt_a_name: str = "Variant A",
        prompt_b_name: str = "Variant B",
    ) -> ABTestResult:
        """Compare two answers and decide which one is better."""

        compare_prompt = f"""You are an expert in support answer quality.

Client question: {question}
Context: {context}

{prompt_a_name}: {response_a}

{prompt_b_name}: {response_b}

Which answer is better for a small business client? Consider:
- Information accuracy
- Clarity and conciseness
- Practical usefulness

Reply in JSON:
{{"winner": "A or B or tie", "reasoning": "explanation in 1-2 sentences"}}"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": compare_prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)

        return ABTestResult(
            prompt_a_name=prompt_a_name,
            prompt_b_name=prompt_b_name,
            question=question,
            response_a=response_a,
            response_b=response_b,
            winner=result.get("winner", "tie"),
            reasoning=result.get("reasoning", ""),
        )
