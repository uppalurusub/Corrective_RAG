
import logging
import re
from typing import Literal

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

RetrievalStrategy = Literal[
    "dense",
    "hybrid",
]


class QueryRouteDecision(BaseModel):
    """Structured decision returned by the query router."""

    strategy: RetrievalStrategy = Field(
        description=(
            "dense for semantic conceptual questions; "
            "hybrid for questions involving exact terms, "
            "identifiers, dates, or mixed semantic and lexical needs"
        )
    )
    normalized_query: str = Field(
        min_length=1,
        max_length=5000,
        description="A faithful, standalone retrieval query",
    )
    needs_rewrite: bool = Field(
        description=(
            "Whether the original query is ambiguous, incomplete, "
            "or poorly suited to retrieval"
        )
    )
    reason: str = Field(
        min_length=1,
        max_length=500,
        description="Brief explanation of the routing decision",
    )


class QueryRouterAgent:
    """
    Adaptive retrieval strategy selector.

    Responsibilities:
    - Validate the incoming question.
    - Select an available retrieval strategy.
    - Normalize retrieval queries when needed.
    - Return a structured, validated decision.

    This agent does not retrieve documents, grade evidence,
    or generate the final answer.
    """

    def __init__(
        self,
        llm=None,
        available_strategies: tuple[str, ...] = (
            "dense",
            "hybrid",
        ),
        default_strategy: str = "hybrid",
        enable_llm_routing: bool = True,
    ) -> None:
        self.available_strategies = tuple(
            dict.fromkeys(available_strategies)
        )

        if not self.available_strategies:
            raise ValueError(
                "At least one retrieval strategy is required."
            )

        if any(
            strategy not in ("dense", "hybrid")
            for strategy in self.available_strategies
        ):
            raise ValueError(
                "Strategies must be 'dense' or 'hybrid'."
            )

        if default_strategy not in self.available_strategies:
            raise ValueError(
                "default_strategy must be available."
            )

        self.default_strategy = default_strategy
        self.llm = llm
        self.enable_llm_routing = enable_llm_routing

        self._router_chain = None

        if enable_llm_routing and llm is not None:
            self._router_chain = (
                self._build_router_chain(llm)
            )

        logger.info(
            "Query router initialized | strategies=%s "
            "| default=%s | llm_routing=%s",
            self.available_strategies,
            self.default_strategy,
            self._router_chain is not None,
        )

    @staticmethod
    def _build_router_chain(llm):
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """
You are a retrieval-strategy classifier for a RAG system.

Choose the best strategy using only the user's query.

Strategies:
- dense: semantic similarity is the primary retrieval need.
- hybrid: combine semantic and lexical retrieval. Prefer this for
  exact terminology, IDs, names, dates, acronyms, technical terms,
  quoted phrases, or mixed semantic and keyword requirements.

Rules:
- Preserve the user's intent and important constraints.
- Do not answer the question.
- Do not invent facts or entities.
- Rewrite only when needed to make the query self-contained.
- If uncertain, prefer hybrid when available.
- Return only the requested structured output.
"""
            ),
            (
                "human",
                "Available strategies: {strategies}\n"
                "User question:\n{question}"
            ),
        ])

        return prompt | llm.with_structured_output(
            QueryRouteDecision
        )

    def route(self, question: str) -> QueryRouteDecision:
        """
        Classify a query and return a validated routing decision.

        If model-based routing fails, use the deterministic fallback.
        """
        question = self._validate_question(question)

        if self._router_chain is None:
            return self._fallback_route(question)

        try:
            decision = self._router_chain.invoke({
                "question": question,
                "strategies": ", ".join(
                    self.available_strategies
                ),
            })

            if not isinstance(decision, QueryRouteDecision):
                decision = QueryRouteDecision.model_validate(
                    decision
                )

            # Never allow the LLM to select an unavailable strategy.
            if decision.strategy not in self.available_strategies:
                logger.warning(
                    "Router selected unavailable strategy; "
                    "using fallback."
                )
                return self._fallback_route(question)

            normalized_query = (
                decision.normalized_query.strip()
            )

            if not normalized_query:
                return self._fallback_route(question)

            return decision.model_copy(update={
                "normalized_query": normalized_query,
            })

        except Exception:
            logger.exception(
                "LLM query routing failed; using fallback."
            )
            return self._fallback_route(question)

    def _fallback_route(
        self,
        question: str,
    ) -> QueryRouteDecision:
        """
        Deterministic fallback for provider outages or invalid output.

        This is intentionally conservative. Tune these rules using
        measured performance on a labeled query dataset.
        """
        exact_term_pattern = re.compile(
            r"""
            ["'`]                  # quoted phrase or identifier
            | \b\d{4,}\b           # long numeric identifiers
            | \b[A-Z]{2,}\d*       # acronyms / codes
            | \b\d{4}-\d{2}-\d{2}\b
            """,
            re.VERBOSE,
        )

        preferred = (
            "hybrid"
            if exact_term_pattern.search(question)
            else self.default_strategy
        )

        if preferred not in self.available_strategies:
            preferred = self.default_strategy

        return QueryRouteDecision(
            strategy=preferred,
            normalized_query=question,
            needs_rewrite=False,
            reason="Deterministic fallback routing.",
        )

    @staticmethod
    def _validate_question(question: str) -> str:
        if not isinstance(question, str):
            raise ValueError("Question must be a string.")

        question = question.strip()

        if len(question) < 3:
            raise ValueError(
                "Question must contain at least 3 characters."
            )

        if len(question) > 5000:
            raise ValueError(
                "Question must not exceed 5000 characters."
            )

        return question
