"""LangGraph-based investment evaluation pipeline."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from config import get_config
from models import CompanyInfo, ParsedInput, PipelineContext
from layers.analysis_engine import process_analysis_engine
from layers.document_retrieval_layer import process_document_retrieval_layer
from layers.external_search_layer import process_external_search_layer
from layers.input_layer import process_input_layer
from layers.knowledge_base_layer import process_knowledge_base_layer
from layers.output_layer import process_output_layer
from layers.quality_check_layer import process_quality_check_layer
from layers.report_generation_layer import process_report_generation_layer
from layers.risk_assessment_layer import process_risk_assessment_layer
from layers.scoring_engine import process_scoring_ranking_engine

logger = logging.getLogger(__name__)


class PipelineState(TypedDict, total=False):
    """State passed between LangGraph nodes."""

    context: PipelineContext
    user_input: str
    output_format: str
    save_to_file: bool
    output_path: Optional[str]
    skip_external_search: bool
    max_retries: int
    attempt: int
    quality_passed: bool
    final_output: str


def _input_node(state: PipelineState) -> PipelineState:
    context = process_input_layer(state["user_input"], state["context"])
    return {"context": context}


def _knowledge_node(state: PipelineState) -> PipelineState:
    return {"context": process_knowledge_base_layer(state["context"])}


def _document_node(state: PipelineState) -> PipelineState:
    return {"context": process_document_retrieval_layer(state["context"])}


def _external_node(state: PipelineState) -> PipelineState:
    return {"context": process_external_search_layer(state["context"])}


def _analysis_node(state: PipelineState) -> PipelineState:
    return {"context": process_analysis_engine(state["context"])}


def _scoring_node(state: PipelineState) -> PipelineState:
    return {"context": process_scoring_ranking_engine(state["context"])}


def _risk_node(state: PipelineState) -> PipelineState:
    return {"context": process_risk_assessment_layer(state["context"])}


def _report_node(state: PipelineState) -> PipelineState:
    return {"context": process_report_generation_layer(state["context"])}


def _quality_node(state: PipelineState) -> PipelineState:
    context = process_quality_check_layer(state["context"], state["user_input"])
    passed = bool(context.quality_check and context.quality_check.passed)
    updates: PipelineState = {"context": context, "quality_passed": passed}
    if not passed:
        updates["attempt"] = state.get("attempt", 0) + 1
    return updates


def _reset_node(state: PipelineState) -> PipelineState:
    context = state["context"]
    context.analysis_results = []
    context.risk_assessments = []
    context.final_report = None
    context.quality_check = None
    context.processing_steps.append("LANGGRAPH_PIPELINE: 재시도를 위한 상태 초기화")
    return {"context": context}


def _output_node(state: PipelineState) -> PipelineState:
    final_output = process_output_layer(
        context=state["context"],
        output_format=state["output_format"],
        save_to_file=state["save_to_file"],
        output_path=state["output_path"],
    )
    return {"final_output": final_output}


def _should_run_external(state: PipelineState) -> str:
    return "skip" if state.get("skip_external_search") else "run"


def _should_retry(state: PipelineState) -> str:
    if state.get("quality_passed"):
        return "pass"
    attempt = state.get("attempt", 0)
    if attempt <= state.get("max_retries", 0):
        return "retry"
    return "pass"


def build_langgraph_pipeline() -> StateGraph:
    builder = StateGraph(PipelineState)

    builder.add_node("INPUT", _input_node)
    builder.add_node("KNOWLEDGE", _knowledge_node)
    builder.add_node("DOCUMENT", _document_node)
    builder.add_node("EXTERNAL", _external_node)
    builder.add_node("ANALYSIS", _analysis_node)
    builder.add_node("SCORING", _scoring_node)
    builder.add_node("RISK", _risk_node)
    builder.add_node("REPORT", _report_node)
    builder.add_node("QUALITY", _quality_node)
    builder.add_node("RESET", _reset_node)
    builder.add_node("OUTPUT", _output_node)

    builder.add_edge(START, "INPUT")
    builder.add_edge("INPUT", "KNOWLEDGE")
    builder.add_edge("KNOWLEDGE", "DOCUMENT")

    builder.add_conditional_edges(
        "DOCUMENT",
        _should_run_external,
        {
            "run": "EXTERNAL",
            "skip": "ANALYSIS",
        },
    )
    builder.add_edge("EXTERNAL", "ANALYSIS")

    builder.add_edge("ANALYSIS", "SCORING")
    builder.add_edge("SCORING", "RISK")
    builder.add_edge("RISK", "REPORT")
    builder.add_edge("REPORT", "QUALITY")

    builder.add_conditional_edges(
        "QUALITY",
        _should_retry,
        {
            "retry": "RESET",
            "pass": "OUTPUT",
        },
    )
    builder.add_edge("RESET", "ANALYSIS")
    builder.add_edge("OUTPUT", END)

    return builder


def run_langgraph_investment_evaluation(
    user_input: str,
    output_format: str = "console",
    save_to_file: bool = False,
    output_path: Optional[str] = None,
    skip_external_search: bool = False,
    max_retries: int = 1,
) -> str:
    logger.info("LangGraph pipeline execution started: %s", user_input)

    context = PipelineContext(
        parsed_input=ParsedInput(company_name="", evaluation_type=None),
        company_info=CompanyInfo(name=""),
        execution_start_time=datetime.now(),
    )

    initial_state: PipelineState = {
        "context": context,
        "user_input": user_input,
        "output_format": output_format,
        "save_to_file": save_to_file,
        "output_path": output_path,
        "skip_external_search": skip_external_search,
        "max_retries": max_retries,
        "attempt": 0,
        "quality_passed": False,
    }

    graph = build_langgraph_pipeline().compile()
    recursion_limit = max(50, (max_retries + 1) * 10)
    final_state = graph.invoke(initial_state, config={"recursion_limit": recursion_limit})
    logger.info("LangGraph pipeline execution finished")
    return final_state.get("final_output", "")


if __name__ == "__main__":
    _ = get_config()
    sample_output = run_langgraph_investment_evaluation(
        "토스의 투자 가치를 평가해줘",
        output_format="console",
        save_to_file=False,
        skip_external_search=False,
        max_retries=1,
    )
    print("\n=== LangGraph Pipeline Output ===")
    print(sample_output)
