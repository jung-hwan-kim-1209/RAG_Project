"""Graph-based orchestration for the investment evaluation pipeline."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

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


@dataclass
class GraphNode:
    """단일 파이프라인 노드를 나타냅니다."""

    name: str
    handler: Callable[[Dict], None]
    condition: Optional[Callable[[Dict], bool]] = None

    def should_run(self, state: Dict) -> bool:
        if self.condition is None:
            return True
        return bool(self.condition(state))


class GraphPipeline:
    """그래프 형태로 파이프라인 단계를 실행합니다."""

    def __init__(self):
        self.config = get_config()
        self.nodes: List[GraphNode] = [
            GraphNode("INPUT_LAYER", self._run_input),
            GraphNode("KNOWLEDGE_BASE_LAYER", self._run_knowledge_base),
            GraphNode("DOCUMENT_RETRIEVAL_LAYER", self._run_document_retrieval),
            GraphNode(
                "EXTERNAL_SEARCH_LAYER",
                self._run_external_search,
                condition=lambda state: not state.get("skip_external_search", False),
            ),
            GraphNode("ANALYSIS_ENGINE", self._run_analysis),
            GraphNode("SCORING_RANKING_ENGINE", self._run_scoring),
            GraphNode("RISK_ASSESSMENT_LAYER", self._run_risk),
            GraphNode("REPORT_GENERATION_LAYER", self._run_report),
            GraphNode("QUALITY_CHECK_LAYER", self._run_quality_check),
        ]

    # ------------------------------------------------------------------
    # 노드 래퍼 함수
    # ------------------------------------------------------------------
    def _run_input(self, state: Dict) -> None:
        context: PipelineContext = state["context"]
        state["context"] = process_input_layer(state["user_input"], context)

    def _run_knowledge_base(self, state: Dict) -> None:
        state["context"] = process_knowledge_base_layer(state["context"])

    def _run_document_retrieval(self, state: Dict) -> None:
        state["context"] = process_document_retrieval_layer(state["context"])

    def _run_external_search(self, state: Dict) -> None:
        state["context"] = process_external_search_layer(state["context"])

    def _run_analysis(self, state: Dict) -> None:
        state["context"] = process_analysis_engine(state["context"])

    def _run_scoring(self, state: Dict) -> None:
        state["context"] = process_scoring_ranking_engine(state["context"])

    def _run_risk(self, state: Dict) -> None:
        state["context"] = process_risk_assessment_layer(state["context"])

    def _run_report(self, state: Dict) -> None:
        state["context"] = process_report_generation_layer(state["context"])

    def _run_quality_check(self, state: Dict) -> None:
        ctx = process_quality_check_layer(state["context"], state["user_input"])
        state["context"] = ctx
        quality = ctx.quality_check
        state["quality_passed"] = bool(quality and quality.passed)

    # ------------------------------------------------------------------
    # 실행 진입점
    # ------------------------------------------------------------------
    def execute(
        self,
        user_input: str,
        output_format: str = "console",
        save_to_file: bool = False,
        output_path: Optional[str] = None,
        skip_external_search: bool = False,
        max_retries: int = 1,
    ) -> str:
        logger.info("Graph pipeline execution started: %s", user_input)

        context = PipelineContext(
            parsed_input=ParsedInput(company_name="", evaluation_type=None),
            company_info=CompanyInfo(name=""),
            execution_start_time=datetime.now(),
        )

        state: Dict = {
            "user_input": user_input,
            "context": context,
            "output_format": output_format,
            "save_to_file": save_to_file,
            "output_path": output_path,
            "skip_external_search": skip_external_search,
            "max_retries": max_retries,
            "attempt": 0,
            "quality_passed": False,
        }

        while state["attempt"] <= max_retries:
            try:
                logger.debug("Graph attempt %d", state["attempt"] + 1)
                self._run_attempt(state)
                if state.get("quality_passed"):
                    break

                if state["attempt"] >= max_retries:
                    logger.warning("Maximum retries reached; continuing with current result.")
                    break

                logger.info("Quality check failed. Retrying (attempt %d/%d)", state["attempt"] + 1, max_retries)
                self._reset_for_retry(state)
                state["attempt"] += 1

            except Exception as err:  # pragma: no cover
                logger.exception("Graph pipeline error during attempt %d", state["attempt"] + 1)
                state["context"].processing_steps.append(f"GRAPH_PIPELINE_ERROR: {err}")
                if state["attempt"] >= max_retries:
                    break
                self._reset_for_retry(state)
                state["attempt"] += 1

        final_output = process_output_layer(
            context=state["context"],
            output_format=output_format,
            save_to_file=save_to_file,
            output_path=output_path,
        )
        logger.info("Graph pipeline execution finished")
        return final_output

    def _run_attempt(self, state: Dict) -> None:
        for node in self.nodes:
            if not node.should_run(state):
                state["context"].processing_steps.append(f"{node.name}: 건너뜀")
                continue
            logger.debug("Running node: %s", node.name)
            node.handler(state)

    def _reset_for_retry(self, state: Dict) -> None:
        ctx: PipelineContext = state["context"]
        ctx.analysis_results = []
        ctx.risk_assessments = []
        ctx.final_report = None
        ctx.quality_check = None
        ctx.processing_steps.append("GRAPH_PIPELINE: 재시도를 위한 상태 초기화")


def run_graph_investment_evaluation(
    user_input: str,
    output_format: str = "console",
    save_to_file: bool = False,
    output_path: Optional[str] = None,
    skip_external_search: bool = False,
    max_retries: int = 1,
) -> str:
    pipeline = GraphPipeline()
    return pipeline.execute(
        user_input=user_input,
        output_format=output_format,
        save_to_file=save_to_file,
        output_path=output_path,
        skip_external_search=skip_external_search,
        max_retries=max_retries,
    )
