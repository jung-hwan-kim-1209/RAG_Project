"""
AI 스타트업 투자 평가 에이전트 메인 파이프라인
10개 레이어를 순차적으로 실행하는 파이프라인
"""
import logging
import os
from typing import Optional, Dict, Any
from datetime import datetime

from models import PipelineContext, CompanyInfo, ParsedInput

# Layer imports
from layers.input_layer import process_input_layer
from layers.knowledge_base_layer import process_knowledge_base_layer
from layers.document_retrieval_layer import process_document_retrieval_layer
from layers.external_search_layer import process_external_search_layer
from layers.analysis_engine import process_analysis_engine
from layers.scoring_engine import process_scoring_ranking_engine
from layers.risk_assessment_layer import process_risk_assessment_layer
from layers.report_generation_layer import process_report_generation_layer
from layers.quality_check_layer import process_quality_check_layer
from layers.output_layer import process_output_layer

from config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InvestmentEvaluationPipeline:
    """AI 스타트업 투자 평가 파이프라인"""

    def __init__(self):
        self.config = get_config()
        self.pipeline_layers = [
            ("INPUT_LAYER", process_input_layer),
            ("KNOWLEDGE_BASE_LAYER", process_knowledge_base_layer),
            ("DOCUMENT_RETRIEVAL_LAYER", process_document_retrieval_layer),
            ("EXTERNAL_SEARCH_LAYER", process_external_search_layer),
            ("ANALYSIS_ENGINE", process_analysis_engine),
            ("SCORING_RANKING_ENGINE", process_scoring_ranking_engine),
            ("RISK_ASSESSMENT_LAYER", process_risk_assessment_layer),
            ("REPORT_GENERATION_LAYER", process_report_generation_layer),
            ("QUALITY_CHECK_LAYER", process_quality_check_layer),
            ("OUTPUT_LAYER", process_output_layer)
        ]

    def execute_pipeline(
        self,
        user_input: str,
        output_format: str = None,
        save_to_file: bool = False,
        output_path: str = None,
        skip_external_search: bool = False,
        max_retries: int = None
    ) -> str:
        """파이프라인 실행"""

        if output_format is None:
            output_format = os.getenv("DEFAULT_OUTPUT_FORMAT", "console")
        if max_retries is None:
            max_retries = int(os.getenv("MAX_RETRIES", "1"))

        logger.info(f"파이프라인 실행 시작: {user_input}")

        # 컨텍스트 초기화
        context = PipelineContext(
            parsed_input=ParsedInput(company_name="", evaluation_type=None),
            company_info=CompanyInfo(name=""),
            execution_start_time=datetime.now()
        )

        original_request = user_input
        retry_count = 0

        while retry_count <= max_retries:
            try:
                # Layer 1: INPUT LAYER
                logger.info("Layer 1: INPUT LAYER 실행")
                context = process_input_layer(user_input, context)

                # Layer 2: KNOWLEDGE BASE LAYER
                logger.info("Layer 2: KNOWLEDGE BASE LAYER 실행")
                context = process_knowledge_base_layer(context)

                # Layer 3: DOCUMENT RETRIEVAL LAYER
                logger.info("Layer 3: DOCUMENT RETRIEVAL LAYER 실행")
                context = process_document_retrieval_layer(context)

                # Layer 4: EXTERNAL SEARCH LAYER (옵션)
                if not skip_external_search:
                    logger.info("Layer 4: EXTERNAL SEARCH LAYER 실행")
                    context = process_external_search_layer(context)
                else:
                    context.processing_steps.append("EXTERNAL_SEARCH_LAYER: 건너뜀")

                # Layer 5: ANALYSIS ENGINE
                logger.info("Layer 5: ANALYSIS ENGINE 실행")
                context = process_analysis_engine(context)

                # Layer 6: SCORING & RANKING ENGINE
                logger.info("Layer 6: SCORING & RANKING ENGINE 실행")
                context = process_scoring_ranking_engine(context)

                # Layer 7: RISK ASSESSMENT LAYER
                logger.info("Layer 7: RISK ASSESSMENT LAYER 실행")
                context = process_risk_assessment_layer(context)

                # Layer 8: REPORT GENERATION LAYER
                logger.info("Layer 8: REPORT GENERATION LAYER 실행")
                context = process_report_generation_layer(context)

                # Layer 9: QUALITY CHECK LAYER
                logger.info("Layer 9: QUALITY CHECK LAYER 실행")
                context = process_quality_check_layer(context, original_request)

                # 품질 검증 결과 확인
                if context.quality_check and context.quality_check.passed:
                    break  # 품질 검증 통과 시 루프 종료
                elif retry_count < max_retries:
                    logger.info(f"품질 검증 실패 - 재시도 {retry_count + 1}/{max_retries}")
                    retry_count += 1
                    # 컨텍스트 일부 재설정
                    context.analysis_results = []
                    context.risk_assessments = []
                    context.final_report = None
                    continue
                else:
                    logger.warning("최대 재시도 횟수 초과 - 현재 결과로 진행")
                    break

            except Exception as e:
                logger.error(f"파이프라인 실행 오류 (시도 {retry_count + 1}): {e}")
                if retry_count < max_retries:
                    retry_count += 1
                    continue
                else:
                    # 오류 발생 시 기본 응답
                    context.processing_steps.append(f"파이프라인 실행 오류: {str(e)}")
                    break

        # Layer 10: OUTPUT LAYER
        logger.info("Layer 10: OUTPUT LAYER 실행")
        final_output = process_output_layer(
            context=context,
            output_format=output_format,
            save_to_file=save_to_file,
            output_path=output_path
        )

        logger.info("파이프라인 실행 완료")
        return final_output

    def execute_partial_pipeline(
        self,
        user_input: str,
        start_layer: str = "INPUT_LAYER",
        end_layer: str = "OUTPUT_LAYER",
        context: Optional[PipelineContext] = None
    ) -> PipelineContext:
        """부분 파이프라인 실행"""

        if context is None:
            context = PipelineContext(
                parsed_input=ParsedInput(company_name="", evaluation_type=None),
                company_info=CompanyInfo(name=""),
                execution_start_time=datetime.now()
            )

        # 레이어 인덱스 찾기
        layer_names = [name for name, _ in self.pipeline_layers]
        start_idx = layer_names.index(start_layer) if start_layer in layer_names else 0
        end_idx = layer_names.index(end_layer) if end_layer in layer_names else len(self.pipeline_layers) - 1

        # 선택된 레이어들 실행
        for i in range(start_idx, end_idx + 1):
            layer_name, layer_func = self.pipeline_layers[i]
            logger.info(f"부분 실행: {layer_name}")

            try:
                if layer_name == "INPUT_LAYER":
                    context = layer_func(user_input, context)
                elif layer_name == "QUALITY_CHECK_LAYER":
                    context = layer_func(context, user_input)
                elif layer_name == "OUTPUT_LAYER":
                    # OUTPUT_LAYER는 문자열 반환하므로 별도 처리
                    break
                else:
                    context = layer_func(context)

            except Exception as e:
                logger.error(f"{layer_name} 실행 오류: {e}")
                context.processing_steps.append(f"{layer_name}: 실행 오류 - {str(e)}")

        return context

    def get_pipeline_status(self, context: PipelineContext) -> Dict[str, Any]:
        """파이프라인 실행 상태 조회"""
        return {
            "execution_start_time": context.execution_start_time.isoformat() if context.execution_start_time else None,
            "execution_end_time": context.execution_end_time.isoformat() if context.execution_end_time else None,
            "processed_steps": len(context.processing_steps),
            "processing_steps": context.processing_steps,
            "company_name": context.company_info.name if context.company_info else "",
            "analysis_completed": len(context.analysis_results),
            "risks_assessed": len(context.risk_assessments),
            "documents_retrieved": len(context.retrieved_documents),
            "external_results": len(context.external_search_results),
            "final_report_generated": context.final_report is not None,
            "quality_check_passed": context.quality_check.passed if context.quality_check else None
        }

def create_pipeline() -> InvestmentEvaluationPipeline:
    """파이프라인 생성자"""
    return InvestmentEvaluationPipeline()

def run_investment_evaluation(
    user_input: str,
    output_format: str = None,
    save_to_file: bool = False,
    output_path: str = None,
    **kwargs
) -> str:
    """투자 평가 실행 함수 (외부 API)"""
    if output_format is None:
        output_format = os.getenv("DEFAULT_OUTPUT_FORMAT", "console")
    
    pipeline = create_pipeline()
    return pipeline.execute_pipeline(
        user_input=user_input,
        output_format=output_format,
        save_to_file=save_to_file,
        output_path=output_path,
        **kwargs
    )