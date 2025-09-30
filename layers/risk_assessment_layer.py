"""Layer 7: RISK ASSESSMENT LAYER
각 리스크 유형에 대한 평가를 수행한다.
프롬프트와 파싱 로직을 정리해 향후 그래프 노드 전환 시 재사용을 용이하게 한다.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

from config import get_config
from models import (
    AnalysisResult,
    CompanyInfo,
    DocumentChunk,
    ExternalSearchResult,
    PipelineContext,
    RiskAssessment,
    RiskLevel,
)

# 리스크 유형별 정의 (프롬프트 요소 및 기본값)
RISK_DEFINITIONS: Dict[str, Dict[str, object]] = {
    "market_risk": {
        "title": "시장",
        "factors": [
            "시장 포화도 및 성장 한계",
            "경기 변동 민감도",
            "소비자 선호도 변화",
            "신기술 등장 영향",
            "글로벌 시장 진입 장벽",
        ],
        "defaults": {"impact_score": 7.0, "probability": 0.5},
    },
    "regulatory_risk": {
        "title": "규제",
        "factors": [
            "현재 규제 위반 가능성",
            "향후 규제 강화 리스크",
            "국가별 규제 차이",
            "데이터/개인정보 보호 요구",
            "업종 특화 규제 준수 비용",
        ],
        "defaults": {"impact_score": 6.0, "probability": 0.4},
    },
    "competitive_risk": {
        "title": "경쟁",
        "factors": [
            "기존 대기업의 시장 재진입",
            "신규 경쟁자 등장",
            "대체재 등장",
            "가격 경쟁 심화",
            "핵심 인재 유출",
        ],
        "defaults": {"impact_score": 7.5, "probability": 0.6},
    },
    "financial_risk": {
        "title": "재무",
        "factors": [
            "현금 소진 속도와 자금 여력",
            "추가 투자 유치 실패 가능성",
            "수익성 개선 지연",
            "고정비 증가",
            "금리·환율 변동 노출",
        ],
        "defaults": {"impact_score": 6.5, "probability": 0.5},
    },
    "technology_risk": {
        "title": "기술",
        "factors": [
            "핵심 기술 노후화",
            "보안 취약점 및 사이버 공격",
            "기술 인재 이탈",
            "외부 플랫폼 의존성",
            "확장 시 기술 병목",
        ],
        "defaults": {"impact_score": 6.0, "probability": 0.4},
    },
    "team_risk": {
        "title": "팀",
        "factors": [
            "창업자·핵심 인재 이탈",
            "조직 문화 및 커뮤니케이션",
            "핵심 직군 확보 어려움",
            "의사결정 구조 미비",
            "조직 확장 대응 능력",
        ],
        "defaults": {"impact_score": 6.5, "probability": 0.3},
    },
}


class BaseRiskEvaluator:
    """리스크 평가기 기본 클래스"""

    def __init__(self, risk_category: str, definition: Dict[str, object]):
        self.risk_category = risk_category
        self.definition = definition
        self.config = get_config()
        self.llm = OpenAI(
            temperature=0.1,
            model_name="gpt-3.5-turbo-instruct",
        )
        self.evaluation_prompt = self._build_prompt()

    def evaluate(
        self,
        company_info: CompanyInfo,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        analysis_results: List[AnalysisResult],
    ) -> RiskAssessment:
        raise NotImplementedError

    def _build_prompt(self) -> PromptTemplate:
        title = self.definition.get("title", "해당")
        factors = self.definition.get("factors", [])
        defaults = self.definition.get("defaults", {})
        impact_default = defaults.get("impact_score", 5.0)
        probability_default = defaults.get("probability", 0.5)

        factors_text = "\n".join(
            f"{index}. {factor}" for index, factor in enumerate(factors, start=1)
        ) or "- 참고 요소 없음"

        template = f"""다음 정보를 바탕으로 {{company_name}}의 {title} 리스크를 평가해주세요.

회사명: {{company_name}}
업종: {{industry}}

관련 정보 요약:
{{context}}

아래 항목을 참고하여 정량·정성 평가를 수행하세요:
{factors_text}

반드시 JSON 형식으로 응답하세요.
{{
    "impact_score": {impact_default},
    "probability": {probability_default},
    "description": "위험 설명 (2-3문장)",
    "mitigation_strategies": ["완화 전략 1", "완화 전략 2"]
}}"""

        return PromptTemplate(
            input_variables=["company_name", "industry", "context"],
            template=template,
        )

    def _create_analysis_context(
        self,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        analysis_results: List[AnalysisResult],
    ) -> str:
        context_parts: List[str] = []

        if documents:
            doc_summaries = []
            for doc in documents[:3]:
                doc_summaries.append(f"- {doc.content[:150]}...")
            context_parts.append("관련 문서:\n" + "\n".join(doc_summaries))

        if external_results:
            external_summaries = []
            for result in external_results[:2]:
                external_summaries.append(f"- {result.title}: {result.content[:100]}...")
            context_parts.append("최신 외부 정보:\n" + "\n".join(external_summaries))

        if analysis_results:
            analysis_summaries = []
            for result in analysis_results:
                analysis_summaries.append(
                    f"- {result.category}: {result.score}점 ({result.grade})"
                )
            context_parts.append("분석 결과:\n" + "\n".join(analysis_summaries))

        return "\n\n".join(context_parts)

    def _call_model(self, company_info: CompanyInfo, context: str) -> Dict[str, object]:
        prompt = self.evaluation_prompt.format(
            company_name=company_info.name,
            industry=company_info.industry or "업종 정보 없음",
            context=context or "관련 데이터를 확인할 수 없습니다. 기존 분석과 문서를 기반으로 합리적 추정을 제시하세요.",
        )
        raw = self.llm(prompt)
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(response_text: str) -> Dict[str, object]:
        try:
            cleaned = response_text.strip()
            start = cleaned.find("{")
            end = cleaned.rfind("}") + 1
            if start == -1 or end <= start:
                raise ValueError("JSON boundary not found")
            return json.loads(cleaned[start:end])
        except Exception:
            return {}

    @staticmethod
    def _coerce_float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _coerce_text(value, default: str) -> str:
        if isinstance(value, str) and value.strip():
            return value.strip()
        return default


class ConfigurableRiskEvaluator(BaseRiskEvaluator):
    """RISK_DEFINITIONS 기반으로 동작하는 리스크 평가기"""

    def __init__(self, risk_key: str, definition: Dict[str, object]):
        super().__init__(risk_key, definition)

    def evaluate(
        self,
        company_info: CompanyInfo,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        analysis_results: List[AnalysisResult],
    ) -> RiskAssessment:
        context = self._create_analysis_context(documents, external_results, analysis_results)
        result = self._call_model(company_info, context)
        defaults = self.definition.get("defaults", {})

        impact_score = self._coerce_float(result.get("impact_score"), defaults.get("impact_score", 5.0))
        probability = self._coerce_float(result.get("probability"), defaults.get("probability", 0.5))
        description = self._coerce_text(result.get("description"), "리스크 설명 부족")

        mitigation_raw = result.get("mitigation_strategies", [])
        if not isinstance(mitigation_raw, list):
            mitigation_raw = []
        mitigation = [self._coerce_text(item, "") for item in mitigation_raw if item]

        return RiskAssessment(
            category=self.risk_category,
            risk_level=self._calculate_risk_level(impact_score, probability),
            description=description,
            impact_score=impact_score,
            probability=probability,
            mitigation_strategies=mitigation,
        )

    def _calculate_risk_level(self, impact_score: float, probability: float) -> RiskLevel:
        risk_score = impact_score * probability
        if risk_score >= 8.0:
            return RiskLevel.CRITICAL
        if risk_score >= 6.0:
            return RiskLevel.HIGH
        if risk_score >= 4.0:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


class RiskEvaluator:
    """리스크 평가 메인 클래스"""

    def __init__(self):
        self.risk_evaluators = {
            key: ConfigurableRiskEvaluator(key, definition)
            for key, definition in RISK_DEFINITIONS.items()
        }

    def evaluate_all_risks(
        self,
        company_info: CompanyInfo,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        analysis_results: List[AnalysisResult],
        selected_risks: List[str] = None,
    ) -> List[RiskAssessment]:
        if selected_risks is None:
            selected_risks = list(self.risk_evaluators.keys())

        selected = {
            name: evaluator
            for name, evaluator in self.risk_evaluators.items()
            if name in selected_risks
        }

        with ThreadPoolExecutor(max_workers=len(selected)) as executor:
            future_to_name = {
                executor.submit(
                    evaluator.evaluate,
                    company_info,
                    documents,
                    external_results,
                    analysis_results,
                ): name
                for name, evaluator in selected.items()
            }

            results: List[RiskAssessment] = []
            for future in future_to_name:
                name = future_to_name[future]
                try:
                    results.append(future.result(timeout=60))
                except Exception as err:  # pragma: no cover (LLM 오류 대비)
                    results.append(
                        RiskAssessment(
                            category=name,
                            risk_level=RiskLevel.MEDIUM,
                            description=f"{name} 평가 실패: {err}",
                            impact_score=5.0,
                            probability=0.5,
                            mitigation_strategies=[],
                        )
                    )

        return results

    def calculate_overall_risk_level(self, risk_assessments: List[RiskAssessment]) -> RiskLevel:
        if not risk_assessments:
            return RiskLevel.MEDIUM

        level_map = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 2.0,
            RiskLevel.HIGH: 3.0,
            RiskLevel.CRITICAL: 4.0,
        }

        weighted_sum = 0.0
        total_weight = 0.0
        for assessment in risk_assessments:
            weight = assessment.impact_score * assessment.probability
            weighted_sum += level_map.get(assessment.risk_level, 2.0) * weight
            total_weight += weight

        if total_weight == 0:
            return RiskLevel.MEDIUM

        average = weighted_sum / total_weight
        if average >= 3.5:
            return RiskLevel.CRITICAL
        if average >= 2.5:
            return RiskLevel.HIGH
        if average >= 1.5:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW


def create_risk_assessment_layer() -> RiskEvaluator:
    return RiskEvaluator()


def process_risk_assessment_layer(context: PipelineContext) -> PipelineContext:
    evaluator = create_risk_assessment_layer()
    risk_assessments = evaluator.evaluate_all_risks(
        company_info=context.company_info,
        documents=context.retrieved_documents,
        external_results=context.external_search_results,
        analysis_results=context.analysis_results,
    )

    context.risk_assessments = risk_assessments
    overall = evaluator.calculate_overall_risk_level(risk_assessments)
    context.processing_steps.append(
        "RISK_ASSESSMENT_LAYER: "
        f"{len(risk_assessments)}개 리스크 평가 완료, 전체 리스크 {overall.value}"
    )
    return context
