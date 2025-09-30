"""
Layer 7: RISK ASSESSMENT LAYER
risk_evaluator를 실행하여 시장, 규제, 경쟁, 재무 리스크를 평가하는 레이어
"""
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    RiskAssessment, RiskLevel, DocumentChunk, ExternalSearchResult,
    PipelineContext, CompanyInfo, AnalysisResult
)
from config import get_config

class BaseRiskEvaluator:
    """리스크 평가기 기본 클래스"""

    def __init__(self, risk_category: str):
        self.risk_category = risk_category
        self.config = get_config()
        self.llm = ChatOpenAI(
            openai_api_key=self.config["model"].openai_api_key,
            temperature=0.1,
            model=self.config["model"].model_name  # 예: "gpt-4o-mini"
        )

    def evaluate(
        self,
        company_info: CompanyInfo,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        analysis_results: List[AnalysisResult]
    ) -> RiskAssessment:
        """리스크 평가 실행 (하위 클래스에서 구현)"""
        raise NotImplementedError

    def _calculate_risk_level(self, impact_score: float, probability: float) -> RiskLevel:
        """리스크 레벨 계산"""
        risk_score = impact_score * probability

        if risk_score >= 8.0:
            return RiskLevel.CRITICAL
        elif risk_score >= 6.0:
            return RiskLevel.HIGH
        elif risk_score >= 4.0:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _create_analysis_context(
        self,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        analysis_results: List[AnalysisResult]
    ) -> str:
        """분석 컨텍스트 생성"""
        context_parts = []

        # 관련 문서 정보
        if documents:
            doc_summaries = []
            for doc in documents[:3]:
                doc_summaries.append(f"- {doc.content[:150]}...")
            context_parts.append("관련 문서:\n" + "\n".join(doc_summaries))

        # 외부 검색 결과
        if external_results:
            external_summaries = []
            for result in external_results[:2]:
                external_summaries.append(f"- {result.title}: {result.content[:100]}...")
            context_parts.append("최신 정보:\n" + "\n".join(external_summaries))

        # 분석 결과 요약
        if analysis_results:
            analysis_summaries = []
            for result in analysis_results:
                analysis_summaries.append(f"- {result.category}: {result.score}점 ({result.grade})")
            context_parts.append("분석 결과:\n" + "\n".join(analysis_summaries))

        return "\n\n".join(context_parts)

class MarketRiskEvaluator(BaseRiskEvaluator):
    """시장 리스크 평가기"""

    def __init__(self):
        super().__init__("market_risk")
        self.evaluation_prompt = PromptTemplate(
            input_variables=["company_name", "industry", "context"],
            template="""다음 정보를 바탕으로 {company_name}의 시장 리스크를 평가해주세요.

회사명: {company_name}
업종: {industry}

관련 정보:
{context}

다음 시장 리스크 요소들을 평가해주세요:
1. 시장 포화도 및 성장 한계
2. 경기 변동에 대한 민감도
3. 소비자 선호도 변화 리스크
4. 신기술 등장으로 인한 시장 변화
5. 글로벌 시장 진출 시 장벽

평가 기준:
- 영향도(impact_score): 0-10점 (10이 가장 심각)
- 발생 확률(probability): 0-1 (1이 확실)

JSON 형식으로 응답해주세요:
{{
    "impact_score": 7.5,
    "probability": 0.6,
    "description": "시장 리스크 설명",
    "mitigation_strategies": ["완화 전략 1", "완화 전략 2"]
}}"""
        )

    def evaluate(
        self,
        company_info: CompanyInfo,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        analysis_results: List[AnalysisResult]
    ) -> RiskAssessment:
        """시장 리스크 평가 실행"""
        context = self._create_analysis_context(documents, external_results, analysis_results)

        try:
            response = self.llm.invoke(self.evaluation_prompt.format(
                company_name=company_info.name,
                industry=company_info.industry,
                context=context
            ))

            # GPT 응답을 터미널에 출력
            print(f"\n🔍 RISK_ASSESSMENT_LAYER ({self.risk_category.upper()}) - GPT 응답:")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

            import json
            risk_data = json.loads(response.content.strip())

            impact_score = risk_data.get("impact_score", 5.0)
            probability = risk_data.get("probability", 0.5)

            return RiskAssessment(
                category=self.risk_category,
                risk_level=self._calculate_risk_level(impact_score, probability),
                description=risk_data.get("description", ""),
                impact_score=impact_score,
                probability=probability,
                mitigation_strategies=risk_data.get("mitigation_strategies", [])
            )

        except Exception as e:
            return RiskAssessment(
                category=self.risk_category,
                risk_level=RiskLevel.MEDIUM,
                description=f"시장 리스크 평가 오류: {str(e)}",
                impact_score=5.0,
                probability=0.5,
                mitigation_strategies=[]
            )

class RegulatoryRiskEvaluator(BaseRiskEvaluator):
    """규제 리스크 평가기"""

    def __init__(self):
        super().__init__("regulatory_risk")
        self.evaluation_prompt = PromptTemplate(
            input_variables=["company_name", "industry", "context"],
            template="""다음 정보를 바탕으로 {company_name}의 규제 리스크를 평가해주세요.

회사명: {company_name}
업종: {industry}

관련 정보:
{context}

다음 규제 리스크 요소들을 평가해주세요:
1. 현재 규제 위반 가능성
2. 미래 규제 강화 리스크
3. 국가별 규제 차이로 인한 확장 제약
4. 개인정보보호 및 데이터 관련 규제
5. 업종별 특화 규제 요구사항

JSON 형식으로 응답해주세요:
{{
    "impact_score": 6.0,
    "probability": 0.4,
    "description": "규제 리스크 설명",
    "mitigation_strategies": ["완화 전략 1", "완화 전략 2"]
}}"""
        )

    def evaluate(
        self,
        company_info: CompanyInfo,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        analysis_results: List[AnalysisResult]
    ) -> RiskAssessment:
        """규제 리스크 평가 실행"""
        context = self._create_analysis_context(documents, external_results, analysis_results)

        try:
            response = self.llm.invoke(self.evaluation_prompt.format(
                company_name=company_info.name,
                industry=company_info.industry,
                context=context
            ))

            # GPT 응답을 터미널에 출력
            print(f"\n🔍 RISK_ASSESSMENT_LAYER ({self.risk_category.upper()}) - GPT 응답:")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

            import json
            risk_data = json.loads(response.content.strip())

            impact_score = risk_data.get("impact_score", 5.0)
            probability = risk_data.get("probability", 0.5)

            return RiskAssessment(
                category=self.risk_category,
                risk_level=self._calculate_risk_level(impact_score, probability),
                description=risk_data.get("description", ""),
                impact_score=impact_score,
                probability=probability,
                mitigation_strategies=risk_data.get("mitigation_strategies", [])
            )

        except Exception as e:
            return RiskAssessment(
                category=self.risk_category,
                risk_level=RiskLevel.MEDIUM,
                description=f"규제 리스크 평가 오류: {str(e)}",
                impact_score=5.0,
                probability=0.5,
                mitigation_strategies=[]
            )

class CompetitiveRiskEvaluator(BaseRiskEvaluator):
    """경쟁 리스크 평가기"""

    def __init__(self):
        super().__init__("competitive_risk")
        self.evaluation_prompt = PromptTemplate(
            input_variables=["company_name", "industry", "context"],
            template="""다음 정보를 바탕으로 {company_name}의 경쟁 리스크를 평가해주세요.

회사명: {company_name}
업종: {industry}

관련 정보:
{context}

다음 경쟁 리스크 요소들을 평가해주세요:
1. 기존 대기업의 시장 진입 위험
2. 신규 경쟁자의 등장 가능성
3. 대체재 출현 리스크
4. 가격 경쟁 심화 위험
5. 핵심 인재 유출 리스크

JSON 형식으로 응답해주세요:
{{
    "impact_score": 8.0,
    "probability": 0.7,
    "description": "경쟁 리스크 설명",
    "mitigation_strategies": ["완화 전략 1", "완화 전략 2"]
}}"""
        )

    def evaluate(
        self,
        company_info: CompanyInfo,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        analysis_results: List[AnalysisResult]
    ) -> RiskAssessment:
        """경쟁 리스크 평가 실행"""
        context = self._create_analysis_context(documents, external_results, analysis_results)

        try:
            response = self.llm.invoke(self.evaluation_prompt.format(
                company_name=company_info.name,
                industry=company_info.industry,
                context=context
            ))

            # GPT 응답을 터미널에 출력
            print(f"\n🔍 RISK_ASSESSMENT_LAYER ({self.risk_category.upper()}) - GPT 응답:")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

            import json
            risk_data = json.loads(response.content.strip())

            impact_score = risk_data.get("impact_score", 5.0)
            probability = risk_data.get("probability", 0.5)

            return RiskAssessment(
                category=self.risk_category,
                risk_level=self._calculate_risk_level(impact_score, probability),
                description=risk_data.get("description", ""),
                impact_score=impact_score,
                probability=probability,
                mitigation_strategies=risk_data.get("mitigation_strategies", [])
            )

        except Exception as e:
            return RiskAssessment(
                category=self.risk_category,
                risk_level=RiskLevel.MEDIUM,
                description=f"경쟁 리스크 평가 오류: {str(e)}",
                impact_score=5.0,
                probability=0.5,
                mitigation_strategies=[]
            )

class FinancialRiskEvaluator(BaseRiskEvaluator):
    """재무 리스크 평가기"""

    def __init__(self):
        super().__init__("financial_risk")
        self.evaluation_prompt = PromptTemplate(
            input_variables=["company_name", "industry", "context"],
            template="""다음 정보를 바탕으로 {company_name}의 재무 리스크를 평가해주세요.

회사명: {company_name}
업종: {industry}

관련 정보:
{context}

다음 재무 리스크 요소들을 평가해주세요:
1. 현금 소진 위험 (burn rate vs runway)
2. 추가 투자 유치 실패 리스크
3. 수익성 달성 지연 위험
4. 고정비 부담 증가 리스크
5. 환율 및 이자율 변동 리스크

JSON 형식으로 응답해주세요:
{{
    "impact_score": 9.0,
    "probability": 0.5,
    "description": "재무 리스크 설명",
    "mitigation_strategies": ["완화 전략 1", "완화 전략 2"]
}}"""
        )

    def evaluate(
        self,
        company_info: CompanyInfo,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        analysis_results: List[AnalysisResult]
    ) -> RiskAssessment:
        """재무 리스크 평가 실행"""
        context = self._create_analysis_context(documents, external_results, analysis_results)

        try:
            response = self.llm.invoke(self.evaluation_prompt.format(
                company_name=company_info.name,
                industry=company_info.industry,
                context=context
            ))

            # GPT 응답을 터미널에 출력
            print(f"\n🔍 RISK_ASSESSMENT_LAYER ({self.risk_category.upper()}) - GPT 응답:")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

            import json
            risk_data = json.loads(response.content.strip())

            impact_score = risk_data.get("impact_score", 5.0)
            probability = risk_data.get("probability", 0.5)

            return RiskAssessment(
                category=self.risk_category,
                risk_level=self._calculate_risk_level(impact_score, probability),
                description=risk_data.get("description", ""),
                impact_score=impact_score,
                probability=probability,
                mitigation_strategies=risk_data.get("mitigation_strategies", [])
            )

        except Exception as e:
            return RiskAssessment(
                category=self.risk_category,
                risk_level=RiskLevel.MEDIUM,
                description=f"재무 리스크 평가 오류: {str(e)}",
                impact_score=5.0,
                probability=0.5,
                mitigation_strategies=[]
            )

class TechnologyRiskEvaluator(BaseRiskEvaluator):
    """기술 리스크 평가기"""

    def __init__(self):
        super().__init__("technology_risk")
        self.evaluation_prompt = PromptTemplate(
            input_variables=["company_name", "industry", "context"],
            template="""다음 정보를 바탕으로 {company_name}의 기술 리스크를 평가해주세요.

회사명: {company_name}
업종: {industry}

관련 정보:
{context}

다음 기술 리스크 요소들을 평가해주세요:
1. 핵심 기술의 노후화 위험
2. 보안 취약점 및 사이버 공격 리스크
3. 기술 인재 이탈 위험
4. 플랫폼 의존성 리스크
5. 스케일링 시 기술적 한계

JSON 형식으로 응답해주세요:
{{
    "impact_score": 6.5,
    "probability": 0.4,
    "description": "기술 리스크 설명",
    "mitigation_strategies": ["완화 전략 1", "완화 전략 2"]
}}"""
        )

    def evaluate(
        self,
        company_info: CompanyInfo,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        analysis_results: List[AnalysisResult]
    ) -> RiskAssessment:
        """기술 리스크 평가 실행"""
        context = self._create_analysis_context(documents, external_results, analysis_results)

        try:
            response = self.llm.invoke(self.evaluation_prompt.format(
                company_name=company_info.name,
                industry=company_info.industry,
                context=context
            ))

            # GPT 응답을 터미널에 출력
            print(f"\n🔍 RISK_ASSESSMENT_LAYER ({self.risk_category.upper()}) - GPT 응답:")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

            import json
            risk_data = json.loads(response.content.strip())

            impact_score = risk_data.get("impact_score", 5.0)
            probability = risk_data.get("probability", 0.5)

            return RiskAssessment(
                category=self.risk_category,
                risk_level=self._calculate_risk_level(impact_score, probability),
                description=risk_data.get("description", ""),
                impact_score=impact_score,
                probability=probability,
                mitigation_strategies=risk_data.get("mitigation_strategies", [])
            )

        except Exception as e:
            return RiskAssessment(
                category=self.risk_category,
                risk_level=RiskLevel.MEDIUM,
                description=f"기술 리스크 평가 오류: {str(e)}",
                impact_score=5.0,
                probability=0.5,
                mitigation_strategies=[]
            )

class TeamRiskEvaluator(BaseRiskEvaluator):
    """팀 리스크 평가기"""

    def __init__(self):
        super().__init__("team_risk")
        self.evaluation_prompt = PromptTemplate(
            input_variables=["company_name", "industry", "context"],
            template="""다음 정보를 바탕으로 {company_name}의 팀 관련 리스크를 평가해주세요.

회사명: {company_name}
업종: {industry}

관련 정보:
{context}

다음 팀 리스크 요소들을 평가해주세요:
1. 창업자/핵심 인재 이탈 위험
2. 팀 내부 갈등 및 분열 가능성
3. 핵심 기술자 확보 어려움
4. 조직 문화 및 관리 체계 미비
5. 성장에 따른 인재 관리 어려움

JSON 형식으로 응답해주세요:
{{
    "impact_score": 8.5,
    "probability": 0.3,
    "description": "팀 리스크 설명",
    "mitigation_strategies": ["완화 전략 1", "완화 전략 2"]
}}"""
        )

    def evaluate(
        self,
        company_info: CompanyInfo,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        analysis_results: List[AnalysisResult]
    ) -> RiskAssessment:
        """팀 리스크 평가 실행"""
        context = self._create_analysis_context(documents, external_results, analysis_results)

        try:
            response = self.llm.invoke(self.evaluation_prompt.format(
                company_name=company_info.name,
                industry=company_info.industry,
                context=context
            ))

            # GPT 응답을 터미널에 출력
            print(f"\n🔍 RISK_ASSESSMENT_LAYER ({self.risk_category.upper()}) - GPT 응답:")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

            import json
            risk_data = json.loads(response.content.strip())

            impact_score = risk_data.get("impact_score", 5.0)
            probability = risk_data.get("probability", 0.5)

            return RiskAssessment(
                category=self.risk_category,
                risk_level=self._calculate_risk_level(impact_score, probability),
                description=risk_data.get("description", ""),
                impact_score=impact_score,
                probability=probability,
                mitigation_strategies=risk_data.get("mitigation_strategies", [])
            )

        except Exception as e:
            return RiskAssessment(
                category=self.risk_category,
                risk_level=RiskLevel.MEDIUM,
                description=f"팀 리스크 평가 오류: {str(e)}",
                impact_score=5.0,
                probability=0.5,
                mitigation_strategies=[]
            )

class RiskEvaluator:
    """리스크 평가 메인 클래스"""

    def __init__(self):
        self.risk_evaluators = {
            "market_risk": MarketRiskEvaluator(),
            "regulatory_risk": RegulatoryRiskEvaluator(),
            "competitive_risk": CompetitiveRiskEvaluator(),
            "financial_risk": FinancialRiskEvaluator(),
            "technology_risk": TechnologyRiskEvaluator(),
            "team_risk": TeamRiskEvaluator()
        }

    def evaluate_all_risks(
        self,
        company_info: CompanyInfo,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        analysis_results: List[AnalysisResult],
        selected_risks: List[str] = None
    ) -> List[RiskAssessment]:
        """모든 리스크 평가 실행 (병렬)"""

        if selected_risks is None:
            selected_risks = list(self.risk_evaluators.keys())

        # 선택된 리스크 평가기들만 실행
        selected_evaluators = {
            name: evaluator for name, evaluator in self.risk_evaluators.items()
            if name in selected_risks
        }

        # ThreadPoolExecutor를 사용한 병렬 실행
        with ThreadPoolExecutor(max_workers=len(selected_evaluators)) as executor:
            future_to_evaluator = {
                executor.submit(
                    evaluator.evaluate,
                    company_info, documents, external_results, analysis_results
                ): name
                for name, evaluator in selected_evaluators.items()
            }

            risk_assessments = []
            for future in future_to_evaluator:
                try:
                    result = future.result(timeout=60)  # 60초 타임아웃
                    risk_assessments.append(result)
                except Exception as e:
                    evaluator_name = future_to_evaluator[future]
                    error_assessment = RiskAssessment(
                        category=evaluator_name,
                        risk_level=RiskLevel.MEDIUM,
                        description=f"{evaluator_name} 평가 실패: {str(e)}",
                        impact_score=5.0,
                        probability=0.5,
                        mitigation_strategies=[]
                    )
                    risk_assessments.append(error_assessment)

        return risk_assessments

    def calculate_overall_risk_level(self, risk_assessments: List[RiskAssessment]) -> RiskLevel:
        """전체 리스크 레벨 계산"""
        if not risk_assessments:
            return RiskLevel.MEDIUM

        # 리스크 레벨을 숫자로 변환
        risk_level_values = {
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4
        }

        # 가중 평균 계산 (영향도와 확률 고려)
        total_weighted_risk = 0.0
        total_weight = 0.0

        for assessment in risk_assessments:
            risk_value = risk_level_values[assessment.risk_level]
            weight = assessment.impact_score * assessment.probability
            total_weighted_risk += risk_value * weight
            total_weight += weight

        if total_weight == 0:
            return RiskLevel.MEDIUM

        average_risk = total_weighted_risk / total_weight

        # 평균값을 리스크 레벨로 변환
        if average_risk >= 3.5:
            return RiskLevel.CRITICAL
        elif average_risk >= 2.5:
            return RiskLevel.HIGH
        elif average_risk >= 1.5:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

class CompetitiveRiskEvaluator(BaseRiskEvaluator):
    """경쟁 리스크 평가기 (누락된 클래스 추가)"""

    def __init__(self):
        super().__init__("competitive_risk")
        self.evaluation_prompt = PromptTemplate(
            input_variables=["company_name", "industry", "context"],
            template="""다음 정보를 바탕으로 {company_name}의 경쟁 리스크를 평가해주세요.

회사명: {company_name}
업종: {industry}

관련 정보:
{context}

다음 경쟁 리스크 요소들을 평가해주세요:
1. 기존 대기업의 시장 진입 위험
2. 신규 경쟁자의 등장 가능성
3. 대체재 출현 리스크
4. 가격 경쟁 심화 위험
5. 시장 점유율 감소 위험

JSON 형식으로 응답해주세요:
{{
    "impact_score": 7.0,
    "probability": 0.6,
    "description": "경쟁 리스크 설명",
    "mitigation_strategies": ["완화 전략 1", "완화 전략 2"]
}}"""
        )

    def evaluate(
        self,
        company_info: CompanyInfo,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        analysis_results: List[AnalysisResult]
    ) -> RiskAssessment:
        """경쟁 리스크 평가 실행"""
        context = self._create_analysis_context(documents, external_results, analysis_results)

        try:
            response = self.llm.invoke(self.evaluation_prompt.format(
                company_name=company_info.name,
                industry=company_info.industry,
                context=context
            ))

            # GPT 응답을 터미널에 출력
            print(f"\n🔍 RISK_ASSESSMENT_LAYER ({self.risk_category.upper()}) - GPT 응답:")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

            import json
            risk_data = json.loads(response.content.strip())

            impact_score = risk_data.get("impact_score", 5.0)
            probability = risk_data.get("probability", 0.5)

            return RiskAssessment(
                category=self.risk_category,
                risk_level=self._calculate_risk_level(impact_score, probability),
                description=risk_data.get("description", ""),
                impact_score=impact_score,
                probability=probability,
                mitigation_strategies=risk_data.get("mitigation_strategies", [])
            )

        except Exception as e:
            return RiskAssessment(
                category=self.risk_category,
                risk_level=RiskLevel.MEDIUM,
                description=f"경쟁 리스크 평가 오류: {str(e)}",
                impact_score=5.0,
                probability=0.5,
                mitigation_strategies=[]
            )

def create_risk_assessment_layer() -> RiskEvaluator:
    """Risk Assessment Layer 생성자"""
    return RiskEvaluator()

def process_risk_assessment_layer(context: PipelineContext) -> PipelineContext:
    """Risk Assessment Layer 처리 함수"""
    risk_evaluator = create_risk_assessment_layer()

    # 리스크 평가 실행
    risk_assessments = risk_evaluator.evaluate_all_risks(
        company_info=context.company_info,
        documents=context.retrieved_documents,
        external_results=context.external_search_results,
        analysis_results=context.analysis_results
    )

    context.risk_assessments = risk_assessments

    # 전체 리스크 레벨 계산
    overall_risk_level = risk_evaluator.calculate_overall_risk_level(risk_assessments)

    # 처리 단계 기록
    context.processing_steps.append(
        f"RISK_ASSESSMENT_LAYER: {len(risk_assessments)}개 리스크 평가 완료, "
        f"전체 리스크: {overall_risk_level.value}"
    )

    return context