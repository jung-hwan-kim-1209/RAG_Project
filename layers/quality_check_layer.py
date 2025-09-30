"""
Layer 9: QUALITY CHECK LAYER
relevance_checker를 실행하여 답변의 관련성, 근거 충분성, 객관성을 검증하는 레이어
"""
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import os
import re
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    QualityCheckResult, InvestmentReport, AnalysisResult,
    RiskAssessment, DocumentChunk, ExternalSearchResult, PipelineContext, RiskLevel
)
from config import get_config

class RelevanceChecker:
    """관련성 검증기"""

    def __init__(self):
        self.config = get_config()
        self.llm = ChatOpenAI(
            openai_api_key=self.config["model"].openai_api_key,
            temperature=0.1,
            model=self.config["model"].model_name  # 예: "gpt-4o-mini"
        )

        # 관련성 검증 프롬프트
        self.relevance_check_prompt = PromptTemplate(
            input_variables=["company_name", "user_request", "report_content"],
            template="""다음 투자 평가 리포트가 사용자 요청에 얼마나 관련성이 있는지 평가해주세요.

회사명: {company_name}
사용자 요청: {user_request}

생성된 리포트 내용:
{report_content}

다음 기준으로 0-10점 평가해주세요:
1. 요청된 회사에 대한 정확성
2. 요청된 평가 유형과의 일치성
3. 분석 내용의 구체성
4. 결론의 명확성

JSON 형식으로 응답해주세요:
{{
    "relevance_score": 8.5,
    "issues": ["문제점 1", "문제점 2"],
    "strengths": ["강점 1", "강점 2"]
}}"""
        )

    def check_relevance(
        self,
        company_name: str,
        user_request: str,
        report: InvestmentReport
    ) -> float:
        """관련성 점수 계산"""
        try:
            # 리포트 내용 요약
            report_summary = f"""
            Executive Summary: {report.executive_summary[:300]}...
            총점: {report.unicorn_score.total_score}
            추천: {report.recommendation.value}
            """

            response = self.llm.invoke(self.relevance_check_prompt.format(
                company_name=company_name,
                user_request=user_request,
                report_content=report_summary
            ))

            # GPT 응답을 터미널에 출력
            print(f"\n🔍 QUALITY_CHECK_LAYER - GPT 응답:")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

            import json
            result_data = json.loads(response.content.strip())
            return result_data.get("relevance_score", 5.0) / 10.0

        except Exception as e:
            # 기본 관련성 점수 계산
            return self._calculate_basic_relevance(company_name, report)

    def _calculate_basic_relevance(self, company_name: str, report: InvestmentReport) -> float:
        """기본 관련성 계산"""
        score = 0.5  # 기본 점수

        # 회사명 일치 확인
        if company_name.lower() in report.company_info.name.lower():
            score += 0.3

        # 분석 결과 존재 확인
        if report.analysis_results:
            score += 0.1

        # 리스크 평가 존재 확인
        if report.risk_assessments:
            score += 0.1

        return min(score, 1.0)

class EvidenceQualityChecker:
    """근거 품질 검증기"""

    def __init__(self):
        self.config = get_config()
        self.llm = ChatOpenAI(
            openai_api_key=self.config["model"].openai_api_key,
            temperature=0.1,
            model=self.config["model"].model_name  # 예: "gpt-4o-mini"
        )

    def check_evidence_quality(
        self,
        analysis_results: List[AnalysisResult],
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult]
    ) -> float:
        """근거 품질 점수 계산"""
        quality_factors = []

        # 1. 데이터 소스의 다양성
        source_diversity = self._calculate_source_diversity(documents, external_results)
        quality_factors.append(source_diversity)

        # 2. 근거의 구체성
        evidence_specificity = self._calculate_evidence_specificity(analysis_results)
        quality_factors.append(evidence_specificity)

        # 3. 데이터의 최신성
        data_freshness = self._calculate_data_freshness(external_results)
        quality_factors.append(data_freshness)

        # 4. 정량적 근거의 비율
        quantitative_ratio = self._calculate_quantitative_ratio(analysis_results)
        quality_factors.append(quantitative_ratio)

        return sum(quality_factors) / len(quality_factors)

    def _calculate_source_diversity(
        self,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult]
    ) -> float:
        """데이터 소스 다양성 계산"""
        sources = set()

        for doc in documents:
            sources.add(doc.source)

        for result in external_results:
            sources.add(result.source)

        # 소스 개수에 따른 점수 (최대 1.0)
        return min(len(sources) / 10.0, 1.0)

    def _calculate_evidence_specificity(self, analysis_results: List[AnalysisResult]) -> float:
        """근거의 구체성 계산"""
        if not analysis_results:
            return 0.0

        total_evidence_items = 0
        for result in analysis_results:
            total_evidence_items += len(result.supporting_evidence)

        # 분석당 평균 근거 개수 (3개 이상이면 1.0점)
        avg_evidence = total_evidence_items / len(analysis_results)
        return min(avg_evidence / 3.0, 1.0)

    def _calculate_data_freshness(self, external_results: List[ExternalSearchResult]) -> float:
        """데이터 최신성 계산"""
        if not external_results:
            return 0.5

        recent_count = 0
        total_count = len(external_results)

        for result in external_results:
            if result.published_date:
                days_old = (datetime.now() - result.published_date).days
                if days_old <= 30:  # 30일 이내
                    recent_count += 1

        return recent_count / total_count if total_count > 0 else 0.5

    def _calculate_quantitative_ratio(self, analysis_results: List[AnalysisResult]) -> float:
        """정량적 근거 비율 계산"""
        if not analysis_results:
            return 0.0

        quantitative_count = 0
        total_evidence = 0

        for result in analysis_results:
            for evidence in result.supporting_evidence:
                total_evidence += 1
                # 숫자가 포함된 근거인지 확인
                if re.search(r'\d+', evidence):
                    quantitative_count += 1

        return quantitative_count / total_evidence if total_evidence > 0 else 0.0

class ObjectivityChecker:
    """객관성 검증기"""

    def __init__(self):
        self.config = get_config()
        self.llm = ChatOpenAI(
            openai_api_key=self.config["model"].openai_api_key,
            temperature=0.1,
            model=self.config["model"].model_name  # 예: "gpt-4o-mini"
        )

        self.objectivity_prompt = PromptTemplate(
            input_variables=["report_content"],
            template="""다음 투자 평가 리포트의 객관성을 평가해주세요.

리포트 내용:
{report_content}

다음 기준으로 0-10점 평가해주세요:
1. 편향되지 않은 분석
2. 장점과 단점의 균형적 제시
3. 감정적 표현의 배제
4. 근거 기반 결론 도출
5. 다양한 관점의 고려

JSON 형식으로 응답해주세요:
{{
    "objectivity_score": 7.5,
    "bias_indicators": ["편향 지표 1", "편향 지표 2"],
    "improvement_suggestions": ["개선 제안 1", "개선 제안 2"]
}}"""
        )

    def check_objectivity(self, report: InvestmentReport) -> Dict[str, Any]:
        """객관성 점수 계산"""
        try:
            # 리포트 주요 내용 추출
            report_content = f"""
            Executive Summary: {report.executive_summary}
            Investment Rationale: {report.investment_rationale[:500]}
            """

            response = self.llm.invoke(self.objectivity_prompt.format(
                report_content=report_content
            ))

            # GPT 응답을 터미널에 출력
            print(f"\n🔍 QUALITY_CHECK_LAYER (OBJECTIVITY) - GPT 응답:")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

            import json
            objectivity_data = json.loads(response.content.strip())

            return {
                "score": objectivity_data.get("objectivity_score", 5.0) / 10.0,
                "bias_indicators": objectivity_data.get("bias_indicators", []),
                "improvement_suggestions": objectivity_data.get("improvement_suggestions", [])
            }

        except Exception as e:
            return {
                "score": 0.7,  # 기본 객관성 점수
                "bias_indicators": [],
                "improvement_suggestions": [f"객관성 평가 오류: {str(e)}"]
            }

class QualityChecker:
    """품질 검증 메인 클래스"""

    def __init__(self):
        self.relevance_checker = RelevanceChecker()
        self.evidence_checker = EvidenceQualityChecker()
        self.objectivity_checker = ObjectivityChecker()

    def perform_quality_check(
        self,
        report: InvestmentReport,
        original_request: str,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult]
    ) -> QualityCheckResult:
        """종합적인 품질 검증 수행"""

        # 1. 관련성 검증
        relevance_score = self.relevance_checker.check_relevance(
            company_name=report.company_info.name,
            user_request=original_request,
            report=report
        )

        # 2. 근거 품질 검증
        evidence_quality = self.evidence_checker.check_evidence_quality(
            analysis_results=report.analysis_results,
            documents=documents,
            external_results=external_results
        )

        # 3. 객관성 검증
        objectivity_data = self.objectivity_checker.check_objectivity(report)
        objectivity_score = objectivity_data["score"]

        # 4. 전체 품질 점수 계산
        overall_quality = (relevance_score * 0.4 + evidence_quality * 0.3 + objectivity_score * 0.3)

        # 5. 품질 검증 통과 여부 결정
        quality_threshold = float(os.getenv("QUALITY_THRESHOLD", "0.4"))  # 기본값 40% 이상이면 통과
        passed = overall_quality >= quality_threshold

        # 6. 이슈 및 제안사항 수집
        issues = []
        suggestions = []

        relevance_threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.7"))
        evidence_threshold = float(os.getenv("EVIDENCE_THRESHOLD", "0.6"))
        objectivity_threshold = float(os.getenv("OBJECTIVITY_THRESHOLD", "0.7"))

        if relevance_score < relevance_threshold:
            issues.append("관련성 부족")
            suggestions.append("더 구체적인 회사 정보 수집 필요")

        if evidence_quality < evidence_threshold:
            issues.append("근거 품질 부족")
            suggestions.append("더 많은 데이터 소스 활용 필요")

        if objectivity_score < objectivity_threshold:
            issues.append("객관성 부족")
            suggestions.extend(objectivity_data["improvement_suggestions"])

        # 추가 이슈 확인
        additional_issues = self._check_additional_issues(report)
        issues.extend(additional_issues["issues"])
        suggestions.extend(additional_issues["suggestions"])

        return QualityCheckResult(
            relevance_score=relevance_score,
            evidence_quality=evidence_quality,
            objectivity_score=objectivity_score,
            overall_quality=overall_quality,
            passed=passed,
            issues=issues,
            suggestions=suggestions
        )

    def _check_additional_issues(self, report: InvestmentReport) -> Dict[str, List[str]]:
        """추가 품질 이슈 확인"""
        issues = []
        suggestions = []

        # 분석 결과 완성도 확인
        if len(report.analysis_results) < 4:
            issues.append("분석 영역 부족")
            suggestions.append("모든 분석 영역 완료 필요")

        # 리스크 평가 완성도 확인
        if len(report.risk_assessments) < 4:
            issues.append("리스크 평가 부족")
            suggestions.append("주요 리스크 영역 평가 필요")

        # 신뢰도 수준 확인
        if report.confidence_level < 0.5:
            issues.append("낮은 분석 신뢰도")
            suggestions.append("추가 데이터 수집 및 분석 필요")

        # 점수 일관성 확인
        score_issues = self._check_score_consistency(report)
        issues.extend(score_issues["issues"])
        suggestions.extend(score_issues["suggestions"])

        return {"issues": issues, "suggestions": suggestions}

    def _check_score_consistency(self, report: InvestmentReport) -> Dict[str, List[str]]:
        """점수 일관성 확인"""
        issues = []
        suggestions = []

        # 점수와 등급 일치성 확인
        total_score = report.unicorn_score.total_score
        grade = report.unicorn_score.grade

        grade_ranges = {
            "S": (90, 100),
            "A": (80, 89),
            "B": (70, 79),
            "C": (60, 69),
            "D": (0, 59)
        }

        expected_range = grade_ranges.get(grade, (0, 100))
        if not (expected_range[0] <= total_score <= expected_range[1]):
            issues.append("점수와 등급 불일치")
            suggestions.append("점수 계산 로직 재검토 필요")

        # 투자 추천과 점수 일관성 확인
        recommendation = report.recommendation.value
        if recommendation == "투자 추천" and total_score < 70:
            issues.append("낮은 점수 대비 투자 추천 불일치")
            suggestions.append("투자 추천 로직 재검토 필요")

        if recommendation == "회피" and total_score > 80:
            issues.append("높은 점수 대비 투자 회피 불일치")
            suggestions.append("리스크 요인 재평가 필요")

        # 유니콘 확률과 총점 일관성 확인
        unicorn_prob = report.unicorn_score.unicorn_probability
        if total_score > 85 and unicorn_prob < 0.5:
            issues.append("높은 점수 대비 낮은 유니콘 확률")
            suggestions.append("유니콘 확률 계산 로직 재검토")

        return {"issues": issues, "suggestions": suggestions}

class QualityCheckLayer:
    """품질 검증 레이어 메인 클래스"""

    def __init__(self):
        self.quality_checker = QualityChecker()

    def perform_quality_check(
        self,
        report: InvestmentReport,
        original_request: str,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult]
    ) -> QualityCheckResult:
        """품질 검증 실행"""
        return self.quality_checker.perform_quality_check(
            report=report,
            original_request=original_request,
            documents=documents,
            external_results=external_results
        )

    def should_regenerate_report(self, quality_result: QualityCheckResult) -> bool:
        """리포트 재생성 필요 여부 판단"""
        # 기본적으로 passed가 False이면 재생성
        if not quality_result.passed:
            return True

        # 심각한 이슈가 있는 경우 재생성
        critical_issues = [
            "점수와 등급 불일치",
            "낮은 점수 대비 투자 추천 불일치",
            "높은 점수 대비 투자 회피 불일치"
        ]

        for issue in quality_result.issues:
            if issue in critical_issues:
                return True

        return False

    def generate_quality_improvement_recommendations(
        self,
        quality_result: QualityCheckResult
    ) -> List[str]:
        """품질 개선 권장사항 생성"""
        recommendations = []

        if quality_result.relevance_score < 0.7:
            recommendations.append("관련성 개선을 위해 더 구체적인 회사 데이터 수집")

        if quality_result.evidence_quality < 0.6:
            recommendations.append("분석 근거 보강을 위해 추가 데이터 소스 활용")

        if quality_result.objectivity_score < 0.7:
            recommendations.append("객관성 향상을 위해 다양한 관점 고려")

        if quality_result.overall_quality < 0.6:
            recommendations.append("전반적인 분석 품질 향상 필요")

        recommendations.extend(quality_result.suggestions)

        return list(set(recommendations))  # 중복 제거

def create_quality_check_layer() -> QualityCheckLayer:
    """Quality Check Layer 생성자"""
    return QualityCheckLayer()

def process_quality_check_layer(context: PipelineContext, original_request: str) -> PipelineContext:
    """Quality Check Layer 처리 함수"""
    quality_layer = create_quality_check_layer()

    if not context.final_report:
        # 리포트가 없는 경우
        context.quality_check = QualityCheckResult(
            relevance_score=0.0,
            evidence_quality=0.0,
            objectivity_score=0.0,
            overall_quality=0.0,
            passed=False,
            issues=["최종 리포트 생성 실패"],
            suggestions=["파이프라인 전체 재실행 필요"]
        )
        context.processing_steps.append("QUALITY_CHECK_LAYER: 리포트 없음 - 품질 검증 실패")
        return context

    # 품질 검증 실행
    quality_result = quality_layer.perform_quality_check(
        report=context.final_report,
        original_request=original_request,
        documents=context.retrieved_documents,
        external_results=context.external_search_results
    )

    context.quality_check = quality_result

    # 재생성 필요 여부 확인
    needs_regeneration = quality_layer.should_regenerate_report(quality_result)

    # 처리 단계 기록
    status = "통과" if quality_result.passed else "실패"
    regeneration_note = " (재생성 필요)" if needs_regeneration else ""

    context.processing_steps.append(
        f"QUALITY_CHECK_LAYER: 품질 검증 {status} "
        f"(관련성: {quality_result.relevance_score:.1%}, "
        f"근거: {quality_result.evidence_quality:.1%}, "
        f"객관성: {quality_result.objectivity_score:.1%}){regeneration_note}"
    )

    return context