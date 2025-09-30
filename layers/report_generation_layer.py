"""
Layer 8: REPORT GENERATION LAYER
unicorn_report_generator를 실행하여 최종 투자 평가 리포트를 생성하는 레이어
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser # 출력 파서 추가


from models import (
    InvestmentReport, InvestmentRecommendation, UnicornScore,
    AnalysisResult, RiskAssessment, CompanyInfo, PipelineContext, RiskLevel,
    DocumentChunk, ExternalSearchResult
)
from config import get_config


class UnicornReportGenerator:
    """유니콘 투자 평가 리포트 생성기"""


    def __init__(self):
        self.config = get_config()
        # ChatOpenAI 모델과 StrOutputParser를 파이프로 연결하여, 결과가 항상 문자열(str)로 나오도록 체인을 구성합니다.
        self.llm_chain = (
            ChatOpenAI(
                openai_api_key=self.config["model"].openai_api_key,
                temperature=self.config["model"].temperature,
                model_name=self.config["model"].model_name
            )
            | StrOutputParser()
        )


        # Executive Summary 생성 프롬프트
        self.executive_summary_prompt = PromptTemplate(
            input_variables=["company_name", "total_score", "grade", "unicorn_probability", "recommendation"],
            template="""다음 정보를 바탕으로 {company_name}에 대한 투자 평가 Executive Summary를 작성해주세요.


회사명: {company_name}
종합 점수: {total_score}/100
등급: {grade}
유니콘 확률: {unicorn_probability:.1%}
투자 추천: {recommendation}


Executive Summary는 다음 구조로 작성해주세요:
1. 핵심 결론 (2-3줄)
2. 주요 강점 (3-4개 항목)
3. 주요 우려사항 (2-3개 항목)
4. 투자 권장사항 (1-2줄)


전문적이고 간결한 투자 보고서 형식으로 작성해주세요."""
        )


        # 상세 분석 요약 프롬프트
        self.detailed_analysis_prompt = PromptTemplate(
            input_variables=["company_name", "analysis_results", "risk_assessments"],
            template="""다음 분석 결과들을 바탕으로 {company_name}에 대한 상세 분석 보고서를 작성해주세요.


분석 결과:
{analysis_results}


리스크 평가:
{risk_assessments}


다음 섹션별로 구성해주세요:
1. 성장성 분석
2. 비즈니스 모델 평가
3. 기술력 및 보안성
4. 재무 건전성
5. 팀 역량 평가
6. 규제 환경 분석
7. 파트너십 및 네트워크


각 섹션은 점수, 주요 발견사항, 개선 권장사항을 포함해주세요."""
        )


        # 투자 근거 프롬프트
        self.investment_rationale_prompt = PromptTemplate(
            input_variables=["company_name", "recommendation", "unicorn_score", "key_factors"],
            template="""다음 정보를 바탕으로 {company_name}에 대한 투자 근거를 작성해주세요.


투자 추천: {recommendation}
유니콘 점수: {unicorn_score}
주요 요인들: {key_factors}


투자 근거는 다음을 포함해주세요:
1. 투자 결정의 핵심 논리
2. 예상 수익률 및 리스크 균형
3. 투자 타이밍의 적절성
4. 포트폴리오 내 포지셔닝
5. Exit 전략 고려사항


투자자 관점에서 명확하고 설득력 있게 작성해주세요."""
        )


    def determine_investment_recommendation(
        self,
        unicorn_score: UnicornScore,
        risk_assessments: List[RiskAssessment]
    ) -> InvestmentRecommendation:
        """투자 추천 결정"""


        total_score = unicorn_score.total_score
        unicorn_probability = unicorn_score.unicorn_probability


        # 높은 리스크 체크
        high_risk_count = sum(1 for risk in risk_assessments
                              if risk.risk_level.value in ["높음", "매우 높음"])


        # 투자 결정 로직
        if total_score >= 85 and unicorn_probability >= 0.7 and high_risk_count <= 1:
            return InvestmentRecommendation.INVEST
        elif total_score >= 70 and unicorn_probability >= 0.5 and high_risk_count <= 2:
            return InvestmentRecommendation.INVEST
        elif total_score >= 60 and high_risk_count <= 3:
            return InvestmentRecommendation.HOLD
        else:
            return InvestmentRecommendation.AVOID


    def generate_executive_summary(
        self,
        company_info: CompanyInfo,
        unicorn_score: UnicornScore,
        recommendation: InvestmentRecommendation
    ) -> str:
        """Executive Summary 생성"""
        try:
            # --- 여기가 수정된 부분입니다 ---
            # 1. llm_chain 사용, 2. .invoke()로 호출, 3. .content 없이 바로 사용
            prompt = self.executive_summary_prompt.format(
                company_name=company_info.name,
                total_score=unicorn_score.total_score,
                grade=unicorn_score.grade,
                unicorn_probability=unicorn_score.unicorn_probability,
                recommendation=recommendation.value
            )
            response = self.llm_chain.invoke(prompt)
            return response.strip()
        except Exception as e:
            return f"Executive Summary 생성 오류: {str(e)}"


    def generate_detailed_analysis(
        self,
        company_info: CompanyInfo,
        analysis_results: List[AnalysisResult],
        risk_assessments: List[RiskAssessment]
    ) -> str:
        """상세 분석 보고서 생성"""
        try:
            analysis_text = []
            for result in analysis_results:
                analysis_text.append(
                    f"[{result.category}] {result.score}점 ({result.grade}급)\n"
                    f"요약: {result.summary}\n"
                    f"강점: {', '.join(result.key_strengths)}\n"
                    f"약점: {', '.join(result.key_weaknesses)}\n"
                )

            risk_text = []
            for risk in risk_assessments:
                risk_text.append(
                    f"[{risk.category}] {risk.risk_level.value}\n"
                    f"설명: {risk.description}\n"
                    f"완화 전략: {', '.join(risk.mitigation_strategies)}\n"
                )

            # --- 여기가 수정된 부분입니다 ---
            prompt = self.detailed_analysis_prompt.format(
                company_name=company_info.name,
                analysis_results="\n".join(analysis_text),
                risk_assessments="\n".join(risk_text)
            )
            response = self.llm_chain.invoke(prompt)
            return response.strip()
        except Exception as e:
            return f"상세 분석 생성 오류: {str(e)}"


    def generate_investment_rationale(
        self,
        company_info: CompanyInfo,
        recommendation: InvestmentRecommendation,
        unicorn_score: UnicornScore
    ) -> str:
        """투자 근거 생성"""
        try:
            key_factors = unicorn_score.score_breakdown.get("unicorn_factors", [])
            key_factors_text = ", ".join(key_factors) if key_factors else "종합 분석 결과"

            # --- 여기가 수정된 부분입니다 ---
            prompt = self.investment_rationale_prompt.format(
                company_name=company_info.name,
                recommendation=recommendation.value,
                unicorn_score=f"{unicorn_score.total_score:.1f}점 ({unicorn_score.grade}급)",
                key_factors=key_factors_text
            )
            response = self.llm_chain.invoke(prompt)
            return response.strip()
        except Exception as e:
            return f"투자 근거 생성 오류: {str(e)}"

    # ... 이하 코드는 동일 ...
    # (generate_risk_summary 부터 끝까지의 코드는 그대로 두시면 됩니다.)
    def generate_risk_summary(self, risk_assessments: List[RiskAssessment]) -> str:
        """리스크 요약 생성"""
        if not risk_assessments:
            return "리스크 평가 결과가 없습니다."

        risk_summary = []
        risk_summary.append("## 리스크 요약")

        critical_risks = [r for r in risk_assessments if r.risk_level == RiskLevel.CRITICAL]
        high_risks = [r for r in risk_assessments if r.risk_level == RiskLevel.HIGH]
        medium_risks = [r for r in risk_assessments if r.risk_level == RiskLevel.MEDIUM]
        low_risks = [r for r in risk_assessments if r.risk_level == RiskLevel.LOW]

        if critical_risks:
            risk_summary.append("### 🔴 매우 높은 리스크")
            for risk in critical_risks:
                risk_summary.append(f"- **{risk.category}**: {risk.description}")

        if high_risks:
            risk_summary.append("### 🟠 높은 리스크")
            for risk in high_risks:
                risk_summary.append(f"- **{risk.category}**: {risk.description}")

        if medium_risks:
            risk_summary.append("### 🟡 보통 리스크")
            for risk in medium_risks:
                risk_summary.append(f"- **{risk.category}**: {risk.description}")

        if low_risks:
            risk_summary.append("### 🟢 낮은 리스크")
            for risk in low_risks:
                risk_summary.append(f"- **{risk.category}**: {risk.description}")

        return "\n".join(risk_summary)


    def calculate_confidence_level(
        self,
        analysis_results: List[AnalysisResult],
        risk_assessments: List[RiskAssessment],
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult]
    ) -> float:
        """신뢰도 레벨 계산"""
        confidence_factors = []
        data_confidence = min(len(documents) + len(external_results), 20) / 20.0
        confidence_factors.append(data_confidence)
        analysis_confidence = len(analysis_results) / 7.0 
        confidence_factors.append(analysis_confidence)
        risk_confidence = len(risk_assessments) / 6.0 
        confidence_factors.append(risk_confidence)
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        return min(overall_confidence, 1.0)


    def generate_investment_report(
        self,
        company_info: CompanyInfo,
        unicorn_score: UnicornScore,
        analysis_results: List[AnalysisResult],
        risk_assessments: List[RiskAssessment],
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult]
    ) -> InvestmentReport:
        """최종 투자 평가 리포트 생성"""
        recommendation = self.determine_investment_recommendation(unicorn_score, risk_assessments)
        executive_summary = self.generate_executive_summary(company_info, unicorn_score, recommendation)
        detailed_analysis = self.generate_detailed_analysis(company_info, analysis_results, risk_assessments)
        investment_rationale = self.generate_investment_rationale(company_info, recommendation, unicorn_score)
        risk_summary = self.generate_risk_summary(risk_assessments)
        confidence_level = self.calculate_confidence_level(analysis_results, risk_assessments, documents, external_results)
        
        data_sources = list(set([doc.source for doc in documents] + [res.source for res in external_results]))

        limitations = []
        if len(documents) < 5: limitations.append("제한된 내부 문서 데이터")
        if len(external_results) < 3: limitations.append("제한된 외부 검색 결과")
        if confidence_level < 0.7: limitations.append("중간 수준의 분석 신뢰도")

        return InvestmentReport(
            company_info=company_info, evaluation_date=datetime.now(),
            unicorn_score=unicorn_score, recommendation=recommendation,
            analysis_results=analysis_results, risk_assessments=risk_assessments,
            executive_summary=executive_summary, detailed_analysis=detailed_analysis,
            investment_rationale=investment_rationale, risk_summary=risk_summary,
            confidence_level=confidence_level, data_sources=data_sources, limitations=limitations
        )

# ... 이하 코드는 동일 ...

class ReportFormatter:
    """리포트 포맷터"""
    def __init__(self): pass

    def format_console_report(self, report: InvestmentReport) -> str:
        """콘솔용 리포트 포맷팅"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"🦄 AI 스타트업 투자 평가 리포트: {report.company_info.name}")
        lines.append("=" * 80)
        lines.append("\n📊 EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        lines.append(f"종합 점수: {report.unicorn_score.total_score:.1f}/100 ({report.unicorn_score.grade}급)")
        lines.append(f"유니콘 확률: {report.unicorn_score.unicorn_probability:.1%}")
        lines.append(f"투자 추천: {report.recommendation.value}")
        lines.append(f"신뢰도: {report.confidence_level:.1%}")
        lines.append(""); lines.append(report.executive_summary)
        lines.append("\n📈 영역별 점수카드"); lines.append("-" * 40)
        for result in report.analysis_results:
            lines.append(f"{result.category:20} {result.score:5.1f}점 ({result.grade}급)")
        lines.append("\n⚠️ 리스크 평가"); lines.append("-" * 40)
        for risk in report.risk_assessments:
            risk_emoji = {"낮음": "🟢", "보통": "🟡", "높음": "🟠", "매우 높음": "🔴"}
            emoji = risk_emoji.get(risk.risk_level.value, "⚪")
            lines.append(f"{emoji} {risk.category}: {risk.risk_level.value}")
        lines.append(f"\n💰 투자 권장사항"); lines.append("-" * 40)
        lines.append(report.investment_rationale)
        lines.append(f"\n📋 평가 정보"); lines.append("-" * 40)
        lines.append(f"평가 일시: {report.evaluation_date.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"데이터 소스: {len(report.data_sources)}개")
        if report.limitations: lines.append(f"제한사항: {', '.join(report.limitations)}")
        lines.append("=" * 80)
        return "\n".join(lines)
    
    def format_json_report(self, report: InvestmentReport) -> Dict[str, Any]:
        """JSON 형식 리포트 포맷팅"""
        return {
            "company_info": report.company_info.dict(),
            "evaluation_summary": {
                "evaluation_date": report.evaluation_date.isoformat(),
                "total_score": report.unicorn_score.total_score,
                "grade": report.unicorn_score.grade,
                "unicorn_probability": report.unicorn_score.unicorn_probability,
                "recommendation": report.recommendation.value,
                "confidence_level": report.confidence_level
            },
            "analysis_results": [r.dict() for r in report.analysis_results],
            "risk_assessments": [r.dict() for r in report.risk_assessments],
            "reports": {
                "executive_summary": report.executive_summary,
                "detailed_analysis": report.detailed_analysis,
                "investment_rationale": report.investment_rationale,
                "risk_summary": report.risk_summary
            },
            "metadata": {
                "data_sources": report.data_sources,
                "limitations": report.limitations,
                "score_breakdown": report.unicorn_score.score_breakdown
            }
        }


class ReportGenerationLayer:
    """리포트 생성 레이어 메인 클래스"""
    def __init__(self):
        self.report_generator = UnicornReportGenerator()
        self.formatter = ReportFormatter()

    def generate_report(self, context: PipelineContext) -> InvestmentReport:
        """투자 평가 리포트 생성"""
        return self.report_generator.generate_investment_report(
            company_info=context.company_info, unicorn_score=context.unicorn_score,
            analysis_results=context.analysis_results, risk_assessments=context.risk_assessments,
            documents=context.retrieved_documents, external_results=context.external_search_results
        )

    def format_report(self, report: InvestmentReport, format_type: str = "console") -> str:
        """리포트 포맷팅"""
        if format_type == "json": import json; return json.dumps(self.formatter.format_json_report(report), ensure_ascii=False, indent=2)
        return self.formatter.format_console_report(report)

def create_report_generation_layer() -> ReportGenerationLayer: return ReportGenerationLayer()

def process_report_generation_layer(context: PipelineContext) -> PipelineContext:
    """Report Generation Layer 처리 함수"""
    report_layer = create_report_generation_layer()
    investment_report = report_layer.generate_report(context)
    context.final_report = investment_report
    context.processing_steps.append(
        f"REPORT_GENERATION_LAYER: 투자 평가 리포트 생성 완료 (추천: {investment_report.recommendation.value})"
    )
    return context
