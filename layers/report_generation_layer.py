"""
Layer 8: REPORT GENERATION LAYER
unicorn_report_generator를 실행하여 최종 투자 평가 리포트를 생성하는 레이어
"""
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        self.llm = ChatOpenAI(
            openai_api_key=self.config["model"].openai_api_key,
            temperature=0.1,
            model=self.config["model"].model_name  # 예: "gpt-4o-mini"
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
        """투자 추천 결정 (개선된 로직)"""

        total_score = unicorn_score.total_score
        unicorn_probability = unicorn_score.unicorn_probability
        balance_score = unicorn_score.score_breakdown.get('balance_score', 100)

        # 리스크 평가
        high_risk_count = sum(1 for risk in risk_assessments
                             if risk.risk_level.value in ["높음", "매우 높음"])
        critical_risk_count = sum(1 for risk in risk_assessments
                                 if risk.risk_level.value == "매우 높음")

        # 균형도 체크 (어느 한 영역이라도 40점 미만이면 경고)
        balance_warning = balance_score < 40

        # 투자 결정 로직 (개선)
        # S급: 매우 강력한 투자 추천
        if (total_score >= 90 and unicorn_probability >= 0.8 and
            high_risk_count == 0 and not balance_warning):
            return InvestmentRecommendation.INVEST

        # A급: 강력한 투자 추천
        elif (total_score >= 80 and unicorn_probability >= 0.65 and
              critical_risk_count == 0 and not balance_warning):
            return InvestmentRecommendation.INVEST

        # B급: 조건부 투자 추천
        elif (total_score >= 70 and unicorn_probability >= 0.5 and
              high_risk_count <= 2 and balance_score >= 45):
            return InvestmentRecommendation.INVEST

        # C급: 관망 (Hold)
        elif (total_score >= 60 and high_risk_count <= 3 and balance_score >= 40):
            return InvestmentRecommendation.HOLD

        # D급 or 높은 리스크: 회피
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
            response = self.llm.invoke(self.executive_summary_prompt.format(
                company_name=company_info.name,
                total_score=unicorn_score.total_score,
                grade=unicorn_score.grade,
                unicorn_probability=unicorn_score.unicorn_probability,
                recommendation=recommendation.value
            ))

            # GPT 응답을 터미널에 출력
            print(f"\n[검색] REPORT_GENERATION_LAYER (EXECUTIVE_SUMMARY) - GPT 응답:")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

            return response.content.strip()
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
            # 분석 결과 포맷팅
            analysis_text = []
            for result in analysis_results:
                analysis_text.append(
                    f"[{result.category}] {result.score}점 ({result.grade}급)\n"
                    f"요약: {result.summary}\n"
                    f"강점: {', '.join(result.key_strengths)}\n"
                    f"약점: {', '.join(result.key_weaknesses)}\n"
                )

            # 리스크 평가 포맷팅
            risk_text = []
            for risk in risk_assessments:
                risk_text.append(
                    f"[{risk.category}] {risk.risk_level.value}\n"
                    f"설명: {risk.description}\n"
                    f"완화 전략: {', '.join(risk.mitigation_strategies)}\n"
                )

            response = self.llm.invoke(self.detailed_analysis_prompt.format(
                company_name=company_info.name,
                analysis_results="\n".join(analysis_text),
                risk_assessments="\n".join(risk_text)
            ))

            # GPT 응답을 터미널에 출력
            print(f"\n[검색] REPORT_GENERATION_LAYER (DETAILED_ANALYSIS) - GPT 응답:")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

            return response.content.strip()
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

            response = self.llm.invoke(self.investment_rationale_prompt.format(
                company_name=company_info.name,
                recommendation=recommendation.value,
                unicorn_score=f"{unicorn_score.total_score:.1f}점 ({unicorn_score.grade}급)",
                key_factors=key_factors_text
            ))

            # GPT 응답을 터미널에 출력
            print(f"\n[검색] REPORT_GENERATION_LAYER (INVESTMENT_RATIONALE) - GPT 응답:")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

            return response.content.strip()
        except Exception as e:
            return f"투자 근거 생성 오류: {str(e)}"

    def generate_risk_summary(self, risk_assessments: List[RiskAssessment]) -> str:
        """리스크 요약 생성"""
        if not risk_assessments:
            return "리스크 평가 결과가 없습니다."

        risk_summary = []
        risk_summary.append("## 리스크 요약")

        # 리스크 레벨별 분류
        critical_risks = [r for r in risk_assessments if r.risk_level == RiskLevel.CRITICAL]
        high_risks = [r for r in risk_assessments if r.risk_level == RiskLevel.HIGH]
        medium_risks = [r for r in risk_assessments if r.risk_level == RiskLevel.MEDIUM]
        low_risks = [r for r in risk_assessments if r.risk_level == RiskLevel.LOW]

        if critical_risks:
            risk_summary.append("###  매우 높은 리스크")
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
        """신뢰도 레벨 계산 (개선)"""
        confidence_factors = []

        # 1. 데이터 양 기반 신뢰도
        max_data_sources = int(os.getenv("MAX_DATA_SOURCES_FOR_CONFIDENCE", "20"))
        total_data = len(documents) + len(external_results)
        data_confidence = min(total_data, max_data_sources) / max_data_sources
        confidence_factors.append(("data_volume", data_confidence, 0.25))

        # 2. 분석 완성도 기반 신뢰도
        max_analysis_areas = int(os.getenv("MAX_ANALYSIS_AREAS", "4"))
        analysis_confidence = len(analysis_results) / max_analysis_areas
        confidence_factors.append(("analysis_completeness", analysis_confidence, 0.30))

        # 3. 리스크 평가 완성도
        max_risk_categories = int(os.getenv("MAX_RISK_CATEGORIES", "6"))
        risk_confidence = len(risk_assessments) / max_risk_categories
        confidence_factors.append(("risk_assessment", risk_confidence, 0.20))

        # 4. 분석 품질 (평균 점수 분산도)
        if analysis_results:
            scores = [r.score for r in analysis_results]
            avg_score = sum(scores) / len(scores)
            # 점수가 너무 고르면(분산이 낮으면) 신뢰도 향상
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            quality_confidence = max(0, 1 - (variance / 1000))  # 분산이 낮을수록 높은 신뢰도
            confidence_factors.append(("analysis_quality", quality_confidence, 0.15))

        # 5. 데이터 다양성 (내부+외부 데이터 균형)
        if total_data > 0:
            internal_ratio = len(documents) / total_data
            diversity_confidence = 1 - abs(0.5 - internal_ratio)  # 50:50에 가까울수록 높은 신뢰도
            confidence_factors.append(("data_diversity", diversity_confidence, 0.10))

        # 가중 평균 신뢰도 계산
        weighted_sum = sum(conf * weight for _, conf, weight in confidence_factors)
        total_weight = sum(weight for _, _, weight in confidence_factors)
        overall_confidence = weighted_sum / total_weight if total_weight > 0 else 0.5

        return min(max(overall_confidence, 0.0), 1.0)

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

        # 투자 추천 결정
        recommendation = self.determine_investment_recommendation(unicorn_score, risk_assessments)

        # 각 섹션별 내용 생성
        executive_summary = self.generate_executive_summary(
            company_info, unicorn_score, recommendation
        )

        detailed_analysis = self.generate_detailed_analysis(
            company_info, analysis_results, risk_assessments
        )

        investment_rationale = self.generate_investment_rationale(
            company_info, recommendation, unicorn_score
        )

        risk_summary = self.generate_risk_summary(risk_assessments)

        # 신뢰도 계산
        confidence_level = self.calculate_confidence_level(
            analysis_results, risk_assessments, documents, external_results
        )

        # 데이터 소스 정리
        data_sources = []
        for doc in documents:
            data_sources.append(doc.source)
        for result in external_results:
            data_sources.append(result.source)
        data_sources = list(set(data_sources))  # 중복 제거

        # 제한사항 정리
        limitations = []
        min_documents = int(os.getenv("MIN_DOCUMENTS_FOR_LIMITATION", "5"))
        min_external_results = int(os.getenv("MIN_EXTERNAL_RESULTS_FOR_LIMITATION", "3"))
        min_confidence_level = float(os.getenv("MIN_CONFIDENCE_LEVEL_FOR_LIMITATION", "0.7"))
        
        if len(documents) < min_documents:
            limitations.append("제한된 내부 문서 데이터")
        if len(external_results) < min_external_results:
            limitations.append("제한된 외부 검색 결과")
        if confidence_level < min_confidence_level:
            limitations.append("중간 수준의 분석 신뢰도")

        # 최종 리포트 생성
        report = InvestmentReport(
            company_info=company_info,
            evaluation_date=datetime.now(),
            unicorn_score=unicorn_score,
            recommendation=recommendation,
            analysis_results=analysis_results,
            risk_assessments=risk_assessments,
            executive_summary=executive_summary,
            detailed_analysis=detailed_analysis,
            investment_rationale=investment_rationale,
            risk_summary=risk_summary,
            confidence_level=confidence_level,
            data_sources=data_sources,
            limitations=limitations
        )

        return report

class ReportFormatter:
    """리포트 포맷터"""

    def __init__(self):
        pass

    def format_pdf_report(self, report: InvestmentReport, output_path: str) -> bool:
        """PDF 형식 리포트 생성"""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import A4, letter
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
            from reportlab.platypus import Image as RLImage
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
            import os

            # 한글 폰트 설정 (Windows)
            try:
                font_path = "C:\\Windows\\Fonts\\malgun.ttf"  # 맑은 고딕
                if os.path.exists(font_path):
                    pdfmetrics.registerFont(TTFont('Malgun', font_path))
                    font_name = 'Malgun'
                else:
                    font_name = 'Helvetica'  # 폴백 폰트
            except:
                font_name = 'Helvetica'

            # PDF 문서 생성
            doc = SimpleDocTemplate(output_path, pagesize=A4,
                                   rightMargin=72, leftMargin=72,
                                   topMargin=72, bottomMargin=18)

            # 스타일 설정
            styles = getSampleStyleSheet()

            # 커스텀 스타일 정의
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontName=font_name,
                fontSize=24,
                textColor=colors.HexColor('#1a237e'),
                spaceAfter=30,
                alignment=TA_CENTER
            )

            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontName=font_name,
                fontSize=16,
                textColor=colors.HexColor('#283593'),
                spaceAfter=12,
                spaceBefore=12
            )

            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontName=font_name,
                fontSize=10,
                leading=14
            )

            # 문서 구성 요소들
            story = []

            # 제목
            story.append(Paragraph(f" AI 스타트업 투자 평가 리포트", title_style))
            story.append(Paragraph(f"{report.company_info.name}", title_style))
            story.append(Spacer(1, 20))

            # Executive Summary
            story.append(Paragraph("[분석] EXECUTIVE SUMMARY", heading_style))

            # 요약 정보 테이블
            balance_info = report.unicorn_score.score_breakdown.get('balance_score', 'N/A')
            balance_text = f"{balance_info:.1f}점" if isinstance(balance_info, (int, float)) else str(balance_info)

            summary_data = [
                ['항목', '값'],
                ['종합 점수', f"{report.unicorn_score.total_score:.1f}/100 ({report.unicorn_score.grade}급)"],
                ['유니콘 확률', f"{report.unicorn_score.unicorn_probability:.1%}"],
                ['투자 추천', report.recommendation.value],
                ['신뢰도', f"{report.confidence_level:.1%}"],
                ['균형도 (최저점수)', balance_text],
                ['평가 일시', report.evaluation_date.strftime('%Y-%m-%d %H:%M:%S')]
            ]

            summary_table = Table(summary_data, colWidths=[2.5*inch, 3.5*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), font_name),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(summary_table)
            story.append(Spacer(1, 20))

            # Executive Summary 내용
            summary_text = report.executive_summary.replace('\n', '<br/>')
            story.append(Paragraph(summary_text, normal_style))
            story.append(Spacer(1, 20))

            # 영역별 점수카드
            story.append(Paragraph("[성장] 영역별 점수카드", heading_style))

            score_data = [['영역', '점수', '등급']]
            for result in report.analysis_results:
                score_data.append([
                    result.category,
                    f"{result.score:.1f}점",
                    result.grade
                ])

            score_table = Table(score_data, colWidths=[3*inch, 1.5*inch, 1.5*inch])
            score_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, -1), font_name),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(score_table)
            story.append(Spacer(1, 20))

            # 시각화: 점수 바 차트
            try:
                from reportlab.graphics.shapes import Drawing
                from reportlab.graphics.charts.barcharts import VerticalBarChart
                from reportlab.graphics.charts.legends import Legend

                drawing = Drawing(400, 200)
                bc = VerticalBarChart()
                bc.x = 50
                bc.y = 50
                bc.height = 125
                bc.width = 300
                bc.data = [[result.score for result in report.analysis_results]]
                bc.strokeColor = colors.black
                bc.valueAxis.valueMin = 0
                bc.valueAxis.valueMax = 100
                bc.valueAxis.valueStep = 20
                bc.categoryAxis.labels.boxAnchor = 'ne'
                bc.categoryAxis.labels.dx = 8
                bc.categoryAxis.labels.dy = -2
                bc.categoryAxis.labels.angle = 30
                bc.categoryAxis.categoryNames = [r.category[:15] for r in report.analysis_results]

                # 바 색상 설정
                bc.bars[0].fillColor = colors.HexColor('#4285F4')

                drawing.add(bc)
                story.append(drawing)
                story.append(Spacer(1, 20))
            except Exception as e:
                print(f"바 차트 생성 실패: {e}")

            # 시각화: 레이더 차트 (종합 평가)
            try:
                from reportlab.graphics.charts.spider import SpiderChart

                drawing2 = Drawing(400, 200)
                sp = SpiderChart()
                sp.x = 120
                sp.y = 20
                sp.width = 180
                sp.height = 180
                sp.data = [[result.score for result in report.analysis_results]]
                sp.labels = [r.category[:10] + '...' if len(r.category) > 10 else r.category
                            for r in report.analysis_results]
                sp.strands[0].fillColor = colors.HexColor('#4285F4')
                sp.strands[0].strokeColor = colors.HexColor('#1a237e')
                sp.strands[0].strokeWidth = 2

                drawing2.add(sp)
                story.append(drawing2)
                story.append(Spacer(1, 20))
            except Exception as e:
                print(f"레이더 차트 생성 실패: {e}")

            # 리스크 평가
            story.append(Paragraph("[경고] 리스크 평가", heading_style))

            risk_data = [['카테고리', '리스크 레벨', '설명']]
            for risk in report.risk_assessments:
                risk_emoji = {
                    "낮음": "[낮음]",
                    "보통": "[보통]",
                    "높음": "[높음]",
                    "매우 높음": "[매우높음]"
                }
                emoji = risk_emoji.get(risk.risk_level.value, "")
                risk_data.append([
                    risk.category,
                    f"{emoji} {risk.risk_level.value}",
                    risk.description[:100] + "..." if len(risk.description) > 100 else risk.description
                ])

            risk_table = Table(risk_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), font_name),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgoldenrodyellow),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            story.append(risk_table)
            story.append(Spacer(1, 20))

            # 페이지 나누기
            story.append(PageBreak())

            # 상세 분석
            story.append(Paragraph(" 상세 분석", heading_style))
            detailed_text = report.detailed_analysis.replace('\n', '<br/>')
            story.append(Paragraph(detailed_text, normal_style))
            story.append(Spacer(1, 20))

            # 투자 권장사항
            story.append(Paragraph("[재무] 투자 권장사항", heading_style))
            rationale_text = report.investment_rationale.replace('\n', '<br/>')
            story.append(Paragraph(rationale_text, normal_style))
            story.append(Spacer(1, 20))

            # 메타데이터
            story.append(Paragraph(" 평가 정보", heading_style))
            meta_data = [
                ['데이터 소스', f"{len(report.data_sources)}개"],
                ['제한사항', ', '.join(report.limitations) if report.limitations else '없음']
            ]
            meta_table = Table(meta_data, colWidths=[2*inch, 4*inch])
            meta_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), font_name),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            story.append(meta_table)

            # PDF 생성
            doc.build(story)
            return True

        except Exception as e:
            print(f"PDF 생성 실패: {e}")
            import traceback
            traceback.print_exc()
            return False

    def format_console_report(self, report: InvestmentReport) -> str:
        """콘솔용 리포트 포맷팅"""
        lines = []

        # 헤더
        lines.append("=" * 80)
        lines.append(f" AI 스타트업 투자 평가 리포트: {report.company_info.name}")
        lines.append("=" * 80)

        # Executive Summary
        lines.append("\n[분석] EXECUTIVE SUMMARY")
        lines.append("-" * 40)
        lines.append(f"종합 점수: {report.unicorn_score.total_score:.1f}/100 ({report.unicorn_score.grade}급)")
        lines.append(f"유니콘 확률: {report.unicorn_score.unicorn_probability:.1%}")
        lines.append(f"투자 추천: {report.recommendation.value}")
        lines.append(f"신뢰도: {report.confidence_level:.1%}")
        lines.append("")
        lines.append(report.executive_summary)

        # 영역별 점수
        lines.append("\n[성장] 영역별 점수카드")
        lines.append("-" * 40)
        for result in report.analysis_results:
            lines.append(f"{result.category:20} {result.score:5.1f}점 ({result.grade}급)")

        # 리스크 평가
        lines.append("\n[경고] 리스크 평가")
        lines.append("-" * 40)
        for risk in report.risk_assessments:
            risk_emoji = {
                "낮음": "[낮음]",
                "보통": "[보통]",
                "높음": "[높음]",
                "매우 높음": "[매우높음]"
            }
            emoji = risk_emoji.get(risk.risk_level.value, "")
            lines.append(f"{emoji} {risk.category}: {risk.risk_level.value}")

        # 투자 권장사항
        lines.append(f"\n[재무] 투자 권장사항")
        lines.append("-" * 40)
        lines.append(report.investment_rationale)

        # 메타데이터
        lines.append(f"\n 평가 정보")
        lines.append("-" * 40)
        lines.append(f"평가 일시: {report.evaluation_date.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"데이터 소스: {len(report.data_sources)}개")
        if report.limitations:
            lines.append(f"제한사항: {', '.join(report.limitations)}")

        lines.append("=" * 80)

        return "\n".join(lines)

    def format_json_report(self, report: InvestmentReport) -> Dict[str, Any]:
        """JSON 형식 리포트 포맷팅"""
        return {
            "company_info": {
                "name": report.company_info.name,
                "industry": report.company_info.industry,
                "founded_year": report.company_info.founded_year,
                "headquarters": report.company_info.headquarters,
                "description": report.company_info.description
            },
            "evaluation_summary": {
                "evaluation_date": report.evaluation_date.isoformat(),
                "total_score": report.unicorn_score.total_score,
                "grade": report.unicorn_score.grade,
                "unicorn_probability": report.unicorn_score.unicorn_probability,
                "recommendation": report.recommendation.value,
                "confidence_level": report.confidence_level
            },
            "analysis_results": [
                {
                    "category": result.category,
                    "score": result.score,
                    "grade": result.grade,
                    "summary": result.summary,
                    "strengths": result.key_strengths,
                    "weaknesses": result.key_weaknesses
                }
                for result in report.analysis_results
            ],
            "risk_assessments": [
                {
                    "category": risk.category,
                    "risk_level": risk.risk_level.value,
                    "description": risk.description,
                    "impact_score": risk.impact_score,
                    "probability": risk.probability,
                    "mitigation_strategies": risk.mitigation_strategies
                }
                for risk in report.risk_assessments
            ],
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

    def generate_report(
        self,
        company_info: CompanyInfo,
        unicorn_score: UnicornScore,
        analysis_results: List[AnalysisResult],
        risk_assessments: List[RiskAssessment],
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult]
    ) -> InvestmentReport:
        """투자 평가 리포트 생성"""
        return self.report_generator.generate_investment_report(
            company_info=company_info,
            unicorn_score=unicorn_score,
            analysis_results=analysis_results,
            risk_assessments=risk_assessments,
            documents=documents,
            external_results=external_results
        )

    def format_report(self, report: InvestmentReport, format_type: str = "console") -> str:
        """리포트 포맷팅"""
        if format_type == "console":
            return self.formatter.format_console_report(report)
        elif format_type == "json":
            import json
            return json.dumps(self.formatter.format_json_report(report), ensure_ascii=False, indent=2)
        else:
            return self.formatter.format_console_report(report)

def create_report_generation_layer() -> ReportGenerationLayer:
    """Report Generation Layer 생성자"""
    return ReportGenerationLayer()

def process_report_generation_layer(context: PipelineContext) -> PipelineContext:
    """Report Generation Layer 처리 함수"""
    report_layer = create_report_generation_layer()

    # 최종 투자 평가 리포트 생성
    investment_report = report_layer.generate_report(
        company_info=context.company_info,
        unicorn_score=context.unicorn_score,
        analysis_results=context.analysis_results,
        risk_assessments=context.risk_assessments,
        documents=context.retrieved_documents,
        external_results=context.external_search_results
    )

    context.final_report = investment_report

    # 처리 단계 기록
    context.processing_steps.append(
        f"REPORT_GENERATION_LAYER: 투자 평가 리포트 생성 완료 "
        f"(추천: {investment_report.recommendation.value})"
    )

    return context