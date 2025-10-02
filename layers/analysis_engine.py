"""
Layer 5: ANALYSIS ENGINE
분석 레이어 (4개 주요 분석기 병렬 실행)
"""
import os
from typing import List
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AnalysisResult, DocumentChunk, ExternalSearchResult, PipelineContext, CompanyInfo
from config import get_config
from layers.quantitative_scorer import QuantitativeScorer

class BaseAnalyzer:
    def __init__(self, analyzer_name: str):
        self.analyzer_name = analyzer_name
        self.config = get_config()
        self.llm = ChatOpenAI(
            openai_api_key=self.config["model"].openai_api_key,
            temperature=0.1,
            model=self.config["model"].model_name
        )

    def analyze(self, company_info, documents, external_results) -> AnalysisResult:
        raise NotImplementedError

    def _create_context_summary(self, documents, external_results) -> str:
        context_parts = []
        if documents:
            doc_summaries = [f"- {doc.content[:200]}..." for doc in documents[:5]]
            context_parts.append("관련 문서:\n" + "\n".join(doc_summaries))
        if external_results:
            ext_summaries = [f"- {r.title}: {r.content[:150]}..." for r in external_results[:3]]
            context_parts.append("외부 검색:\n" + "\n".join(ext_summaries))
        return "\n\n".join(context_parts)

    def _calculate_grade(self, score: float) -> str:
        th = self.config["scoring"].grade_thresholds
        if score >= th["S"]: return "S"
        elif score >= th["A"]: return "A"
        elif score >= th["B"]: return "B"
        elif score >= th["C"]: return "C"
        return "D"

# ---------------------------
# GrowthAnalyzer
# ---------------------------
class GrowthAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("growth_analysis")
        self.analysis_prompt = PromptTemplate(
            input_variables=["company_name", "company_info", "context"],
            template="""{company_name}의 성장성을 **정량적 지표 중심으로** 평가하세요.

회사 정보:
{company_info}
자료:
{context}

**평가 기준 (정량 지표 우선):**
1. 매출 성장률 (30점): YoY 100% 이상=30, 50%=25, 30%=20, 10%=10, 데이터없음=5
2. 거래액/사용자 증가율 (30점): 누적 1조 이상=30, 5000억=25, 1000억=15, 100억=5, 데이터없음=3
3. 시장 점유율/MAU 성장 (20점): 업계 Top3=20, Top10=15, 그 외=10, 데이터없음=5
4. 투자 유치 규모 (20점): 300억 이상=20, 100억=15, 50억=10, 그 이하=5, 데이터없음=3

**필수 요구사항:**
- 각 지표마다 출처를 명시하세요 (예: "2023년 재무제표", "IR 자료", "뉴스 기사" 등)
- 구체적인 숫자와 연도를 반드시 포함하세요
- supporting_evidence에는 "지표명: 수치 (출처)" 형식으로 작성하세요

0~100 점수와 JSON으로:
{{
 "score": 80,
 "summary": "핵심 지표 3개 이내로 간결하게",
 "detailed_analysis": "각 평가 기준별로 점수 산정 근거와 출처 명시",
 "key_strengths": ["강점 + 수치 + 출처"],
 "key_weaknesses": ["약점 + 구체적 이유"],
 "supporting_evidence": ["매출: 500억원 (2023년 재무제표)", "투자유치: 100억원 (2022년 IR자료)"]
}}"""
        )

    def analyze(self, company_info, documents, external_results) -> AnalysisResult:
        if not documents and not external_results:
            return AnalysisResult(category=self.analyzer_name, score=30.0, grade="D",
                                  summary="성장성 분석: 데이터 부족으로 제한적 평가",
                                  detailed_analysis="충분한 데이터가 없어 상세 분석이 어렵습니다.",
                                  key_strengths=[], key_weaknesses=["데이터 부족"], supporting_evidence=[])
        context = self._create_context_summary(documents, external_results)
        info_text = f"업종:{company_info.industry}, 설립년도:{company_info.founded_year}"
        try:
            resp = self.llm.invoke(self.analysis_prompt.format(
                company_name=company_info.name, company_info=info_text, context=context
            ))
            print(f"\n[검색] {self.analyzer_name.upper()} GPT 응답:\n{resp.content}\n")
            import json
            # JSON 파싱 개선
            content = resp.content.strip()
            # 마크다운 코드블록 제거
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            data = json.loads(content)
            score = float(data.get("score", 40.0))
            # 점수 범위 검증
            score = max(0.0, min(100.0, score))
            return AnalysisResult(
                category=self.analyzer_name,
                score=score,
                grade=self._calculate_grade(score),
                summary=data.get("summary", ""),
                detailed_analysis=data.get("detailed_analysis", ""),
                key_strengths=data.get("key_strengths", []),
                key_weaknesses=data.get("key_weaknesses", []),
                supporting_evidence=data.get("supporting_evidence", []),
            )
        except Exception as e:
            print(f"[경고] {self.analyzer_name} 분석 오류: {e}")
            return AnalysisResult(category=self.analyzer_name, score=40.0, grade="D",
                                  summary="성장성 분석 중 오류 발생",
                                  detailed_analysis=f"분석 처리 중 오류: {str(e)}",
                                  key_strengths=[], key_weaknesses=["분석 오류"],
                                  supporting_evidence=[])

# ---------------------------
# BusinessModelAnalyzer
# ---------------------------
class BusinessModelAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("business_model_analysis")
        self.analysis_prompt = PromptTemplate(
            input_variables=["company_name", "company_info", "context"],
            template="""{company_name}의 비즈니스 모델을 **수익성과 지속가능성 중심으로** 평가하세요.

회사 정보:
{company_info}
자료:
{context}

**평가 기준 (정량 지표 우선):**
1. 수익 모델 명확성 (25점): B2B2C 수수료 등 명확한 수익원=25, 불명확=10, 데이터없음=5
2. 수익성 개선 추세 (25점): 흑자 전환=25, 적자 감소 추세=20, 적자 확대=5, 데이터없음=3
3. 거래 규모의 경제 (25점): 연간 거래액 1조 이상=25, 1000억=15, 100억=5, 데이터없음=3
4. 사업 다각화 (25점): 3개 이상 사업=25, 2개=15, 1개=10, 데이터없음=5

**필수 요구사항:**
- 각 지표마다 출처를 명시하세요
- 매출액, 영업손실, 수수료율 등 구체적 재무 숫자 필수
- supporting_evidence에는 "지표명: 수치 (출처)" 형식으로 작성

0~100 점수와 JSON으로:
{{
 "score": 75,
 "summary": "핵심 비즈니스 모델 3줄 요약",
 "detailed_analysis": "각 평가 기준별 점수 산정 근거와 출처",
 "key_strengths": ["강점 + 수치 + 출처"],
 "key_weaknesses": ["약점 + 이유"],
 "supporting_evidence": ["연간 거래액: 7조원 (IR자료)", "영업이익률: -15% (재무제표)"]
}}"""
        )

    def analyze(self, company_info, documents, external_results) -> AnalysisResult:
        if not documents and not external_results:
            return AnalysisResult(category=self.analyzer_name, score=30.0, grade="D",
                                  summary="비즈니스 모델 분석: 데이터 부족으로 제한적 평가",
                                  detailed_analysis="충분한 데이터가 없어 상세 분석이 어렵습니다.",
                                  key_strengths=[], key_weaknesses=["데이터 부족"], supporting_evidence=[])
        context = self._create_context_summary(documents, external_results)
        info_text = f"업종:{company_info.industry}, 설명:{company_info.description}"
        try:
            resp = self.llm.invoke(self.analysis_prompt.format(
                company_name=company_info.name, company_info=info_text, context=context
            ))
            print(f"\n[검색] {self.analyzer_name.upper()} GPT 응답:\n{resp.content}\n")
            import json
            content = resp.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            data = json.loads(content)
            score = float(data.get("score", 40.0))
            score = max(0.0, min(100.0, score))
            return AnalysisResult(
                category=self.analyzer_name,
                score=score,
                grade=self._calculate_grade(score),
                summary=data.get("summary", ""),
                detailed_analysis=data.get("detailed_analysis", ""),
                key_strengths=data.get("key_strengths", []),
                key_weaknesses=data.get("key_weaknesses", []),
                supporting_evidence=data.get("supporting_evidence", []),
            )
        except Exception as e:
            print(f"[경고] {self.analyzer_name} 분석 오류: {e}")
            return AnalysisResult(category=self.analyzer_name, score=40.0, grade="D",
                                  summary="비즈니스 모델 분석 중 오류 발생",
                                  detailed_analysis=f"분석 처리 중 오류: {str(e)}",
                                  key_strengths=[], key_weaknesses=["분석 오류"],
                                  supporting_evidence=[])

# ---------------------------
# TechSecurityAnalyzer
# ---------------------------
class TechSecurityAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("tech_security_analysis")
        self.analysis_prompt = PromptTemplate(
            input_variables=["company_name", "company_info", "context"],
            template="""{company_name}의 기술력/보안성을 평가하세요.
회사 정보:
{company_info}
자료:
{context}

0~100 점수와 JSON으로:
{{
 "score": 70,
 "summary": "...",
 "detailed_analysis": "...",
 "key_strengths": ["..."],
 "key_weaknesses": ["..."],
 "supporting_evidence": ["..."]
}}"""
        )

    def analyze(self, company_info, documents, external_results) -> AnalysisResult:
        if not documents and not external_results:
            return AnalysisResult(category=self.analyzer_name, score=30.0, grade="D",
                                  summary="기술력/보안성 분석: 데이터 부족으로 제한적 평가",
                                  detailed_analysis="충분한 데이터가 없어 상세 분석이 어렵습니다.",
                                  key_strengths=[], key_weaknesses=["데이터 부족"], supporting_evidence=[])
        context = self._create_context_summary(documents, external_results)
        info_text = f"업종:{company_info.industry}, 설명:{company_info.description}"
        try:
            resp = self.llm.invoke(self.analysis_prompt.format(
                company_name=company_info.name, company_info=info_text, context=context
            ))
            print(f"\n[검색] {self.analyzer_name.upper()} GPT 응답:\n{resp.content}\n")
            import json
            content = resp.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            data = json.loads(content)
            score = float(data.get("score", 40.0))
            score = max(0.0, min(100.0, score))
            return AnalysisResult(
                category=self.analyzer_name,
                score=score,
                grade=self._calculate_grade(score),
                summary=data.get("summary", ""),
                detailed_analysis=data.get("detailed_analysis", ""),
                key_strengths=data.get("key_strengths", []),
                key_weaknesses=data.get("key_weaknesses", []),
                supporting_evidence=data.get("supporting_evidence", []),
            )
        except Exception as e:
            print(f"[경고] {self.analyzer_name} 분석 오류: {e}")
            return AnalysisResult(category=self.analyzer_name, score=40.0, grade="D",
                                  summary="기술력/보안성 분석 중 오류 발생",
                                  detailed_analysis=f"분석 처리 중 오류: {str(e)}",
                                  key_strengths=[], key_weaknesses=["분석 오류"],
                                  supporting_evidence=[])

# ---------------------------
# FinancialHealthAnalyzer
# ---------------------------
class FinancialHealthAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("financial_health_analysis")
        self.analysis_prompt = PromptTemplate(
            input_variables=["company_name", "company_info", "context"],
            template="""{company_name}의 재무건전성을 **정량 재무지표 중심으로** 평가하세요.

회사 정보:
{company_info}
자료:
{context}

**평가 기준 (재무제표 숫자 기반):**
1. 투자 유치 실적 (30점): 최근 투자 500억 이상=30, 300억=25, 100억=15, 50억 이하=5
2. 매출 규모 (25점): 연매출 100억 이상=25, 50억=20, 30억=15, 10억 이하=5
3. 적자/결손 상태 (25점): 흑자=25, 적자 감소=20, 적자 증가=10, 누적 결손 심각=5
4. 자본잠식 여부 (20점): 순자산 양호=20, 자본잠식 우려=10, 자본잠식=0

**중요:**
- 매출액, 영업손실, 순손실, 투자금액 등 문서에 명시된 **실제 숫자**를 반드시 인용
- "2022년 매출 31억 vs 2023년 45억 (45% 성장)" 같은 구체적 비교 필수

0~100 점수와 JSON으로:
{{
 "score": 65,
 "summary": "구체적 재무 숫자 포함",
 "detailed_analysis": "실제 매출/투자 금액 기반 분석",
 "key_strengths": ["정확한 금액 인용"],
 "key_weaknesses": ["구체적 재무 약점"],
 "supporting_evidence": ["재무제표 숫자"]
}}"""
        )

    def analyze(self, company_info, documents, external_results) -> AnalysisResult:
        if not documents and not external_results:
            return AnalysisResult(category=self.analyzer_name, score=30.0, grade="D",
                                  summary="재무건전성 분석: 데이터 부족으로 제한적 평가",
                                  detailed_analysis="충분한 데이터가 없어 상세 분석이 어렵습니다.",
                                  key_strengths=[], key_weaknesses=["데이터 부족"], supporting_evidence=[])
        context = self._create_context_summary(documents, external_results)
        info_text = f"업종:{company_info.industry}, 설립년도:{company_info.founded_year}"
        try:
            resp = self.llm.invoke(self.analysis_prompt.format(
                company_name=company_info.name, company_info=info_text, context=context
            ))
            print(f"\n[검색] {self.analyzer_name.upper()} GPT 응답:\n{resp.content}\n")
            import json
            content = resp.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()
            data = json.loads(content)
            score = float(data.get("score", 40.0))
            score = max(0.0, min(100.0, score))
            return AnalysisResult(
                category=self.analyzer_name,
                score=score,
                grade=self._calculate_grade(score),
                summary=data.get("summary", ""),
                detailed_analysis=data.get("detailed_analysis", ""),
                key_strengths=data.get("key_strengths", []),
                key_weaknesses=data.get("key_weaknesses", []),
                supporting_evidence=data.get("supporting_evidence", []),
            )
        except Exception as e:
            print(f"[경고] {self.analyzer_name} 분석 오류: {e}")
            return AnalysisResult(category=self.analyzer_name, score=40.0, grade="D",
                                  summary="재무건전성 분석 중 오류 발생",
                                  detailed_analysis=f"분석 처리 중 오류: {str(e)}",
                                  key_strengths=[], key_weaknesses=["분석 오류"],
                                  supporting_evidence=[])

# ---------------------------
# AnalysisEngine
# ---------------------------
class AnalysisEngine:
    def __init__(self):
        self.analyzers = {
            "growth_analysis": GrowthAnalyzer(),
            "business_model_analysis": BusinessModelAnalyzer(),
            "tech_security_analysis": TechSecurityAnalyzer(),
            "financial_health_analysis": FinancialHealthAnalyzer(),
        }
        self.quant_scorer = QuantitativeScorer()

    def run_parallel_analysis(self, company_info, documents, external_results, selected=None):
        if selected is None:
            selected = list(self.analyzers.keys())

        # STEP 1: 정량 평가 먼저 실행 (규칙 기반)
        print("\n" + "="*80)
        print("[분석] STEP 1: 정량 지표 기반 객관적 평가")
        print("="*80)

        metrics = self.quant_scorer.extract_financial_metrics(documents, external_results)
        print(f"\n추출된 재무 지표:")
        print(f"  - 매출: {metrics['revenue']:.0f}억 원")
        print(f"  - 거래액: {metrics['transaction_volume']:.1f}조 원")
        print(f"  - 투자 유치: {metrics['investment']:.0f}억 원")
        print(f"  - 기업가치: {metrics['valuation']:.0f}억 원")
        print(f"  - 영업손실: {metrics['loss']:.0f}억 원")
        print(f"  - 사용자: {metrics['users']:.0f}만 명")
        print(f"  - 성장률: {metrics['growth_rate']:.0f}%")

        quant_scores = self.quant_scorer.calculate_scores(metrics)
        print(f"\n정량 평가 점수:")
        for category, score in quant_scores.items():
            print(f"  - {category}: {score:.1f}점")

        # STEP 2: GPT 정성 분석 (정량 점수를 기준으로 보정)
        print("\n" + "="*80)
        print("[AI] STEP 2: GPT 정성 분석 (정량 점수 기반 보정)")
        print("="*80)

        selected_analyzers = {n:a for n,a in self.analyzers.items() if n in selected}
        results = []

        with ThreadPoolExecutor(max_workers=len(selected_analyzers)) as ex:
            futures = {ex.submit(a.analyze, company_info, documents, external_results): n
                       for n,a in selected_analyzers.items()}
            for f in futures:
                try:
                    r = f.result(timeout=int(os.getenv("ANALYSIS_TIMEOUT_SECONDS","60")))

                    # 정량 점수와 GPT 점수 혼합 (정량 70%, GPT 30%)
                    if r.category in quant_scores:
                        quant_score = quant_scores[r.category]
                        gpt_score = r.score
                        final_score = quant_score * 0.7 + gpt_score * 0.3

                        print(f"\n{r.category}:")
                        print(f"  정량: {quant_score:.1f}, GPT: {gpt_score:.1f} → 최종: {final_score:.1f}")

                        # 점수 업데이트
                        r.score = final_score
                        r.grade = self._calculate_grade(final_score)

                    results.append(r)
                except Exception as e:
                    results.append(AnalysisResult(category=futures[f], score=50.0,
                                                  grade="C", summary="분석 실패",
                                                  detailed_analysis=str(e),
                                                  key_strengths=[], key_weaknesses=[],
                                                  supporting_evidence=[]))

        print("\n" + "="*80)
        print("최종 평가 결과:")
        for r in results:
            print(f"  {r.category}: {r.score:.1f}점 ({r.grade}급)")
        print("="*80 + "\n")

        return results

    def _calculate_grade(self, score: float) -> str:
        from config import get_config
        th = get_config()["scoring"].grade_thresholds
        if score >= th["S"]: return "S"
        elif score >= th["A"]: return "A"
        elif score >= th["B"]: return "B"
        elif score >= th["C"]: return "C"
        return "D"

def process_analysis_engine(context: PipelineContext) -> PipelineContext:
    engine = AnalysisEngine()
    eval_type = context.parsed_input.evaluation_type
    selected = list(engine.analyzers.keys()) if eval_type.value=="전체 평가" else ["growth_analysis"]
    context.analysis_results = engine.run_parallel_analysis(
        company_info=context.company_info,
        documents=context.retrieved_documents,
        external_results=context.external_search_results,
        selected=selected
    )
    context.processing_steps.append(f"ANALYSIS_ENGINE: {len(context.analysis_results)}개 분석 완료")
    return context
