"""
Layer 5: ANALYSIS ENGINE
7개 분석 영역을 병렬로 실행하는 레이어
"""
import asyncio
import os
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AnalysisResult, DocumentChunk, ExternalSearchResult, PipelineContext, CompanyInfo, GPTResponse
from config import get_config

class BaseAnalyzer:
    """분석기 기본 클래스"""

    def __init__(self, analyzer_name: str):
        self.analyzer_name = analyzer_name
        self.config = get_config()
        self.llm = ChatOpenAI(
            openai_api_key=self.config["model"].openai_api_key,
            temperature=0.1,
            model=self.config["model"].model_name  # 예: "gpt-4o-mini"
        )

    def analyze(
        self,
        company_info: CompanyInfo,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        context: PipelineContext = None
    ) -> AnalysisResult:
        """분석 실행 (하위 클래스에서 구현)"""
        raise NotImplementedError

    def _create_context_summary(
        self,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult]
    ) -> str:
        """문서와 외부 검색 결과를 요약하여 컨텍스트 생성"""
        context_parts = []

        # 상위 문서들 요약
        if documents:
            doc_summaries = []
            for doc in documents[:5]:  # 상위 5개 문서만
                doc_summaries.append(f"- {doc.content[:200]}...")
            context_parts.append("관련 문서 정보:\n" + "\n".join(doc_summaries))

        # 외부 검색 결과 요약
        if external_results:
            external_summaries = []
            for result in external_results[:3]:  # 상위 3개 결과만
                external_summaries.append(f"- {result.title}: {result.content[:150]}...")
            context_parts.append("최신 정보:\n" + "\n".join(external_summaries))

        return "\n\n".join(context_parts)

    def _calculate_grade(self, score: float) -> str:
        """점수를 등급으로 변환"""
        grade_thresholds = self.config["scoring"].grade_thresholds

        if score >= grade_thresholds["S"]:
            return "S"
        elif score >= grade_thresholds["A"]:
            return "A"
        elif score >= grade_thresholds["B"]:
            return "B"
        elif score >= grade_thresholds["C"]:
            return "C"
        else:
            return "D"

class GrowthAnalyzer(BaseAnalyzer):
    """성장성 분석기"""

    def __init__(self):
        super().__init__("growth_analysis")
        self.analysis_prompt = PromptTemplate(
            input_variables=["company_name", "company_info", "context"],
            template="""다음 정보를 바탕으로 {company_name}의 성장성을 분석해주세요.

회사 정보:
{company_info}

관련 자료:
{context}

다음 항목들을 중점적으로 분석하고 0-100점으로 점수를 매겨주세요:
1. 매출 성장률
2. 시장 확장 가능성
3. 고객 증가율
4. 제품/서비스 확장성
5. 시장 점유율 증가 잠재력

JSON 형식으로 응답해주세요:
{{
    "score": 85,
    "summary": "성장성 요약",
    "detailed_analysis": "상세 분석 내용",
    "key_strengths": ["강점1", "강점2"],
    "key_weaknesses": ["약점1", "약점2"],
    "supporting_evidence": ["근거1", "근거2"]
}}"""
        )

    def analyze(
        self,
        company_info: CompanyInfo,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        context: PipelineContext = None
    ) -> AnalysisResult:
        """성장성 분석 실행"""
        context_summary = self._create_context_summary(documents, external_results)
        company_info_text = f"업종: {company_info.industry}, 설립년도: {company_info.founded_year}, 본사: {company_info.headquarters}"

        try:
            formatted_prompt = self.analysis_prompt.format(
                company_name=company_info.name,
                company_info=company_info_text,
                context=context_summary
            )
            
            response = self.llm.invoke(formatted_prompt)

            # GPT 응답을 터미널에 출력
            print(f"\n🔍 {self.analyzer_name.upper()} - GPT 응답:")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

            # GPT 응답을 컨텍스트에 저장
            if context:
                gpt_response = GPTResponse(
                    layer_name="ANALYSIS_ENGINE",
                    analyzer_name=self.analyzer_name,
                    prompt=formatted_prompt,
                    response=response.content
                )
                context.gpt_responses.append(gpt_response)

            import json
            result_data = json.loads(response.content.strip())

            return AnalysisResult(
                category=self.analyzer_name,
                score=result_data.get("score", 50.0),
                grade=self._calculate_grade(result_data.get("score", 50.0)),
                summary=result_data.get("summary", ""),
                detailed_analysis=result_data.get("detailed_analysis", ""),
                key_strengths=result_data.get("key_strengths", []),
                key_weaknesses=result_data.get("key_weaknesses", []),
                supporting_evidence=result_data.get("supporting_evidence", [])
            )

        except Exception as e:
            # 분석 실패 시 기본값 반환
            return AnalysisResult(
                category=self.analyzer_name,
                score=50.0,
                grade="C",
                summary="성장성 분석 오류",
                detailed_analysis=f"분석 중 오류 발생: {str(e)}",
                key_strengths=[],
                key_weaknesses=[],
                supporting_evidence=[]
            )

class BusinessModelAnalyzer(BaseAnalyzer):
    """비즈니스 모델 분석기"""

    def __init__(self):
        super().__init__("business_model_analysis")
        self.analysis_prompt = PromptTemplate(
            input_variables=["company_name", "company_info", "context"],
            template="""다음 정보를 바탕으로 {company_name}의 비즈니스 모델을 분석해주세요.

회사 정보:
{company_info}

관련 자료:
{context}

다음 항목들을 중점적으로 분석하고 0-100점으로 점수를 매겨주세요:
1. 수익 모델의 지속가능성
2. 고객 획득 비용 vs 고객 생애 가치
3. 시장 진입 장벽
4. 경쟁 우위 요소
5. 수익화 구조의 명확성

JSON 형식으로 응답해주세요:
{{
    "score": 75,
    "summary": "비즈니스 모델 요약",
    "detailed_analysis": "상세 분석 내용",
    "key_strengths": ["강점1", "강점2"],
    "key_weaknesses": ["약점1", "약점2"],
    "supporting_evidence": ["근거1", "근거2"]
}}"""
        )

    def analyze(
        self,
        company_info: CompanyInfo,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        context: PipelineContext = None
    ) -> AnalysisResult:
        """비즈니스 모델 분석 실행"""
        context_summary = self._create_context_summary(documents, external_results)
        company_info_text = f"업종: {company_info.industry}, 설명: {company_info.description}"

        try:
            formatted_prompt = self.analysis_prompt.format(
                company_name=company_info.name,
                company_info=company_info_text,
                context=context_summary
            )
            
            response = self.llm.invoke(formatted_prompt)

            # GPT 응답을 터미널에 출력
            print(f"\n🔍 {self.analyzer_name.upper()} - GPT 응답:")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

            # GPT 응답을 컨텍스트에 저장
            if context:
                gpt_response = GPTResponse(
                    layer_name="ANALYSIS_ENGINE",
                    analyzer_name=self.analyzer_name,
                    prompt=formatted_prompt,
                    response=response.content
                )
                context.gpt_responses.append(gpt_response)

            import json
            result_data = json.loads(response.content.strip())

            return AnalysisResult(
                category=self.analyzer_name,
                score=result_data.get("score", 50.0),
                grade=self._calculate_grade(result_data.get("score", 50.0)),
                summary=result_data.get("summary", ""),
                detailed_analysis=result_data.get("detailed_analysis", ""),
                key_strengths=result_data.get("key_strengths", []),
                key_weaknesses=result_data.get("key_weaknesses", []),
                supporting_evidence=result_data.get("supporting_evidence", [])
            )

        except Exception as e:
            return AnalysisResult(
                category=self.analyzer_name,
                score=50.0,
                grade="C",
                summary="비즈니스 모델 분석 오류",
                detailed_analysis=f"분석 중 오류 발생: {str(e)}",
                key_strengths=[],
                key_weaknesses=[],
                supporting_evidence=[]
            )

class TechSecurityAnalyzer(BaseAnalyzer):
    """기술력/보안성 분석기"""

    def __init__(self):
        super().__init__("tech_security_analysis")
        self.analysis_prompt = PromptTemplate(
            input_variables=["company_name", "company_info", "context"],
            template="""다음 정보를 바탕으로 {company_name}의 기술력과 보안성을 분석해주세요.

회사 정보:
{company_info}

관련 자료:
{context}

다음 항목들을 중점적으로 분석하고 0-100점으로 점수를 매겨주세요:
1. 핵심 기술의 차별성
2. 특허 및 지적재산권
3. 개발팀의 기술 역량
4. 보안 체계 및 데이터 보호
5. 기술 혁신성 및 미래 대응력

JSON 형식으로 응답해주세요:
{{
    "score": 80,
    "summary": "기술력/보안성 요약",
    "detailed_analysis": "상세 분석 내용",
    "key_strengths": ["강점1", "강점2"],
    "key_weaknesses": ["약점1", "약점2"],
    "supporting_evidence": ["근거1", "근거2"]
}}"""
        )

    def analyze(
        self,
        company_info: CompanyInfo,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        context: PipelineContext = None
    ) -> AnalysisResult:
        """기술력/보안성 분석 실행"""
        context_summary = self._create_context_summary(documents, external_results)
        company_info_text = f"업종: {company_info.industry}, 설명: {company_info.description}"

        try:
            formatted_prompt = self.analysis_prompt.format(
                company_name=company_info.name,
                company_info=company_info_text,
                context=context_summary
            )
            
            response = self.llm.invoke(formatted_prompt)

            # GPT 응답을 터미널에 출력
            print(f"\n🔍 {self.analyzer_name.upper()} - GPT 응답:")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

            # GPT 응답을 컨텍스트에 저장
            if context:
                gpt_response = GPTResponse(
                    layer_name="ANALYSIS_ENGINE",
                    analyzer_name=self.analyzer_name,
                    prompt=formatted_prompt,
                    response=response.content
                )
                context.gpt_responses.append(gpt_response)

            import json
            result_data = json.loads(response.content.strip())

            return AnalysisResult(
                category=self.analyzer_name,
                score=result_data.get("score", 50.0),
                grade=self._calculate_grade(result_data.get("score", 50.0)),
                summary=result_data.get("summary", ""),
                detailed_analysis=result_data.get("detailed_analysis", ""),
                key_strengths=result_data.get("key_strengths", []),
                key_weaknesses=result_data.get("key_weaknesses", []),
                supporting_evidence=result_data.get("supporting_evidence", [])
            )

        except Exception as e:
            return AnalysisResult(
                category=self.analyzer_name,
                score=50.0,
                grade="C",
                summary="기술력/보안성 분석 오류",
                detailed_analysis=f"분석 중 오류 발생: {str(e)}",
                key_strengths=[],
                key_weaknesses=[],
                supporting_evidence=[]
            )

class FinancialHealthAnalyzer(BaseAnalyzer):
    """재무건전성 분석기"""

    def __init__(self):
        super().__init__("financial_health_analysis")
        self.analysis_prompt = PromptTemplate(
            input_variables=["company_name", "company_info", "context"],
            template="""다음 정보를 바탕으로 {company_name}의 재무건전성을 분석해주세요.

회사 정보:
{company_info}

관련 자료:
{context}

다음 항목들을 중점적으로 분석하고 0-100점으로 점수를 매겨주세요:
1. 현금 보유 현황 및 운영 자금
2. 매출 성장률 및 수익성
3. 투자 유치 이력 및 밸류에이션
4. 비용 구조 및 효율성
5. 재무 리스크 요소

JSON 형식으로 응답해주세요:
{{
    "score": 70,
    "summary": "재무건전성 요약",
    "detailed_analysis": "상세 분석 내용",
    "key_strengths": ["강점1", "강점2"],
    "key_weaknesses": ["약점1", "약점2"],
    "supporting_evidence": ["근거1", "근거2"]
}}"""
        )

    def analyze(
        self,
        company_info: CompanyInfo,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        context: PipelineContext = None
    ) -> AnalysisResult:
        """재무건전성 분석 실행"""
        context_summary = self._create_context_summary(documents, external_results)
        company_info_text = f"업종: {company_info.industry}, 설립년도: {company_info.founded_year}"

        try:
            formatted_prompt = self.analysis_prompt.format(
                company_name=company_info.name,
                company_info=company_info_text,
                context=context_summary
            )
            
            response = self.llm.invoke(formatted_prompt)

            # GPT 응답을 터미널에 출력
            print(f"\n🔍 {self.analyzer_name.upper()} - GPT 응답:")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

            # GPT 응답을 컨텍스트에 저장
            if context:
                gpt_response = GPTResponse(
                    layer_name="ANALYSIS_ENGINE",
                    analyzer_name=self.analyzer_name,
                    prompt=formatted_prompt,
                    response=response.content
                )
                context.gpt_responses.append(gpt_response)

            import json
            result_data = json.loads(response.content.strip())

            return AnalysisResult(
                category=self.analyzer_name,
                score=result_data.get("score", 50.0),
                grade=self._calculate_grade(result_data.get("score", 50.0)),
                summary=result_data.get("summary", ""),
                detailed_analysis=result_data.get("detailed_analysis", ""),
                key_strengths=result_data.get("key_strengths", []),
                key_weaknesses=result_data.get("key_weaknesses", []),
                supporting_evidence=result_data.get("supporting_evidence", [])
            )

        except Exception as e:
            return AnalysisResult(
                category=self.analyzer_name,
                score=50.0,
                grade="C",
                summary="재무건전성 분석 오류",
                detailed_analysis=f"분석 중 오류 발생: {str(e)}",
                key_strengths=[],
                key_weaknesses=[],
                supporting_evidence=[]
            )




class AnalysisEngine:
    """분석 엔진 메인 클래스"""

    def __init__(self):
        self.analyzers = {
            "growth_analysis": GrowthAnalyzer(),
            "business_model_analysis": BusinessModelAnalyzer(),
            "tech_security_analysis": TechSecurityAnalyzer(),
            "financial_health_analysis": FinancialHealthAnalyzer()
        }

    def run_parallel_analysis(
        self,
        company_info: CompanyInfo,
        documents: List[DocumentChunk],
        external_results: List[ExternalSearchResult],
        selected_analyses: List[str] = None,
        context: PipelineContext = None
    ) -> List[AnalysisResult]:
        """병렬로 분석 실행"""

        if selected_analyses is None:
            selected_analyses = list(self.analyzers.keys())

        # 선택된 분석기들만 실행
        selected_analyzers = {
            name: analyzer for name, analyzer in self.analyzers.items()
            if name in selected_analyses
        }

        # ThreadPoolExecutor를 사용한 병렬 실행
        with ThreadPoolExecutor(max_workers=len(selected_analyzers)) as executor:
            future_to_analyzer = {
                executor.submit(
                    analyzer.analyze, company_info, documents, external_results, context
                ): name
                for name, analyzer in selected_analyzers.items()
            }

            results = []
            for future in future_to_analyzer:
                try:
                    timeout_seconds = int(os.getenv("ANALYSIS_TIMEOUT_SECONDS", "60"))
                    result = future.result(timeout=timeout_seconds)
                    results.append(result)
                except Exception as e:
                    analyzer_name = future_to_analyzer[future]
                    error_result = AnalysisResult(
                        category=analyzer_name,
                        score=0.0,
                        grade="D",
                        summary=f"{analyzer_name} 분석 실패",
                        detailed_analysis=f"분석 중 오류 발생: {str(e)}",
                        key_strengths=[],
                        key_weaknesses=[],
                        supporting_evidence=[]
                    )
                    results.append(error_result)

        return results

def create_analysis_engine() -> AnalysisEngine:
    """Analysis Engine 생성자"""
    return AnalysisEngine()

def process_analysis_engine(context: PipelineContext) -> PipelineContext:
    """Analysis Engine 처리 함수"""
    analysis_engine = create_analysis_engine()

    # 평가 유형에 따른 분석 선택
    evaluation_type = context.parsed_input.evaluation_type
    selected_analyses = []

    # 전체 평가인 경우 모든 분석 실행
    if evaluation_type.value == "전체 평가":
        selected_analyses = list(analysis_engine.analyzers.keys())
    else:
        # 특정 평가 유형에 따른 분석 선택
        analysis_mapping = {
            "성장성 분석": ["growth_analysis", "business_model_analysis"],
            "재무 분석": ["financial_health_analysis", "growth_analysis"],
            "기술 분석": ["tech_security_analysis", "business_model_analysis"],
            "리스크 분석": ["financial_health_analysis", "tech_security_analysis"]
        }
        selected_analyses = analysis_mapping.get(evaluation_type.value, ["growth_analysis"])

    # 병렬 분석 실행
    analysis_results = analysis_engine.run_parallel_analysis(
        company_info=context.company_info,
        documents=context.retrieved_documents,
        external_results=context.external_search_results,
        selected_analyses=selected_analyses,
        context=context
    )

    context.analysis_results = analysis_results

    # 처리 단계 기록
    context.processing_steps.append(
        f"ANALYSIS_ENGINE: {len(analysis_results)}개 분석 완료 (병렬 실행) - 4개 분석기 사용"
    )

    return context