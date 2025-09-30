"""
Layer 3: DOCUMENT RETRIEVAL LAYER
company_document_retriever 실행하여 관련 문서 청크를 추출하는 레이어
"""
import re
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from models import DocumentChunk, PipelineContext, EvaluationType
from config import get_config

class CompanyDocumentRetriever:
    """회사별 문서 검색 및 필터링"""

    def __init__(self):
        self.config = get_config()
        self.llm = ChatOpenAI(
            openai_api_key=self.config["model"].openai_api_key,
            temperature=0.1,
            model_name="gpt-3.5-turbo-instruct"
        )

        # 문서 관련성 평가 프롬프트
        self.relevance_prompt = PromptTemplate(
            input_variables=["company_name", "evaluation_type", "document_content"],
            template="""다음 문서가 {company_name}의 {evaluation_type}에 얼마나 관련이 있는지 0-10점으로 평가해주세요.

회사명: {company_name}
평가 유형: {evaluation_type}

문서 내용:
{document_content}

다음 JSON 형식으로 응답해주세요:
{{
    "relevance_score": 8.5,
    "key_points": ["핵심 포인트 1", "핵심 포인트 2"],
    "section_type": "재무정보|시장분석|기술정보|팀정보|기타",
    "reasoning": "관련성 평가 이유"
}}"""
        )

    def filter_documents_by_company(
        self,
        documents: List[DocumentChunk],
        company_name: str
    ) -> List[DocumentChunk]:
        """회사명으로 문서 필터링"""
        filtered_docs = []

        # 회사명 변형들 생성
        company_variations = self._generate_company_variations(company_name)

        for doc in documents:
            # 메타데이터에서 회사명 확인
            if self._check_company_in_metadata(doc.metadata, company_variations):
                filtered_docs.append(doc)
                continue

            # 문서 내용에서 회사명 확인
            if self._check_company_in_content(doc.content, company_variations):
                filtered_docs.append(doc)
                continue

        return filtered_docs

    def _generate_company_variations(self, company_name: str) -> List[str]:
        """회사명의 다양한 변형 생성"""
        variations = [company_name]

        # 기본 변형
        variations.extend([
            company_name + "㈜",
            company_name + " 주식회사",
            company_name + "코퍼레이션",
            company_name + " Corp",
            company_name + " Inc",
            "㈜" + company_name
        ])

        # 영문/한글 변환 (예시)
        company_mappings = {
            "토스": ["Toss", "비바리퍼블리카", "Viva Republica"],
            "카카오": ["Kakao", "Daum Kakao"],
            "배달의민족": ["배민", "우아한형제들", "Woowa Brothers"],
            "네이버": ["Naver", "NHN"],
            "쿠팡": ["Coupang"],
            "당근마켓": ["당근", "Daangn"]
        }

        if company_name in company_mappings:
            variations.extend(company_mappings[company_name])

        return list(set(variations))  # 중복 제거

    def _check_company_in_metadata(self, metadata: Dict[str, Any], variations: List[str]) -> bool:
        """메타데이터에서 회사명 확인"""
        metadata_text = " ".join([str(v) for v in metadata.values()]).lower()

        for variation in variations:
            if variation.lower() in metadata_text:
                return True
        return False

    def _check_company_in_content(self, content: str, variations: List[str]) -> bool:
        """문서 내용에서 회사명 확인"""
        content_lower = content.lower()

        for variation in variations:
            if variation.lower() in content_lower:
                return True
        return False

    def filter_by_section_type(
        self,
        documents: List[DocumentChunk],
        evaluation_type: EvaluationType
    ) -> List[DocumentChunk]:
        """평가 유형에 따른 섹션별 필터링"""

        section_keywords = {
            EvaluationType.FULL_EVALUATION: [],  # 모든 섹션
            EvaluationType.GROWTH_ANALYSIS: [
                "성장", "매출", "시장점유율", "확장", "성장률", "증가율",
                "growth", "revenue", "expansion", "scale"
            ],
            EvaluationType.FINANCIAL_ANALYSIS: [
                "재무", "손익", "자산", "부채", "현금", "투자", "수익",
                "financial", "profit", "asset", "debt", "cash", "investment"
            ],
            EvaluationType.TECH_ANALYSIS: [
                "기술", "개발", "특허", "R&D", "혁신", "플랫폼", "시스템",
                "technology", "development", "patent", "innovation", "platform"
            ],
            EvaluationType.RISK_ANALYSIS: [
                "리스크", "위험", "규제", "경쟁", "위기", "문제",
                "risk", "regulation", "competition", "crisis", "issue"
            ]
        }

        keywords = section_keywords.get(evaluation_type, [])
        if not keywords:  # 전체 평가인 경우 모든 문서 반환
            return documents

        filtered_docs = []
        for doc in documents:
            content_lower = doc.content.lower()
            if any(keyword.lower() in content_lower for keyword in keywords):
                filtered_docs.append(doc)

        return filtered_docs

    def rank_documents_by_relevance(
        self,
        documents: List[DocumentChunk],
        company_name: str,
        evaluation_type: EvaluationType,
        top_k: int = None
    ) -> List[DocumentChunk]:
        """문서를 관련성으로 순위 매기기"""
        if not documents:
            return []

        top_k = top_k or self.config["vector_db"].top_k_results

        # 각 문서에 대해 관련성 점수 계산
        scored_documents = []
        for doc in documents[:20]:  # LLM 호출 제한을 위해 상위 20개만
            try:
                relevance_data = self._evaluate_document_relevance(
                    doc, company_name, evaluation_type
                )
                doc.metadata.update(relevance_data)
                scored_documents.append(doc)
            except Exception as e:
                # LLM 호출 실패 시 기본 점수 사용
                doc.metadata["relevance_score"] = doc.similarity_score * 10
                scored_documents.append(doc)

        # 나머지 문서들은 similarity_score 기반으로 추가
        for doc in documents[20:]:
            doc.metadata["relevance_score"] = doc.similarity_score * 10
            scored_documents.append(doc)

        # 관련성 점수로 정렬
        scored_documents.sort(
            key=lambda x: x.metadata.get("relevance_score", 0),
            reverse=True
        )

        return scored_documents[:top_k]

    def _evaluate_document_relevance(
        self,
        document: DocumentChunk,
        company_name: str,
        evaluation_type: EvaluationType
    ) -> Dict[str, Any]:
        """LLM을 사용하여 문서 관련성 평가"""
        try:
            # 문서 내용 제한 (토큰 제한 고려)
            content_preview = document.content[:1000]

            response = self.llm(self.relevance_prompt.format(
                company_name=company_name,
                evaluation_type=evaluation_type.value,
                document_content=content_preview
            ))

            import json
            relevance_data = json.loads(response.strip())
            return relevance_data

        except Exception as e:
            # JSON 파싱 실패 시 기본값 반환
            return {
                "relevance_score": 5.0,
                "key_points": [],
                "section_type": "기타",
                "reasoning": "자동 평가 실패"
            }

    def extract_top_k_chunks(
        self,
        documents: List[DocumentChunk],
        company_name: str,
        evaluation_type: EvaluationType,
        k: int = None
    ) -> List[DocumentChunk]:
        """최종 Top-K 문서 청크 추출"""
        if not documents:
            return []

        k = k or self.config["vector_db"].top_k_results

        # 1단계: 회사명으로 필터링
        company_filtered = self.filter_documents_by_company(documents, company_name)

        # 2단계: 섹션 타입으로 필터링
        section_filtered = self.filter_by_section_type(company_filtered, evaluation_type)

        # 필터링 결과가 없으면 원본 문서에서 상위 문서 사용
        if not section_filtered:
            section_filtered = documents

        # 3단계: 관련성으로 순위 매기고 Top-K 선택
        ranked_documents = self.rank_documents_by_relevance(
            section_filtered, company_name, evaluation_type, k
        )

        return ranked_documents

def create_document_retrieval_layer() -> CompanyDocumentRetriever:
    """Document Retrieval Layer 생성자"""
    return CompanyDocumentRetriever()

def process_document_retrieval_layer(context: PipelineContext) -> PipelineContext:
    """Document Retrieval Layer 처리 함수"""
    retriever = create_document_retrieval_layer()

    # 기존 검색된 문서들에서 Top-K 추출
    if context.retrieved_documents:
        top_k_documents = retriever.extract_top_k_chunks(
            documents=context.retrieved_documents,
            company_name=context.company_info.name,
            evaluation_type=context.parsed_input.evaluation_type
        )

        # 컨텍스트 업데이트
        context.retrieved_documents = top_k_documents

        # 처리 단계 기록
        context.processing_steps.append(
            f"DOCUMENT_RETRIEVAL_LAYER: {len(top_k_documents)}개 Top-K 문서 추출 완료"
        )
    else:
        context.processing_steps.append(
            "DOCUMENT_RETRIEVAL_LAYER: 검색된 문서가 없음"
        )

    return context