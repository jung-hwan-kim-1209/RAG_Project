"""
Layer 1: INPUT LAYER
입력된 회사명을 기반으로 기업명과 평가 유형을 추출하는 레이어
"""
import re
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ParsedInput, EvaluationType, CompanyInfo, PipelineContext
from config import get_config

class InputParser:
    """사용자 입력을 파싱하여 기업명과 평가 유형을 추출"""

    def __init__(self):
        self.config = get_config()
        self.llm = ChatOpenAI(
            openai_api_key=self.config["model"].openai_api_key,
            temperature=0.1,
            model=self.config["model"].model_name  # 예: "gpt-4o-mini"
        )

        self.parsing_prompt = PromptTemplate(
            input_variables=["user_input"],
            template="""다음 사용자 입력에서 기업명과 평가 유형을 추출해주세요.

평가 유형:
- 전체 평가: 모든 영역을 종합적으로 평가
- 성장성 분석: 성장 잠재력에 집중
- 재무 분석: 재무 건전성에 집중
- 기술 분석: 기술력과 보안성에 집중
- 리스크 분석: 위험 요소에 집중

사용자 입력: {user_input}

다음 JSON 형식으로 응답해주세요:
{{
    "company_name": "추출된 기업명",
    "evaluation_type": "평가 유형",
    "specific_focus_areas": ["특정 관심 영역들"],
    "additional_requirements": "추가 요구사항"
}}"""
        )

    def parse(self, user_input: str) -> ParsedInput:
        """사용자 입력을 파싱하여 구조화된 데이터로 변환"""
        try:
            # LLM을 통한 입력 파싱
            response = self.llm.invoke(self.parsing_prompt.format(user_input=user_input))

            # GPT 응답을 터미널에 출력
            print(f"\n🔍 INPUT_LAYER - GPT 응답:")
            print("=" * 60)
            print(response.content)
            print("=" * 60)

            # JSON 응답 파싱
            import json
            parsed_data = json.loads(response.content.strip())

            # 평가 유형 매핑
            evaluation_type_map = {
                "전체 평가": EvaluationType.FULL_EVALUATION,
                "성장성 분석": EvaluationType.GROWTH_ANALYSIS,
                "재무 분석": EvaluationType.FINANCIAL_ANALYSIS,
                "기술 분석": EvaluationType.TECH_ANALYSIS,
                "리스크 분석": EvaluationType.RISK_ANALYSIS
            }

            evaluation_type = evaluation_type_map.get(
                parsed_data.get("evaluation_type", "전체 평가"),
                EvaluationType.FULL_EVALUATION
            )

            return ParsedInput(
                company_name=parsed_data.get("company_name", "").strip(),
                evaluation_type=evaluation_type,
                specific_focus_areas=parsed_data.get("specific_focus_areas", []),
                additional_requirements=parsed_data.get("additional_requirements", "")
            )

        except Exception as e:
            # 파싱 실패 시 간단한 규칙 기반 파싱
            return self._fallback_parsing(user_input)

    def _fallback_parsing(self, user_input: str) -> ParsedInput:
        """LLM 파싱 실패 시 규칙 기반 백업 파싱"""

        # 기업명 추출 패턴 (더 구체적인 패턴부터 우선순위 적용)
        company_patterns = [
            r'^([가-힣A-Za-z0-9]+)(?:의|을|를)?\s*(?:투자|평가|분석)',
            r'^([가-힣A-Za-z0-9]+)\s*(?:회사|기업)',
            r'^([가-힣A-Za-z0-9]+)(?:\s|$)'
        ]

        company_name = ""
        for pattern in company_patterns:
            match = re.search(pattern, user_input)
            if match:
                company_name = match.group(1)
                break

        # 평가 유형 결정
        evaluation_type = EvaluationType.FULL_EVALUATION
        if "성장" in user_input:
            evaluation_type = EvaluationType.GROWTH_ANALYSIS
        elif "재무" in user_input or "금융" in user_input:
            evaluation_type = EvaluationType.FINANCIAL_ANALYSIS
        elif "기술" in user_input or "보안" in user_input:
            evaluation_type = EvaluationType.TECH_ANALYSIS
        elif "리스크" in user_input or "위험" in user_input:
            evaluation_type = EvaluationType.RISK_ANALYSIS

        return ParsedInput(
            company_name=company_name,
            evaluation_type=evaluation_type,
            specific_focus_areas=[],
            additional_requirements=""
        )

    def extract_company_info(self, company_name: str) -> CompanyInfo:
        """기업명을 기반으로 기본 회사 정보 추출 (추후 외부 API 연동)"""

        # 기본 정보 생성 (실제로는 외부 데이터베이스나 API에서 조회)
        company_info = CompanyInfo(
            name=company_name,
            industry="",
            founded_year=None,
            headquarters="",
            employee_count=None,
            website="",
            description=""
        )

        # 알려진 기업들에 대한 기본 정보 (예시)
        known_companies = {
            "토스": CompanyInfo(
                name="토스",
                industry="핀테크",
                founded_year=2013,
                headquarters="서울",
                description="간편송금 및 금융서비스 플랫폼"
            ),
            "배달의민족": CompanyInfo(
                name="배달의민족",
                industry="O2O/배달",
                founded_year=2010,
                headquarters="서울",
                description="음식 배달 중개 플랫폼"
            ),
            "카카오": CompanyInfo(
                name="카카오",
                industry="IT/플랫폼",
                founded_year=1995,
                headquarters="제주",
                description="메신저 및 플랫폼 서비스"
            )
        }

        if company_name in known_companies:
            return known_companies[company_name]

        return company_info

def create_input_layer() -> InputParser:
    """Input Layer 생성자"""
    return InputParser()

def process_input_layer(user_input: str, context: PipelineContext) -> PipelineContext:
    """Input Layer 처리 함수"""
    parser = create_input_layer()

    # 사용자 입력 파싱
    parsed_input = parser.parse(user_input)
    context.parsed_input = parsed_input

    # 회사 정보 추출
    company_info = parser.extract_company_info(parsed_input.company_name)
    context.company_info = company_info

    # 처리 단계 기록
    context.processing_steps.append("INPUT_LAYER: 입력 파싱 및 기업 정보 추출 완료")

    return context