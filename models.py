"""
Data models for AI Startup Investment Evaluation Agent
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum

class EvaluationType(Enum):
    FULL_EVALUATION = "전체 평가"
    GROWTH_ANALYSIS = "성장성 분석"
    FINANCIAL_ANALYSIS = "재무 분석"
    TECH_ANALYSIS = "기술 분석"
    RISK_ANALYSIS = "리스크 분석"

class InvestmentRecommendation(Enum):
    INVEST = "투자 추천"
    HOLD = "보류"
    AVOID = "회피"

class RiskLevel(Enum):
    LOW = "낮음"
    MEDIUM = "보통"
    HIGH = "높음"
    CRITICAL = "매우 높음"

@dataclass
class CompanyInfo:
    """회사 기본 정보"""
    name: str
    industry: str = ""
    founded_year: Optional[int] = None
    headquarters: str = ""
    employee_count: Optional[int] = None
    website: str = ""
    description: str = ""

@dataclass
class ParsedInput:
    """파싱된 입력 정보"""
    company_name: str
    evaluation_type: EvaluationType
    specific_focus_areas: List[str] = field(default_factory=list)
    additional_requirements: str = ""

@dataclass
class DocumentChunk:
    """검색된 문서 청크"""
    content: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    similarity_score: float = 0.0

@dataclass
class AnalysisResult:
    """개별 분석 결과"""
    category: str
    score: float  # 0-100
    grade: str    # S, A, B, C, D
    summary: str
    detailed_analysis: str
    key_strengths: List[str] = field(default_factory=list)
    key_weaknesses: List[str] = field(default_factory=list)
    supporting_evidence: List[str] = field(default_factory=list)

@dataclass
class RiskAssessment:
    """리스크 평가 결과"""
    category: str
    risk_level: RiskLevel
    description: str
    impact_score: float  # 0-10
    probability: float   # 0-1
    mitigation_strategies: List[str] = field(default_factory=list)

@dataclass
class UnicornScore:
    """유니콘 점수 정보"""
    total_score: float
    grade: str
    unicorn_probability: float
    category_scores: Dict[str, float] = field(default_factory=dict)
    score_breakdown: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InvestmentReport:
    """최종 투자 평가 리포트"""
    company_info: CompanyInfo
    evaluation_date: datetime
    unicorn_score: UnicornScore
    recommendation: InvestmentRecommendation

    # 상세 분석 결과
    analysis_results: List[AnalysisResult] = field(default_factory=list)
    risk_assessments: List[RiskAssessment] = field(default_factory=list)

    # 리포트 섹션
    executive_summary: str = ""
    detailed_analysis: str = ""
    investment_rationale: str = ""
    risk_summary: str = ""

    # 메타데이터
    confidence_level: float = 0.0
    data_sources: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)

@dataclass
class ExternalSearchResult:
    """외부 검색 결과"""
    title: str
    content: str
    source: str
    url: str
    published_date: Optional[datetime] = None
    relevance_score: float = 0.0

@dataclass
class QualityCheckResult:
    """품질 검증 결과"""
    relevance_score: float    # 관련성 점수
    evidence_quality: float   # 근거 품질 점수
    objectivity_score: float  # 객관성 점수
    overall_quality: float    # 전체 품질 점수
    passed: bool              # 품질 검증 통과 여부
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

@dataclass
class PipelineContext:
    """파이프라인 실행 컨텍스트"""
    parsed_input: ParsedInput
    company_info: CompanyInfo
    retrieved_documents: List[DocumentChunk] = field(default_factory=list)
    external_search_results: List[ExternalSearchResult] = field(default_factory=list)
    analysis_results: List[AnalysisResult] = field(default_factory=list)
    risk_assessments: List[RiskAssessment] = field(default_factory=list)
    unicorn_score: Optional[UnicornScore] = None
    final_report: Optional[InvestmentReport] = None
    quality_check: Optional[QualityCheckResult] = None

    # 실행 메타데이터
    execution_start_time: datetime = field(default_factory=datetime.now)
    execution_end_time: Optional[datetime] = None
    processing_steps: List[str] = field(default_factory=list)