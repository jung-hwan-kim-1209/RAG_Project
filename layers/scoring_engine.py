"""
Layer 6: SCORING & RANKING ENGINE
unicorn_score_calculator를 실행하여 총점, 등급, 유니콘 확률을 계산하는 레이어
"""
import math
from typing import List, Dict, Any, Optional
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate

from models import AnalysisResult, UnicornScore, PipelineContext, CompanyInfo
from config import get_config

class UnicornScoreCalculator:
    """유니콘 점수 계산기"""

    def __init__(self):
        self.config = get_config()
        self.analysis_weights = self.config["analysis_weights"]
        self.scoring_config = self.config["scoring"]
        self.llm = OpenAI(
            openai_api_key=self.config["model"].openai_api_key,
            temperature=0.1,
            model_name="gpt-3.5-turbo-instruct"
        )

        # 유니콘 확률 계산 프롬프트
        self.unicorn_probability_prompt = PromptTemplate(
            input_variables=["company_name", "total_score", "category_scores", "industry"],
            template="""다음 정보를 바탕으로 {company_name}이 유니콘 기업(기업가치 10억 달러 이상)이 될 확률을 분석해주세요.

회사명: {company_name}
업종: {industry}
종합 점수: {total_score}/100
영역별 점수:
{category_scores}

다음 요소들을 고려하여 유니콘 확률을 0-1 사이의 값으로 계산해주세요:
1. 시장 규모와 성장 잠재력
2. 경쟁 우위 및 차별화 요소
3. 팀 역량과 실행력
4. 재무 건전성 및 수익성
5. 업계 트렌드 및 타이밍

JSON 형식으로 응답해주세요:
{{
    "unicorn_probability": 0.65,
    "key_factors": ["주요 요인 1", "주요 요인 2"],
    "reasoning": "확률 산정 근거",
    "comparable_companies": ["비교 기업 1", "비교 기업 2"]
}}"""
        )

    def calculate_weighted_score(self, analysis_results: List[AnalysisResult]) -> Dict[str, float]:
        """가중치가 적용된 점수 계산"""
        category_scores = {}
        weighted_scores = {}

        # 각 분석 결과를 카테고리별로 정리
        for result in analysis_results:
            category_scores[result.category] = result.score

        # 가중치 적용
        weighted_scores["growth_analysis"] = (
            category_scores.get("growth_analysis", 0) * self.analysis_weights.growth_weight
        )
        weighted_scores["business_model_analysis"] = (
            category_scores.get("business_model_analysis", 0) * self.analysis_weights.business_model_weight
        )
        weighted_scores["tech_security_analysis"] = (
            category_scores.get("tech_security_analysis", 0) * self.analysis_weights.tech_security_weight
        )
        weighted_scores["financial_health_analysis"] = (
            category_scores.get("financial_health_analysis", 0) * self.analysis_weights.financial_health_weight
        )
        weighted_scores["team_evaluation"] = (
            category_scores.get("team_evaluation", 0) * self.analysis_weights.team_weight
        )
        weighted_scores["regulatory_analysis"] = (
            category_scores.get("regulatory_analysis", 0) * self.analysis_weights.regulatory_weight
        )
        weighted_scores["partnership_analysis"] = (
            category_scores.get("partnership_analysis", 0) * self.analysis_weights.partnership_weight
        )

        return category_scores, weighted_scores

    def calculate_total_score(self, weighted_scores: Dict[str, float]) -> float:
        """총점 계산"""
        total_score = sum(weighted_scores.values())
        return min(total_score, self.scoring_config.max_score)  # 최대값 제한

    def calculate_grade(self, total_score: float) -> str:
        """등급 계산"""
        thresholds = self.scoring_config.grade_thresholds

        if total_score >= thresholds["S"]:
            return "S"
        elif total_score >= thresholds["A"]:
            return "A"
        elif total_score >= thresholds["B"]:
            return "B"
        elif total_score >= thresholds["C"]:
            return "C"
        else:
            return "D"

    def calculate_unicorn_probability(
        self,
        company_info: CompanyInfo,
        total_score: float,
        category_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """유니콘 확률 계산"""
        try:
            # 카테고리별 점수 포맷팅
            category_scores_text = "\n".join([
                f"- {category}: {score:.1f}점"
                for category, score in category_scores.items()
            ])

            response = self.llm(self.unicorn_probability_prompt.format(
                company_name=company_info.name,
                total_score=total_score,
                category_scores=category_scores_text,
                industry=company_info.industry
            ))

            import json
            probability_data = json.loads(response.strip())

            return {
                "probability": probability_data.get("unicorn_probability", 0.5),
                "key_factors": probability_data.get("key_factors", []),
                "reasoning": probability_data.get("reasoning", ""),
                "comparable_companies": probability_data.get("comparable_companies", [])
            }

        except Exception as e:
            # LLM 호출 실패 시 수학적 모델 사용
            return self._calculate_probability_fallback(total_score, category_scores)

    def _calculate_probability_fallback(
        self,
        total_score: float,
        category_scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """유니콘 확률 계산 백업 방법 (수학적 모델)"""

        # 기본 확률 (총점 기반)
        base_probability = total_score / 100.0

        # 카테고리별 가중치 적용
        weights = self.scoring_config.unicorn_probability_weights

        # 성장성과 시장 크기가 가장 중요
        growth_factor = category_scores.get("growth_analysis", 50) / 100.0 * weights.get("growth_rate", 0.2)

        # 기술력과 팀 역량
        tech_factor = category_scores.get("tech_security_analysis", 50) / 100.0 * weights.get("technology", 0.15)
        team_factor = category_scores.get("team_evaluation", 50) / 100.0 * weights.get("team", 0.15)

        # 비즈니스 모델과 재무
        bm_factor = category_scores.get("business_model_analysis", 50) / 100.0 * weights.get("business_model", 0.15)
        finance_factor = category_scores.get("financial_health_analysis", 50) / 100.0 * weights.get("funding", 0.1)

        # 시장 크기 (업종별 조정)
        market_factor = 0.25  # 기본값

        # 최종 확률 계산
        probability = (
            growth_factor + tech_factor + team_factor +
            bm_factor + finance_factor + market_factor
        )

        # 시그모이드 함수로 0-1 범위 조정
        probability = 1 / (1 + math.exp(-5 * (probability - 0.5)))

        return {
            "probability": min(max(probability, 0.0), 1.0),
            "key_factors": ["종합 점수 기반 계산"],
            "reasoning": "수학적 모델을 통한 확률 계산",
            "comparable_companies": []
        }

    def create_score_breakdown(
        self,
        category_scores: Dict[str, float],
        weighted_scores: Dict[str, float],
        analysis_results: List[AnalysisResult]
    ) -> Dict[str, Any]:
        """점수 상세 분석 생성"""
        breakdown = {
            "category_scores": category_scores,
            "weighted_scores": weighted_scores,
            "weights_applied": {
                "growth_analysis": self.analysis_weights.growth_weight,
                "business_model_analysis": self.analysis_weights.business_model_weight,
                "tech_security_analysis": self.analysis_weights.tech_security_weight,
                "financial_health_analysis": self.analysis_weights.financial_health_weight,
                "team_evaluation": self.analysis_weights.team_weight,
                "regulatory_analysis": self.analysis_weights.regulatory_weight,
                "partnership_analysis": self.analysis_weights.partnership_weight
            },
            "category_grades": {
                result.category: result.grade for result in analysis_results
            },
            "strengths_by_category": {
                result.category: result.key_strengths for result in analysis_results
            },
            "weaknesses_by_category": {
                result.category: result.key_weaknesses for result in analysis_results
            }
        }
        return breakdown

    def calculate_unicorn_score(
        self,
        analysis_results: List[AnalysisResult],
        company_info: CompanyInfo
    ) -> UnicornScore:
        """종합 유니콘 점수 계산"""

        # 1. 가중치 적용된 점수 계산
        category_scores, weighted_scores = self.calculate_weighted_score(analysis_results)

        # 2. 총점 계산
        total_score = self.calculate_total_score(weighted_scores)

        # 3. 등급 계산
        grade = self.calculate_grade(total_score)

        # 4. 유니콘 확률 계산
        unicorn_data = self.calculate_unicorn_probability(
            company_info, total_score, category_scores
        )

        # 5. 점수 상세 분석
        score_breakdown = self.create_score_breakdown(
            category_scores, weighted_scores, analysis_results
        )

        # 6. 추가 메타데이터
        score_breakdown.update({
            "unicorn_factors": unicorn_data.get("key_factors", []),
            "probability_reasoning": unicorn_data.get("reasoning", ""),
            "comparable_companies": unicorn_data.get("comparable_companies", [])
        })

        return UnicornScore(
            total_score=total_score,
            grade=grade,
            unicorn_probability=unicorn_data["probability"],
            category_scores=category_scores,
            score_breakdown=score_breakdown
        )

class ScoringRankingEngine:
    """점수 및 순위 엔진 메인 클래스"""

    def __init__(self):
        self.calculator = UnicornScoreCalculator()

    def process_scoring(
        self,
        analysis_results: List[AnalysisResult],
        company_info: CompanyInfo
    ) -> UnicornScore:
        """점수 계산 처리"""
        return self.calculator.calculate_unicorn_score(analysis_results, company_info)

    def validate_scoring_results(self, unicorn_score: UnicornScore) -> bool:
        """점수 결과 검증"""
        # 기본 검증
        if not (0 <= unicorn_score.total_score <= 100):
            return False

        if not (0 <= unicorn_score.unicorn_probability <= 1):
            return False

        if unicorn_score.grade not in ["S", "A", "B", "C", "D"]:
            return False

        # 점수와 등급 일치성 검증
        config = get_config()["scoring"]
        grade_thresholds = config.grade_thresholds

        expected_grade = "D"
        for grade, threshold in grade_thresholds.items():
            if unicorn_score.total_score >= threshold:
                expected_grade = grade
                break

        if unicorn_score.grade != expected_grade:
            return False

        return True

    def adjust_scores_if_needed(self, unicorn_score: UnicornScore) -> UnicornScore:
        """필요시 점수 조정"""
        if not self.validate_scoring_results(unicorn_score):
            # 점수 정규화
            if unicorn_score.total_score > 100:
                unicorn_score.total_score = 100.0
            elif unicorn_score.total_score < 0:
                unicorn_score.total_score = 0.0

            # 확률 정규화
            if unicorn_score.unicorn_probability > 1:
                unicorn_score.unicorn_probability = 1.0
            elif unicorn_score.unicorn_probability < 0:
                unicorn_score.unicorn_probability = 0.0

            # 등급 재계산
            config = get_config()["scoring"]
            grade_thresholds = config.grade_thresholds

            for grade, threshold in grade_thresholds.items():
                if unicorn_score.total_score >= threshold:
                    unicorn_score.grade = grade
                    break

        return unicorn_score

def create_scoring_ranking_engine() -> ScoringRankingEngine:
    """Scoring & Ranking Engine 생성자"""
    return ScoringRankingEngine()

def process_scoring_ranking_engine(context: PipelineContext) -> PipelineContext:
    """Scoring & Ranking Engine 처리 함수"""
    scoring_engine = create_scoring_ranking_engine()

    if not context.analysis_results:
        # 분석 결과가 없는 경우 기본 점수 생성
        context.unicorn_score = UnicornScore(
            total_score=0.0,
            grade="D",
            unicorn_probability=0.0,
            category_scores={},
            score_breakdown={}
        )
        context.processing_steps.append(
            "SCORING_RANKING_ENGINE: 분석 결과 없음 - 기본 점수 적용"
        )
        return context

    # 유니콘 점수 계산
    unicorn_score = scoring_engine.process_scoring(
        analysis_results=context.analysis_results,
        company_info=context.company_info
    )

    # 점수 검증 및 조정
    unicorn_score = scoring_engine.adjust_scores_if_needed(unicorn_score)

    context.unicorn_score = unicorn_score

    # 처리 단계 기록
    context.processing_steps.append(
        f"SCORING_RANKING_ENGINE: 총점 {unicorn_score.total_score:.1f}점, "
        f"등급 {unicorn_score.grade}, 유니콘 확률 {unicorn_score.unicorn_probability:.1%}"
    )

    return context