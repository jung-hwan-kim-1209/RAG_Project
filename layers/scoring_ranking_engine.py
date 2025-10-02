"""
Layer 6: SCORING & RANKING ENGINE
점수 합산 + 유니콘 확률 계산
"""
import os, math, json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AnalysisResult, UnicornScore, PipelineContext, CompanyInfo
from config import get_config

class UnicornScoreCalculator:
    def __init__(self):
        self.config = get_config()
        self.weights = self.config["analysis_weights"]
        self.scoring = self.config["scoring"]
        self.llm = ChatOpenAI(
            openai_api_key=self.config["model"].openai_api_key,
            temperature=0.1,
            model=self.config["model"].model_name
        )
        self.prob_prompt = PromptTemplate(
            input_variables=["company_name","total_score","category_scores","industry"],
            template="""{company_name}의 유니콘 확률을 0~1 값으로 산정하세요.
점수:{total_score}
영역별:
{category_scores}
JSON:
{{"unicorn_probability":0.5,"key_factors":["..."],"reasoning":"..."}}"""
        )

    def calculate_weighted_score(self, results: List[AnalysisResult]):
        cat_scores, weighted = {}, {}
        for r in results:
            cat_scores[r.category] = r.score
        weighted["growth_analysis"] = cat_scores.get("growth_analysis",0)*self.weights.growth_weight
        weighted["business_model_analysis"] = cat_scores.get("business_model_analysis",0)*self.weights.business_model_weight
        weighted["tech_security_analysis"] = cat_scores.get("tech_security_analysis",0)*self.weights.tech_security_weight
        weighted["financial_health_analysis"] = cat_scores.get("financial_health_analysis",0)*self.weights.financial_health_weight

        print("[DEBUG] Raw category_scores:", cat_scores)
        print("[DEBUG] Weighted scores:", weighted)
        return cat_scores, weighted

    def calculate_total_score(self, weighted): return min(sum(weighted.values()), self.scoring.max_score)

    def calculate_grade(self, total):
        th = self.scoring.grade_thresholds
        if total>=th["S"]: return "S"
        elif total>=th["A"]: return "A"
        elif total>=th["B"]: return "B"
        elif total>=th["C"]: return "C"
        return "D"

    def calculate_unicorn_probability(self, company_info, total, cat_scores):
        try:
            # 점수 기반 객관적 평가 추가
            base_prob = min(total / 100, 1.0)

            # 영역별 가중치 반영
            growth = cat_scores.get("growth_analysis", 0)
            business = cat_scores.get("business_model_analysis", 0)
            tech = cat_scores.get("tech_security_analysis", 0)
            finance = cat_scores.get("financial_health_analysis", 0)

            # 균형도 평가 (어느 한 영역이 너무 낮으면 감점)
            scores_list = [growth, business, tech, finance]
            min_score = min(scores_list)
            balance_penalty = 0
            if min_score < 50:
                balance_penalty = (50 - min_score) * 0.01  # 최대 50% 감점

            text = "\n".join([f"- {k}:{v:.1f}" for k,v in cat_scores.items()])
            enhanced_prompt = f"""{company_info.name}의 유니콘 확률을 0~1 값으로 산정하세요.

종합 점수: {total}/100
영역별 점수:
{text}

평가 기준:
- 성장성({growth:.1f}): 시장 확장성과 성장 추세
- 비즈니스 모델({business:.1f}): 수익 구조의 지속가능성
- 기술력/보안({tech:.1f}): 기술 경쟁력과 보안 수준
- 재무 건전성({finance:.1f}): 재무 안정성과 자금 조달 능력

균형도 분석: 최저 점수 {min_score:.1f}점 {'(불균형 리스크 있음)' if min_score < 60 else '(균형적)'}

JSON 형식으로 응답:
{{"unicorn_probability": 0.X, "key_factors": ["구체적 근거1", "구체적 근거2"], "reasoning": "상세 분석"}}"""

            resp = self.llm.invoke(enhanced_prompt)
            print("\n[검색] SCORING_ENGINE GPT 응답:", resp.content)
            data = json.loads(resp.content.strip())

            # GPT 확률과 객관적 확률의 가중 평균
            gpt_prob = data.get("unicorn_probability", 0.5)
            adjusted_prob = (gpt_prob * 0.7 + base_prob * 0.3) * (1 - balance_penalty)

            return {"probability": min(max(adjusted_prob, 0), 1),
                    "key_factors": data.get("key_factors", []),
                    "reasoning": data.get("reasoning", ""),
                    "balance_score": min_score,
                    "balance_penalty": balance_penalty}
        except Exception as e:
            return {"probability": total/100, "key_factors":["fallback"], "reasoning":str(e), "balance_score": 0, "balance_penalty": 0}

    def calculate_unicorn_score(self, results, company_info):
        cat, weighted = self.calculate_weighted_score(results)
        total = self.calculate_total_score(weighted)
        grade = self.calculate_grade(total)
        prob = self.calculate_unicorn_probability(company_info,total,cat)
        return UnicornScore(total_score=total, grade=grade,
                            unicorn_probability=prob["probability"],
                            category_scores=cat,
                            score_breakdown={"weighted":weighted,"reasoning":prob["reasoning"]})

def process_scoring_ranking_engine(context: PipelineContext) -> PipelineContext:
    calc = UnicornScoreCalculator()
    if not context.analysis_results:
        context.unicorn_score = UnicornScore(total_score=0,grade="D",unicorn_probability=0.0,
                                             category_scores={},score_breakdown={})
        context.processing_steps.append("SCORING_ENGINE: 결과 없음")
        return context
    us = calc.calculate_unicorn_score(context.analysis_results, context.company_info)
    context.unicorn_score = us
    context.processing_steps.append(
        f"SCORING_ENGINE: 총점 {us.total_score:.1f}, 등급 {us.grade}, 확률 {us.unicorn_probability:.2f}"
    )
    return context
