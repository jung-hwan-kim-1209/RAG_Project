"""
정량 평가 스코어러 (규칙 기반)
GPT보다 먼저 실행하여 객관적 점수 산출
"""
import re
from typing import Dict, List, Tuple
from models import DocumentChunk, ExternalSearchResult

class QuantitativeScorer:
    """재무제표와 문서에서 정량 지표를 추출하여 객관적 점수 산출"""

    def __init__(self):
        pass

    def extract_financial_metrics(self, documents: List[DocumentChunk],
                                  external_results: List[ExternalSearchResult]) -> Dict:
        """문서에서 핵심 재무 지표 추출"""
        all_text = ""

        # 모든 문서 텍스트 합치기
        for doc in documents:
            all_text += doc.content + "\n"
        for ext in external_results:
            all_text += ext.content + "\n"

        metrics = {
            "revenue": self._extract_revenue(all_text),
            "transaction_volume": self._extract_transaction_volume(all_text),
            "investment": self._extract_investment(all_text),
            "valuation": self._extract_valuation(all_text),
            "loss": self._extract_loss(all_text),
            "users": self._extract_users(all_text),
            "growth_rate": self._extract_growth_rate(all_text)
        }

        # 디버깅: 추출된 지표 출력
        print(f"\n[DEBUG] 추출된 재무 지표:")
        print(f"  - 매출: {metrics['revenue']}억 원")
        print(f"  - 거래액: {metrics['transaction_volume']}조 원")
        print(f"  - 투자 유치: {metrics['investment']}억 원")
        print(f"  - 기업가치: {metrics['valuation']}억 원")
        print(f"  - 영업손실: {metrics['loss']}억 원")
        print(f"  - 사용자: {metrics['users']}만 명")
        print(f"  - 성장률: {metrics['growth_rate']}%")
        print(f"\n[DEBUG] 문서 텍스트 샘플 (처음 500자):")
        print(f"{all_text[:500]}\n")

        return metrics

    def _extract_revenue(self, text: str) -> float:
        """매출액 추출 (억 원 단위)"""
        patterns = [
            r'연.*?매출.*?(\d+)\s*억',  # 연 매출
            r'매출[액]?.*?(\d+)\s*억',
            r'(\d+)\s*억.*?매출',
            r'수십억.*?중반',  # "수십억 원대 중반" → 50억으로 간주
            r'(\d+,\d+)\s*억.*?매출',  # 쉼표 포함
        ]
        max_revenue = 0.0
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if '수십억' in pattern:
                    return 50.0
                value = float(str(match).replace(',', ''))
                if value > max_revenue:
                    max_revenue = value
        return max_revenue

    def _extract_transaction_volume(self, text: str) -> float:
        """거래액/대출중개액 추출 (조 원 단위) - 최대값 사용"""
        max_volume = 0.0

        # 조 단위 모두 찾기
        patterns_jo = [
            r'(\d+)\s*조\s*원',
            r'대출.*?(\d+)\s*조',
            r'누적.*?(\d+)\s*조',
            r'중개.*?(\d+)\s*조',
        ]
        for pattern in patterns_jo:
            matches = re.findall(pattern, text)
            for match in matches:
                value = float(match)
                if value > max_volume:
                    max_volume = value

        # 조 단위가 없으면 억 단위
        if max_volume == 0:
            patterns_eok = [
                r'(\d+,?\d*)\s*억.*?대출',
                r'대출.*?(\d+,?\d*)\s*억',
            ]
            for pattern in patterns_eok:
                match = re.search(pattern, text)
                if match:
                    value = float(match.group(1).replace(',', ''))
                    return value / 10000  # 조로 변환

        return max_volume

    def _extract_investment(self, text: str) -> float:
        """투자 유치액 추출 (억 원)"""
        # 우선순위: 시리즈 투자 > 투자유치
        patterns = [
            r'시리즈.*?[A-Z].*?(\d+)\s*억',  # 시리즈 투자
            r'투자.*?유치.*?(\d+)\s*억',
            r'(\d+)\s*억.*?투자',
            r'라운드.*?(\d+)\s*억',
            r'투자.*?(\d+)\s*억',
            r'(\d+,\d+)\s*억.*?투자',  # 쉼표 포함
        ]

        max_investment = 0.0

        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                value = float(str(match).replace(',', ''))
                if value > max_investment and value >= 50:  # 최소 50억 이상
                    max_investment = value

        return max_investment

    def _extract_valuation(self, text: str) -> float:
        """기업가치 추출 (억 원)"""
        patterns = [
            r'기업가치.*?약?\s*(\d+),?(\d*)\s*억',  # 쉼표 분리
            r'밸류에이션.*?(\d+),?(\d*)\s*억',
            r'(\d+),?(\d*)\s*억\s*원?\s*대로.*?평가',  # 띄어쓰기 허용
            r'평가.*?(\d+),?(\d*)\s*억',
            r'(\d+),?(\d*)\s*억.*?밸류',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                # 쉼표로 분리된 경우 합치기 (3,000 -> 3000)
                if match.lastindex >= 2 and match.group(2):
                    value = match.group(1) + match.group(2)
                else:
                    value = match.group(1)
                return float(value)
        return 0.0

    def _extract_loss(self, text: str) -> float:
        """영업손실/순손실 추출 (억 원, 음수)"""
        patterns = [
            r'영업손실.*?[-]?(\d+)\s*억',
            r'순손실.*?[-]?(\d+)\s*억',
            r'적자.*?(\d+)\s*억'
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return -float(match.group(1))  # 손실은 음수
        return 0.0

    def _extract_users(self, text: str) -> float:
        """사용자 수 추출 (만 명)"""
        patterns = [
            r'(\d+),?(\d*)\s*만\s*건',  # 1,000만 건
            r'다운로드\s*(\d+),?(\d*)\s*만',
            r'가입자.*?(\d+),?(\d*)\s*만',
            r'MAU.*?(\d+),?(\d*)\s*만',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                # 쉼표 분리 처리
                if match.lastindex == 2 and match.group(2):
                    value = match.group(1) + match.group(2)
                else:
                    value = match.group(1)
                return float(value)
        return 0.0

    def _extract_growth_rate(self, text: str) -> float:
        """성장률 추출 (%) - 가장 높은 값 사용"""
        patterns = [
            r'(\d+)\s*%.*?성장',
            r'성장.*?(\d+)\s*%',
            r'증가.*?(\d+)\s*배',  # "3배 증가" → 200%
            r'(\d+)\s*배.*?증가'
        ]
        max_rate = 0.0
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if '배' in pattern:
                    rate = (float(match) - 1) * 100  # 3배 = 200% 성장
                else:
                    rate = float(match)
                if rate > max_rate:
                    max_rate = rate
        return max_rate

    def calculate_scores(self, metrics: Dict) -> Dict[str, float]:
        """정량 지표를 기반으로 4개 영역 점수 계산 (규칙 기반)"""

        # 1. 성장성 점수 (0-100) - 실제 매출 성장과 사용자 증가 중심
        growth_score = 0.0

        # 매출 성장률 (최우선) - 40점
        if metrics["growth_rate"] >= 200:  # 200% 이상 (3배)
            growth_score += 40
        elif metrics["growth_rate"] >= 100:  # 100% 이상 (2배)
            growth_score += 35
        elif metrics["growth_rate"] >= 50:
            growth_score += 25
        elif metrics["growth_rate"] >= 30:
            growth_score += 15
        elif metrics["growth_rate"] >= 10:
            growth_score += 10

        # 사용자 규모 - 35점
        if metrics["users"] >= 1000:  # 1000만 이상
            growth_score += 35
        elif metrics["users"] >= 500:
            growth_score += 30
        elif metrics["users"] >= 100:
            growth_score += 20
        elif metrics["users"] >= 50:
            growth_score += 10

        # 거래액 규모 (참고용) - 15점 (비중 대폭 축소)
        if metrics["transaction_volume"] >= 10:  # 10조 이상
            growth_score += 15
        elif metrics["transaction_volume"] >= 5:  # 5조 이상
            growth_score += 12
        elif metrics["transaction_volume"] >= 1:  # 1조 이상
            growth_score += 8
        elif metrics["transaction_volume"] >= 0.5:  # 5000억 이상
            growth_score += 5

        # 매출 규모 - 10점
        if metrics["revenue"] >= 300:  # 300억 이상
            growth_score += 10
        elif metrics["revenue"] >= 100:
            growth_score += 8
        elif metrics["revenue"] >= 50:
            growth_score += 5

        # 2. 비즈니스 모델 점수 (0-100) - 수익성 중심
        business_score = 0.0

        # 실제 매출 규모 (최우선) - 50점
        if metrics["revenue"] >= 500:  # 500억 이상
            business_score += 50
        elif metrics["revenue"] >= 300:  # 300억 이상
            business_score += 45
        elif metrics["revenue"] >= 100:  # 100억 이상
            business_score += 35
        elif metrics["revenue"] >= 50:
            business_score += 25
        elif metrics["revenue"] >= 30:
            business_score += 15
        else:
            business_score += 5

        # 수익성 (흑자/적자) - 30점
        if metrics["loss"] == 0:  # 흑자
            business_score += 30
        elif metrics["loss"] >= -30:  # 적자 30억 이내
            business_score += 15
        elif metrics["loss"] >= -50:  # 적자 50억 이내
            business_score += 10
        elif metrics["loss"] >= -100:  # 적자 100억 이내
            business_score += 5
        # 100억 이상 적자는 0점

        # 거래액 대비 수익률 (효율성) - 20점
        if metrics["transaction_volume"] > 0 and metrics["revenue"] > 0:
            revenue_rate = (metrics["revenue"] * 100) / (metrics["transaction_volume"] * 10000)
            if revenue_rate >= 10:  # 10% 이상 (매우 높은 수수료율)
                business_score += 20
            elif revenue_rate >= 5:  # 5% 이상
                business_score += 15
            elif revenue_rate >= 3:  # 3% 이상
                business_score += 10
            elif revenue_rate >= 1:  # 1% 이상 (일반적)
                business_score += 5
            elif revenue_rate >= 0.5:  # 0.5% 이상
                business_score += 3
            # 0.5% 미만은 매우 낮은 수익률 (0점)

        # 3. 재무 건전성 점수 (0-100) - 투자 유치와 기업가치
        financial_score = 0.0

        # 투자 유치 규모 - 40점
        if metrics["investment"] >= 1000:  # 1000억 이상 (시리즈 C/D)
            financial_score += 40
        elif metrics["investment"] >= 500:  # 500억 이상
            financial_score += 35
        elif metrics["investment"] >= 300:  # 300억 이상 (시리즈 B)
            financial_score += 30
        elif metrics["investment"] >= 100:  # 100억 이상 (시리즈 A)
            financial_score += 20
        elif metrics["investment"] >= 50:  # 50억 이상 (시드)
            financial_score += 10

        # 기업가치 - 40점
        if metrics["valuation"] >= 10000:  # 1조 이상 (유니콘)
            financial_score += 40
        elif metrics["valuation"] >= 5000:  # 5천억 이상
            financial_score += 35
        elif metrics["valuation"] >= 3000:  # 3천억 이상
            financial_score += 30
        elif metrics["valuation"] >= 1000:  # 1천억 이상
            financial_score += 20
        elif metrics["valuation"] >= 500:
            financial_score += 10

        # 매출 대비 기업가치 배수 (밸류에이션 합리성) - 20점
        if metrics["revenue"] > 0 and metrics["valuation"] > 0:
            valuation_multiple = metrics["valuation"] / metrics["revenue"]
            if 5 <= valuation_multiple <= 15:  # 적정 배수 (5~15배)
                financial_score += 20
            elif 3 <= valuation_multiple <= 20:  # 합리적 범위
                financial_score += 15
            elif 1 <= valuation_multiple <= 30:  # 허용 범위
                financial_score += 10
            # 30배 이상은 고평가, 1배 미만은 저평가 (감점)

        # 4. 기술력 점수 (기본값 - 문서에서 추출 어려움)
        tech_score = 50.0  # 중간값 기본

        return {
            "growth_analysis": min(growth_score, 100),
            "business_model_analysis": min(business_score, 100),
            "financial_health_analysis": min(financial_score, 100),
            "tech_security_analysis": tech_score
        }
