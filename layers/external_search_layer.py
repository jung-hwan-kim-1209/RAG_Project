"""
Layer 4: EXTERNAL SEARCH LAYER
web_search_agent를 실행하여 최신 뉴스, 투자유치 정보, 실시간 지표를 검색하는 레이어
"""
import asyncio
import aiohttp
import requests
import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import ExternalSearchResult, PipelineContext
from config import get_config

class WebSearchAgent:
    """웹 검색 에이전트"""

    def __init__(self):
        self.config = get_config()
        self.session = None

    async def search_company_news(self, company_name: str, days_back: int = None) -> List[ExternalSearchResult]:
        """회사 관련 최신 뉴스 검색"""
        if days_back is None:
            days_back = int(os.getenv("NEWS_SEARCH_DAYS_BACK", "30"))
        results = []

        # 네이버 뉴스 검색
        naver_results = await self._search_naver_news(company_name, days_back)
        results.extend(naver_results)

        # 구글 뉴스 검색
        google_results = await self._search_google_news(company_name, days_back)
        results.extend(google_results)

        return results

    async def search_investment_info(self, company_name: str) -> List[ExternalSearchResult]:
        """투자유치 정보 검색"""
        results = []

        # 크런치베이스 스타일 검색
        investment_results = await self._search_investment_databases(company_name)
        results.extend(investment_results)

        # 벤처 투자 뉴스 검색
        venture_results = await self._search_venture_news(company_name)
        results.extend(venture_results)

        return results

    async def search_market_indicators(self, company_name: str) -> List[ExternalSearchResult]:
        """실시간 시장 지표 검색 (주가, 밸류에이션 등)"""
        results = []

        # 주가 정보 검색
        stock_results = await self._search_stock_info(company_name)
        results.extend(stock_results)

        # 밸류에이션 정보 검색
        valuation_results = await self._search_valuation_info(company_name)
        results.extend(valuation_results)

        return results

    async def _search_naver_news(self, company_name: str, days_back: int) -> List[ExternalSearchResult]:
        """네이버 뉴스 검색"""
        results = []
        try:
            # 네이버 뉴스 API 또는 스크래핑
            search_url = f"https://search.naver.com/search.naver"
            params = {
                "where": "news",
                "query": f"{company_name} 투자 스타트업",
                "sort": 1,  # 최신순
                "ds": (datetime.now() - timedelta(days=days_back)).strftime("%Y.%m.%d"),
                "de": datetime.now().strftime("%Y.%m.%d")
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(search_url, params=params) as response:
                    if response.status == 200:
                        html = await response.text()
                        results = self._parse_naver_news_results(html, company_name)

        except Exception as e:
            print(f"네이버 뉴스 검색 오류: {e}")

        return results

    async def _search_google_news(self, company_name: str, days_back: int) -> List[ExternalSearchResult]:
        """구글 뉴스 검색"""
        results = []
        try:
            # 구글 뉴스 RSS 또는 API 사용
            search_query = f"{company_name} startup investment funding"

            # 실제 구현에서는 Google News API 또는 RSS 피드 사용
            # 여기서는 예시 구조만 제공
            example_results = [
                ExternalSearchResult(
                    title=f"{company_name} raises Series B funding",
                    content=f"Latest news about {company_name} investment...",
                    source="TechCrunch",
                    url="https://example.com/news1",
                    published_date=datetime.now() - timedelta(days=1),
                    relevance_score=0.9
                )
            ]
            results.extend(example_results)

        except Exception as e:
            print(f"구글 뉴스 검색 오류: {e}")

        return results

    async def _search_investment_databases(self, company_name: str) -> List[ExternalSearchResult]:
        """투자 데이터베이스 검색"""
        results = []
        try:
            # Crunchbase, PitchBook 등의 API 연동
            # 여기서는 예시 구조만 제공

            # 한국 투자 데이터베이스 검색
            korean_results = await self._search_korean_investment_db(company_name)
            results.extend(korean_results)

            # 글로벌 투자 데이터베이스 검색
            global_results = await self._search_global_investment_db(company_name)
            results.extend(global_results)

        except Exception as e:
            print(f"투자 데이터베이스 검색 오류: {e}")

        return results

    async def _search_korean_investment_db(self, company_name: str) -> List[ExternalSearchResult]:
        """한국 투자 데이터베이스 검색"""
        results = []

        # 벤처스퀘어, 로켓펀치 등 한국 벤처 데이터베이스
        # 실제 구현에서는 해당 사이트들의 API 사용
        example_results = [
            ExternalSearchResult(
                title=f"{company_name} 투자 유치 현황",
                content=f"{company_name}의 최근 투자 라운드 정보...",
                source="벤처스퀘어",
                url="https://example.com/investment1",
                published_date=datetime.now() - timedelta(days=7),
                relevance_score=0.85
            )
        ]
        results.extend(example_results)

        return results

    async def _search_global_investment_db(self, company_name: str) -> List[ExternalSearchResult]:
        """글로벌 투자 데이터베이스 검색"""
        results = []

        # Crunchbase API 사용 예시
        # 실제로는 API 키와 함께 요청
        example_results = [
            ExternalSearchResult(
                title=f"{company_name} funding rounds",
                content=f"Global investment data for {company_name}...",
                source="Crunchbase",
                url="https://example.com/crunchbase1",
                published_date=datetime.now() - timedelta(days=14),
                relevance_score=0.8
            )
        ]
        results.extend(example_results)

        return results

    async def _search_venture_news(self, company_name: str) -> List[ExternalSearchResult]:
        """벤처 투자 관련 뉴스 검색"""
        results = []

        venture_news_sources = [
            "https://platum.kr",
            "https://www.venturesquare.net",
            "https://techcrunch.com",
            "https://www.startupnews.kr"
        ]

        for source in venture_news_sources:
            try:
                source_results = await self._search_specific_source(source, company_name)
                results.extend(source_results)
            except Exception as e:
                print(f"{source} 검색 오류: {e}")

        return results

    async def _search_specific_source(self, source_url: str, company_name: str) -> List[ExternalSearchResult]:
        """특정 소스에서 검색"""
        results = []

        # 실제 구현에서는 각 사이트별 스크래핑 또는 API 사용
        # 여기서는 예시 구조만 제공
        if "platum" in source_url:
            example_result = ExternalSearchResult(
                title=f"{company_name} 관련 플래텀 기사",
                content=f"{company_name}에 대한 상세 분석...",
                source="플래텀",
                url=f"{source_url}/article/example",
                published_date=datetime.now() - timedelta(days=5),
                relevance_score=0.75
            )
            results.append(example_result)

        return results

    async def _search_stock_info(self, company_name: str) -> List[ExternalSearchResult]:
        """주가 정보 검색"""
        results = []

        try:
            # 한국거래소, 야후 파이낸스 등에서 주가 정보 검색
            # 실제로는 금융 API 사용 (예: Alpha Vantage, Yahoo Finance API)

            stock_info = ExternalSearchResult(
                title=f"{company_name} 주가 정보",
                content=f"{company_name}의 현재 주가 및 시가총액 정보...",
                source="Yahoo Finance",
                url="https://finance.yahoo.com/quote/example",
                published_date=datetime.now(),
                relevance_score=0.9
            )
            results.append(stock_info)

        except Exception as e:
            print(f"주가 정보 검색 오류: {e}")

        return results

    async def _search_valuation_info(self, company_name: str) -> List[ExternalSearchResult]:
        """밸류에이션 정보 검색"""
        results = []

        try:
            # 프라이빗 마켓 밸류에이션 정보 검색
            valuation_info = ExternalSearchResult(
                title=f"{company_name} 기업가치 평가",
                content=f"{company_name}의 최근 기업가치 평가 정보...",
                source="PitchBook",
                url="https://pitchbook.com/profiles/example",
                published_date=datetime.now() - timedelta(days=30),
                relevance_score=0.85
            )
            results.append(valuation_info)

        except Exception as e:
            print(f"밸류에이션 정보 검색 오류: {e}")

        return results

    def _parse_naver_news_results(self, html: str, company_name: str) -> List[ExternalSearchResult]:
        """네이버 뉴스 결과 파싱"""
        results = []

        try:
            soup = BeautifulSoup(html, 'html.parser')

            # 네이버 뉴스 결과 파싱 로직
            news_items = soup.find_all('div', class_='news_area')

            for item in news_items:
                try:
                    title_elem = item.find('a', class_='news_tit')
                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')

                    # 내용 추출
                    content_elem = item.find('div', class_='news_dsc')
                    content = content_elem.get_text(strip=True) if content_elem else ""

                    # 날짜 추출
                    date_elem = item.find('span', class_='info')
                    date_text = date_elem.get_text(strip=True) if date_elem else ""

                    # 관련성 점수 계산
                    relevance_score = self._calculate_relevance_score(title + " " + content, company_name)

                    if relevance_score > 0.3:  # 최소 관련성 임계값
                        result = ExternalSearchResult(
                            title=title,
                            content=content,
                            source="네이버 뉴스",
                            url=url,
                            published_date=self._parse_date(date_text),
                            relevance_score=relevance_score
                        )
                        results.append(result)

                except Exception as e:
                    continue

        except Exception as e:
            print(f"네이버 뉴스 파싱 오류: {e}")

        return results

    def _calculate_relevance_score(self, text: str, company_name: str) -> float:
        """텍스트와 회사명의 관련성 점수 계산"""
        text_lower = text.lower()
        company_lower = company_name.lower()

        score = 0.0

        # 회사명 직접 언급
        if company_lower in text_lower:
            score += 0.5

        # 투자 관련 키워드
        investment_keywords = ["투자", "펀딩", "유치", "투자유치", "시리즈", "라운드", "밸류에이션"]
        for keyword in investment_keywords:
            if keyword in text_lower:
                score += 0.1

        # 스타트업 관련 키워드
        startup_keywords = ["스타트업", "벤처", "창업", "기업", "사업"]
        for keyword in startup_keywords:
            if keyword in text_lower:
                score += 0.05

        return min(score, 1.0)

    def _parse_date(self, date_text: str) -> Optional[datetime]:
        """날짜 텍스트를 datetime 객체로 변환"""
        try:
            # 다양한 날짜 형식 처리
            if "시간" in date_text or "분" in date_text:
                return datetime.now()
            elif "일" in date_text:
                days = int(re.search(r'(\d+)일', date_text).group(1))
                return datetime.now() - timedelta(days=days)
            else:
                # 기본적으로 현재 시간 반환
                return datetime.now()
        except:
            return datetime.now()

class ExternalSearchLayer:
    """외부 검색 레이어 메인 클래스"""

    def __init__(self):
        self.web_search_agent = WebSearchAgent()

    async def search_external_sources(
        self,
        company_name: str,
        search_types: List[str] = None
    ) -> List[ExternalSearchResult]:
        """외부 소스에서 종합 검색"""

        if search_types is None:
            search_types = ["news", "investment", "market_indicators"]

        all_results = []

        # 병렬 검색 실행
        tasks = []

        if "news" in search_types:
            tasks.append(self.web_search_agent.search_company_news(company_name))

        if "investment" in search_types:
            tasks.append(self.web_search_agent.search_investment_info(company_name))

        if "market_indicators" in search_types:
            tasks.append(self.web_search_agent.search_market_indicators(company_name))

        # 모든 검색 작업 실행
        if tasks:
            search_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in search_results:
                if isinstance(result, list):
                    all_results.extend(result)

        # 중복 제거 및 정렬
        unique_results = self._deduplicate_results(all_results)
        sorted_results = sorted(unique_results, key=lambda x: x.relevance_score, reverse=True)

        max_results = int(os.getenv("MAX_EXTERNAL_RESULTS", "20"))
        return sorted_results[:max_results]  # 환경변수로 설정된 상위 결과만 반환

    def _deduplicate_results(self, results: List[ExternalSearchResult]) -> List[ExternalSearchResult]:
        """중복 결과 제거"""
        seen_urls = set()
        unique_results = []

        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)

        return unique_results

def create_external_search_layer() -> ExternalSearchLayer:
    """External Search Layer 생성자"""
    return ExternalSearchLayer()

def process_external_search_layer(context: PipelineContext) -> PipelineContext:
    """External Search Layer 처리 함수"""
    search_layer = create_external_search_layer()

    # 비동기 검색 실행
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    external_results = loop.run_until_complete(
        search_layer.search_external_sources(context.company_info.name)
    )

    context.external_search_results = external_results

    # 처리 단계 기록
    context.processing_steps.append(
        f"EXTERNAL_SEARCH_LAYER: {len(external_results)}개 외부 검색 결과 수집 완료"
    )

    return context