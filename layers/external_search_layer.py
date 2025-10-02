"""
Layer 4: EXTERNAL SEARCH LAYER
web_search_agent를 실행하여 최신 뉴스, 투자유치 정보, 실시간 지표를 검색하는 레이어
"""
import asyncio
import aiohttp
import os
from typing import List, Optional
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import re
import feedparser   # [완료] 구글 뉴스 RSS
import yfinance as yf  # [완료] 주가 조회용

import sys
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

        # 구글 뉴스 검색 (RSS)
        google_results = await self._search_google_news(company_name, days_back)
        results.extend(google_results)

        return results

    async def search_investment_info(self, company_name: str) -> List[ExternalSearchResult]:
        """투자유치 정보 검색 (개선)"""
        results = []
        try:
            # 투자 전문 검색 쿼리
            investment_queries = [
                f"{company_name} 시리즈 투자",
                f"{company_name} 벤처투자",
                f"{company_name} 밸류에이션",
                f"{company_name} 기업가치"
            ]

            search_url = "https://search.naver.com/search.naver"
            async with aiohttp.ClientSession() as session:
                for query in investment_queries:
                    params = {
                        "where": "news",
                        "query": query,
                        "sort": 1,
                        "ds": (datetime.now() - timedelta(days=180)).strftime("%Y.%m.%d"),  # 6개월
                        "de": datetime.now().strftime("%Y.%m.%d")
                    }
                    print(f"[재무] 투자정보 검색: '{query}'")

                    try:
                        async with session.get(search_url, params=params) as response:
                            if response.status == 200:
                                html = await response.text()
                                query_results = self._parse_naver_news_results(html, company_name)
                                # 투자 관련성이 높은 것만 필터링
                                filtered = [r for r in query_results if r.relevance_score >= 0.7]
                                results.extend(filtered)
                    except:
                        continue

        except Exception as e:
            print(f"투자정보 검색 오류: {e}")

        return results[:10]  # 상위 10개

    async def search_market_indicators(self, company_name: str) -> List[ExternalSearchResult]:
        """실시간 시장 지표 검색 (주가 포함)"""
        results = []

        # 주가 정보 검색
        stock_results = await self._search_stock_info(company_name)
        results.extend(stock_results)

        return results

    async def _search_naver_news(self, company_name: str, days_back: int) -> List[ExternalSearchResult]:
        """네이버 뉴스 검색 (개선된 쿼리)"""
        results = []
        try:
            search_url = f"https://search.naver.com/search.naver"
            # 핵심 키워드 조합으로 정확도 향상
            search_queries = [
                f"{company_name} 투자유치",
                f"{company_name} 매출",
                f"{company_name} 성장",
                f"{company_name} 펀딩",
                f"{company_name} 시리즈"
            ]

            for query in search_queries:
                params = {
                    "where": "news",
                    "query": query,
                    "sort": 1,  # 최신순
                    "ds": (datetime.now() - timedelta(days=days_back)).strftime("%Y.%m.%d"),
                    "de": datetime.now().strftime("%Y.%m.%d")
                }
                print(f"[뉴스] 네이버 뉴스 검색: '{query}'")

                async with aiohttp.ClientSession() as session:
                    async with session.get(search_url, params=params) as response:
                        if response.status == 200:
                            html = await response.text()
                            query_results = self._parse_naver_news_results(html, company_name)
                            results.extend(query_results)

                # 중복 제거 (URL 기준)
                seen_urls = set()
                unique_results = []
                for r in results:
                    if r.url not in seen_urls:
                        seen_urls.add(r.url)
                        unique_results.append(r)
                results = unique_results

        except Exception as e:
            print(f"네이버 뉴스 검색 오류: {e}")

        return results

    async def _search_google_news(self, company_name: str, days_back: int) -> List[ExternalSearchResult]:
        """구글 뉴스 검색 (RSS 기반)"""
        results = []
        try:
            rss_url = f"https://news.google.com/rss/search?q={company_name}&hl=ko&gl=KR&ceid=KR:ko"
            print(f"[웹] 구글 뉴스 검색: '{company_name}'")
            feed = feedparser.parse(rss_url)

            for entry in feed.entries[:20]:  # 최대 20개
                try:
                    published = (
                        datetime(*entry.published_parsed[:6])
                        if hasattr(entry, "published_parsed") and entry.published_parsed
                        else datetime.now()
                    )
                    results.append(
                        ExternalSearchResult(
                            title=entry.title,
                            content=getattr(entry, "summary", ""),
                            source="Google News",
                            url=entry.link,
                            published_date=published,
                            relevance_score=0.8
                        )
                    )
                except Exception:
                    continue

        except Exception as e:
            print(f"구글 뉴스 RSS 검색 오류: {e}")

        return results

    async def _search_stock_info(self, company_name: str) -> List[ExternalSearchResult]:
        """주가 정보 검색 (yfinance 활용)"""
        results = []
        try:
            ticker_map = {
                "삼성전자": "005930.KQ",
                "카카오": "035720.KQ",
                "네이버": "035420.KQ",
                "쿠팡": "CPNG",
                # 필요시 확장 가능
            }

            ticker = ticker_map.get(company_name)
            if ticker:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1d")
                if not hist.empty:
                    latest_price = hist["Close"].iloc[-1]
                    result = ExternalSearchResult(
                        title=f"{company_name} 주가 정보",
                        content=f"{company_name}의 현재 종가: {latest_price:.2f}",
                        source="Yahoo Finance",
                        url=f"https://finance.yahoo.com/quote/{ticker}",
                        published_date=datetime.now(),
                        relevance_score=0.9
                    )
                    results.append(result)
            else:
                results.append(
                    ExternalSearchResult(
                        title=f"{company_name} 주가 정보",
                        content=f"{company_name}의 티커 심볼이 매핑되지 않았습니다.",
                        source="Yahoo Finance",
                        url="https://finance.yahoo.com",
                        published_date=datetime.now(),
                        relevance_score=0.5
                    )
                )
        except Exception as e:
            print(f"주가 정보 검색 오류: {e}")

        return results

    def _parse_naver_news_results(self, html: str, company_name: str) -> List[ExternalSearchResult]:
        """네이버 뉴스 결과 파싱"""
        results = []
        try:
            soup = BeautifulSoup(html, 'html.parser')
            news_items = soup.find_all('div', class_='news_area')

            for item in news_items[:20]:  # 최대 20개
                try:
                    title_elem = item.find('a', class_='news_tit')
                    if not title_elem:
                        continue
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')

                    content_elem = item.find('div', class_='news_dsc')
                    content = content_elem.get_text(strip=True) if content_elem else ""

                    date_elem = item.find('span', class_='info')
                    date_text = date_elem.get_text(strip=True) if date_elem else ""

                    relevance_score = self._calculate_relevance_score(title + " " + content, company_name)

                    if relevance_score >= 0.5:  # 기준 상향 (0.2 -> 0.5)
                        results.append(
                            ExternalSearchResult(
                                title=title,
                                content=content,
                                source="네이버 뉴스",
                                url=url,
                                published_date=self._parse_date(date_text),
                                relevance_score=relevance_score
                            )
                        )
                except Exception:
                    continue
        except Exception as e:
            print(f"네이버 뉴스 파싱 오류: {e}")

        return results

    def _calculate_relevance_score(self, text: str, company_name: str) -> float:
        """텍스트와 회사명의 관련성 점수 계산 (재무지표 중심 강화)"""
        text_lower = text.lower()
        company_lower = company_name.lower()

        score = 0.0

        # 회사명 정확 매칭 (필수)
        if company_lower not in text_lower:
            return 0.0  # 회사명이 없으면 무조건 0점

        score = 0.3  # 기본 점수

        # 제목에 회사명이 있으면 추가 점수
        title_lower = text[:100].lower()  # 제목 부분으로 간주
        if company_lower in title_lower:
            score += 0.2

        # 핵심 투자/재무 키워드 (최고 가중치)
        critical_keywords = ["투자유치", "시리즈", "라운드", "펀딩", "밸류에이션", "기업가치",
                            "매출", "억원", "조원", "대출중개", "거래액", "누적"]
        matched_critical = sum(1 for kw in critical_keywords if kw in text_lower)
        score += min(matched_critical * 0.15, 0.35)

        # 정량 지표 언급 (숫자 + 단위)
        import re
        if re.search(r'\d+억|\d+조|\d+%', text):
            score += 0.15

        # 스타트업 관련성
        startup_keywords = ["스타트업", "벤처", "창업", "유니콘", "핀테크", "p2p"]
        if any(kw in text_lower for kw in startup_keywords):
            score += 0.1

        return min(score, 1.0)

    def _parse_date(self, date_text: str) -> Optional[datetime]:
        """날짜 텍스트를 datetime 객체로 변환"""
        try:
            if "시간" in date_text or "분" in date_text:
                return datetime.now()
            elif "일" in date_text:
                days = int(re.search(r'(\d+)일', date_text).group(1))
                return datetime.now() - timedelta(days=days)
            else:
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
        """외부 소스에서 종합 검색 (개선)"""
        if search_types is None:
            search_types = ["news", "investment", "market_indicators"]

        all_results = []
        tasks = []

        if "news" in search_types:
            tasks.append(self.web_search_agent.search_company_news(company_name))
        if "investment" in search_types:
            tasks.append(self.web_search_agent.search_investment_info(company_name))
        if "market_indicators" in search_types:
            tasks.append(self.web_search_agent.search_market_indicators(company_name))

        if tasks:
            search_results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in search_results:
                if isinstance(result, list):
                    all_results.extend(result)

        unique_results = self._deduplicate_results(all_results)
        sorted_results = sorted(unique_results, key=lambda x: x.relevance_score, reverse=True)

        max_results = int(os.getenv("MAX_EXTERNAL_RESULTS", "20"))
        return sorted_results[:max_results]

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
    return ExternalSearchLayer()


def process_external_search_layer(context: PipelineContext) -> PipelineContext:
    search_layer = create_external_search_layer()
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    external_results = loop.run_until_complete(
        search_layer.search_external_sources(context.company_info.name)
    )

    context.external_search_results = external_results

    # CLI 출력: 참고한 뉴스 목록
    print("\n" + "="*80)
    print(f"[뉴스] 참고한 뉴스/외부 자료 ({len(external_results)}개)")
    print("="*80)
    for i, result in enumerate(external_results[:10], 1):  # 상위 10개만 출력
        print(f"\n[{i}] {result.title}")
        print(f"    출처: {result.source} | 관련도: {result.relevance_score:.2f}")
        print(f"    URL: {result.url}")
        if result.content:
            preview = result.content[:100].replace('\n', ' ')
            print(f"    내용: {preview}...")
    if len(external_results) > 10:
        print(f"\n... 외 {len(external_results) - 10}개 더 참고")
    print("="*80 + "\n")

    context.processing_steps.append(
        f"EXTERNAL_SEARCH_LAYER: {len(external_results)}개 외부 검색 결과 수집 완료"
    )
    return context
