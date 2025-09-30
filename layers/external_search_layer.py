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
from serpapi import GoogleSearch

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
            days_back = int(os.getenv("NEWS_SEARCH_DAYS_BACK", "730"))
        
        print(f"🔍 COMPANY_NEWS_SEARCH - {company_name} (최근 {days_back}일)")
        print("=" * 60)
        
        results = []

        # 네이버 뉴스 검색
        print(f"📰 네이버 뉴스 검색 시작...")
        naver_results = await self._search_naver_news(company_name, days_back)
        print(f"📰 네이버 뉴스 검색 완료: {len(naver_results)}개 결과")
        for i, result in enumerate(naver_results, 1):
            print(f"  {i}. {result.title} ({result.source}) - {result.relevance_score:.2f}")
        results.extend(naver_results)

        # 구글 뉴스 검색
        print(f"🌐 구글 뉴스 검색 시작...")
        google_results = await self._search_google_news(company_name, days_back)
        print(f"🌐 구글 뉴스 검색 완료: {len(google_results)}개 결과")
        for i, result in enumerate(google_results, len(naver_results) + 1):
            print(f"  {i}. {result.title} ({result.source}) - {result.relevance_score:.2f}")
        results.extend(google_results)

        print(f"📊 총 뉴스 검색 결과: {len(results)}개")
        print("=" * 60)
        return results

    async def search_investment_info(self, company_name: str) -> List[ExternalSearchResult]:
        """투자유치 정보 검색"""
        print(f"💰 INVESTMENT_INFO_SEARCH - {company_name}")
        print("=" * 60)
        
        results = []

        # 크런치베이스 스타일 검색
        print(f"📊 투자 데이터베이스 검색 시작...")
        investment_results = await self._search_investment_databases(company_name)
        print(f"📊 투자 데이터베이스 검색 완료: {len(investment_results)}개 결과")
        for i, result in enumerate(investment_results, 1):
            print(f"  {i}. {result.title} ({result.source}) - {result.relevance_score:.2f}")
        results.extend(investment_results)

        # 벤처 투자 뉴스 검색
        print(f"🚀 벤처 투자 뉴스 검색 시작...")
        venture_results = await self._search_venture_news(company_name)
        print(f"🚀 벤처 투자 뉴스 검색 완료: {len(venture_results)}개 결과")
        for i, result in enumerate(venture_results, len(investment_results) + 1):
            print(f"  {i}. {result.title} ({result.source}) - {result.relevance_score:.2f}")
        results.extend(venture_results)

        print(f"📊 총 투자 정보 검색 결과: {len(results)}개")
        print("=" * 60)
        return results

    async def search_market_indicators(self, company_name: str) -> List[ExternalSearchResult]:
        """실시간 시장 지표 검색 (주가, 밸류에이션 등)"""
        print(f"📈 MARKET_INDICATORS_SEARCH - {company_name}")
        print("=" * 60)
        
        results = []

        # 주가 정보 검색
        print(f"📊 주가 정보 검색 시작...")
        stock_results = await self._search_stock_info(company_name)
        print(f"📊 주가 정보 검색 완료: {len(stock_results)}개 결과")
        for i, result in enumerate(stock_results, 1):
            print(f"  {i}. {result.title} ({result.source}) - {result.relevance_score:.2f}")
        results.extend(stock_results)

        # 밸류에이션 정보 검색
        print(f"💎 밸류에이션 정보 검색 시작...")
        valuation_results = await self._search_valuation_info(company_name)
        print(f"💎 밸류에이션 정보 검색 완료: {len(valuation_results)}개 결과")
        for i, result in enumerate(valuation_results, len(stock_results) + 1):
            print(f"  {i}. {result.title} ({result.source}) - {result.relevance_score:.2f}")
        results.extend(valuation_results)

        print(f"📊 총 시장 지표 검색 결과: {len(results)}개")
        print("=" * 60)
        return results

    async def _search_naver_news(self, company_name: str, days_back: int) -> List[ExternalSearchResult]:
        """네이버 뉴스 검색 (네이버 API 직접 사용)"""
        results = []
        try:
            # 네이버 API 키 확인
            naver_client_id = os.getenv("NAVER_CLIENT_ID")
            naver_client_secret = os.getenv("NAVER_CLIENT_SECRET")
            
            if not naver_client_id or not naver_client_secret:
                print(f"  ❌ 네이버 API 키가 설정되지 않음 (NAVER_CLIENT_ID, NAVER_CLIENT_SECRET)")
                return results

            search_query = f"{company_name} 투자"
            print(f"  🔍 네이버 검색 쿼리: '{search_query}'")
            print(f"  📅 검색 기간: 최근 2년")

            # 네이버 뉴스 API 직접 호출
            url = f"https://openapi.naver.com/v1/search/news.json?query={search_query}&display=20&sort=sim"
            headers = {
                'X-Naver-Client-Id': naver_client_id,
                'X-Naver-Client-Secret': naver_client_secret
            }

            print(f"  🔧 네이버 API URL: {url}")

            response = requests.get(url, headers=headers)
            result = response.json()
            
            print(f"  📊 네이버 API 응답 상태: {response.status_code}")

            # 오류 체크
            if response.status_code != 200:
                print(f"  ❌ 네이버 API 오류: {result}")
                return results

            # 결과 유효성 체크
            if "items" not in result or not result["items"]:
                print(f"  ⚠️ 네이버 뉴스 검색 결과 없음")
                return results

            for item in result["items"]:
                try:
                    title = item.get("title", "").replace("<b>", "").replace("</b>", "")
                    url = item.get("link", "")
                    description = item.get("description", "").replace("<b>", "").replace("</b>", "")
                    pub_date = item.get("pubDate", "")
                    
                    print(f"  📰 기사: {title[:50]}...")
                    
                    # 관련성 점수 계산
                    relevance_score = self._calculate_relevance_score(title + " " + description, company_name)
                    
                    if relevance_score > 0.3:  # 최소 관련성 임계값
                        external_result = ExternalSearchResult(
                            title=title,
                            content=description,
                            source="네이버 뉴스",
                            url=url,
                            published_date=datetime.now(),
                            relevance_score=relevance_score
                        )
                        results.append(external_result)
                except Exception as e:
                    print(f"  ❌ 기사 처리 오류: {e}")
                    continue

            print(f"  ✅ 네이버 뉴스 검색 완료: {len(results)}개 기사")

        except Exception as e:
            print(f"  ❌ 네이버 뉴스 검색 오류: {e}")

        return results

    async def _search_google_news(self, company_name: str, days_back: int) -> List[ExternalSearchResult]:
        """구글 뉴스 검색 (SERPAPI 사용)"""
        results = []
        try:
            serpapi_key = os.getenv("SERPAPI_API_KEY")
            if not serpapi_key:
                print(f"  ❌ SERPAPI_API_KEY가 설정되지 않음")
                return results

            search_query = f"{company_name} investment"
            print(f"  🔍 구글 검색 쿼리: '{search_query}'")

            # SERPAPI를 사용한 구글 뉴스 검색 (단순화된 쿼리)
            params = {
                "engine": "google",
                "q": search_query,
                "tbm": "nws",  # 뉴스 검색
                "hl": "ko",
                "gl": "kr",
                "api_key": serpapi_key,
                "num": 10,  # 결과 수 줄임
                "tbs": "qdr:y2"  # 최근 2년 범위 고정
            }
            
            print(f"  🔧 SERPAPI 파라미터: {params}")

            search = GoogleSearch(params)
            search_results = search.get_dict()
            
            print(f"  📊 SERPAPI 응답 키: {list(search_results.keys())}")

            # 오류 체크
            if "error" in search_results:
                print(f"  ❌ SERPAPI 오류: {search_results['error']}")
                return results

            # 결과 유효성 체크
            if "news_results" not in search_results or not search_results["news_results"]:
                print(f"  ⚠️ 구글 뉴스 검색 결과 없음")
                return results

            for result in search_results["news_results"]:
                try:
                    title = result.get("title", "")
                    url = result.get("link", "")
                    snippet = result.get("snippet", "")
                    source = result.get("source", "Google News")
                    
                    print(f"  📰 기사: {title[:50]}...")
                    
                    # 관련성 점수 계산
                    relevance_score = self._calculate_relevance_score(title + " " + snippet, company_name)
                    
                    if relevance_score > 0.3:  # 최소 관련성 임계값
                        external_result = ExternalSearchResult(
                            title=title,
                            content=snippet,
                            source=source,
                            url=url,
                            published_date=datetime.now(),
                            relevance_score=relevance_score
                        )
                        results.append(external_result)
                except Exception as e:
                    print(f"  ❌ 기사 처리 오류: {e}")
                    continue

            print(f"  ✅ 구글 뉴스 검색 완료: {len(results)}개 기사")

        except Exception as e:
            print(f"  ❌ 구글 뉴스 검색 오류: {e}")

        return results

    async def _search_investment_databases(self, company_name: str) -> List[ExternalSearchResult]:
        """투자 데이터베이스 검색 (SERPAPI 사용)"""
        results = []
        try:
            serpapi_key = os.getenv("SERPAPI_API_KEY")
            if not serpapi_key:
                print(f"    ❌ SERPAPI_API_KEY가 설정되지 않음")
                return results

            print(f"    🔍 투자 데이터베이스 검색 시작: {company_name}")
            
            # 한국 투자 관련 검색 (단순화된 쿼리)
            print(f"    🇰🇷 한국 투자 정보 검색...")
            korean_query = f"{company_name} 투자"
            korean_results = await self._search_with_serpapi(korean_query, serpapi_key, "한국 투자")
            print(f"    🇰🇷 한국 투자 정보 결과: {len(korean_results)}개")
            results.extend(korean_results)

            # 글로벌 투자 관련 검색 (단순화된 쿼리)
            print(f"    🌍 글로벌 투자 정보 검색...")
            global_query = f"{company_name} funding"
            global_results = await self._search_with_serpapi(global_query, serpapi_key, "글로벌 투자")
            print(f"    🌍 글로벌 투자 정보 결과: {len(global_results)}개")
            results.extend(global_results)

        except Exception as e:
            print(f"    ❌ 투자 데이터베이스 검색 오류: {e}")

        return results

    async def _search_with_serpapi(self, query: str, api_key: str, source_name: str) -> List[ExternalSearchResult]:
        """SERPAPI를 사용한 일반 검색"""
        results = []
        try:
            params = {
                "engine": "google",
                "q": query,
                "hl": "ko",
                "gl": "kr",
                "api_key": api_key,
                "num": 5,  # 결과 수 더 줄임
                "tbs": "qdr:y2"  # 최근 2년 범위 고정
            }
            
            print(f"      🔧 {source_name} SERPAPI 파라미터: {params}")

            search = GoogleSearch(params)
            search_results = search.get_dict()
            
            print(f"      📊 {source_name} SERPAPI 응답 키: {list(search_results.keys())}")

            # 오류 체크
            if "error" in search_results:
                print(f"      ❌ {source_name} SERPAPI 오류: {search_results['error']}")
                return results

            # 결과 유효성 체크
            if "organic_results" not in search_results or not search_results["organic_results"]:
                print(f"      ⚠️ {source_name} 검색 결과 없음")
                return results

            for result in search_results["organic_results"]:
                try:
                    title = result.get("title", "")
                    url = result.get("link", "")
                    snippet = result.get("snippet", "")
                    
                    print(f"      📰 {source_name} 결과: {title[:50]}...")
                    
                    # 관련성 점수 계산
                    relevance_score = self._calculate_relevance_score(title + " " + snippet, query)
                    
                    if relevance_score > 0.3:  # 최소 관련성 임계값
                        external_result = ExternalSearchResult(
                            title=title,
                            content=snippet,
                            source=source_name,
                            url=url,
                            published_date=datetime.now(),
                            relevance_score=relevance_score
                        )
                        results.append(external_result)
                except Exception as e:
                    print(f"      ❌ {source_name} 결과 처리 오류: {e}")
                    continue

        except Exception as e:
            print(f"      ❌ {source_name} 검색 오류: {e}")

        return results

    async def _search_venture_news(self, company_name: str) -> List[ExternalSearchResult]:
        """벤처 투자 관련 뉴스 검색 (SERPAPI 사용)"""
        results = []
        try:
            serpapi_key = os.getenv("SERPAPI_API_KEY")
            if not serpapi_key:
                print(f"    ❌ SERPAPI_API_KEY가 설정되지 않음")
                return results

            # 단순화된 검색 쿼리
            print(f"    🚀 벤처 뉴스 검색 시작: {company_name}")
            
            # 일반적인 벤처 뉴스 검색 (단순화된 쿼리)
            query = f"{company_name} 벤처"
            source_results = await self._search_with_serpapi(query, serpapi_key, "벤처 뉴스")
            results.extend(source_results)
            print(f"    🚀 벤처 뉴스 결과: {len(source_results)}개")

        except Exception as e:
            print(f"    ❌ 벤처 뉴스 검색 오류: {e}")

        return results


    async def _search_stock_info(self, company_name: str) -> List[ExternalSearchResult]:
        """주가 정보 검색 (SERPAPI 사용)"""
        results = []
        try:
            serpapi_key = os.getenv("SERPAPI_API_KEY")
            if not serpapi_key:
                print(f"    ❌ SERPAPI_API_KEY가 설정되지 않음")
                return results

            print(f"    📊 주가 정보 검색 시작: {company_name}")
            
            # 주가 관련 검색 (단순화된 쿼리)
            stock_query = f"{company_name} 주가"
            stock_results = await self._search_with_serpapi(stock_query, serpapi_key, "주가 정보")
            results.extend(stock_results)
            print(f"    📊 주가 정보 검색 완료: {len(stock_results)}개")

        except Exception as e:
            print(f"    ❌ 주가 정보 검색 오류: {e}")

        return results

    async def _search_valuation_info(self, company_name: str) -> List[ExternalSearchResult]:
        """밸류에이션 정보 검색 (SERPAPI 사용)"""
        results = []
        try:
            serpapi_key = os.getenv("SERPAPI_API_KEY")
            if not serpapi_key:
                print(f"    ❌ SERPAPI_API_KEY가 설정되지 않음")
                return results

            print(f"    💎 밸류에이션 정보 검색 시작: {company_name}")
            
            # 밸류에이션 관련 검색 (단순화된 쿼리)
            valuation_query = f"{company_name} 기업가치"
            valuation_results = await self._search_with_serpapi(valuation_query, serpapi_key, "밸류에이션 정보")
            results.extend(valuation_results)
            print(f"    💎 밸류에이션 정보 검색 완료: {len(valuation_results)}개")

        except Exception as e:
            print(f"    ❌ 밸류에이션 정보 검색 오류: {e}")

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

        print(f"🌐 EXTERNAL_SEARCH_LAYER - {company_name}")
        print(f"🔍 검색 타입: {search_types}")
        print("=" * 60)

        all_results = []

        # 병렬 검색 실행
        tasks = []

        if "news" in search_types:
            tasks.append(self.web_search_agent.search_company_news(company_name))

        if "investment" in search_types:
            tasks.append(self.web_search_agent.search_investment_info(company_name))

        if "market_indicators" in search_types:
            tasks.append(self.web_search_agent.search_market_indicators(company_name))

        print(f"📋 총 {len(tasks)}개 검색 작업 시작...")

        # 모든 검색 작업 실행
        if tasks:
            search_results = await asyncio.gather(*tasks, return_exceptions=True)

            for i, result in enumerate(search_results, 1):
                if isinstance(result, list):
                    all_results.extend(result)
                    print(f"📋 작업 {i} 완료: {len(result)}개 결과")
                else:
                    print(f"❌ 작업 {i} 실패: {result}")

        print(f"📊 전체 검색 결과: {len(all_results)}개")

        # 중복 제거 및 정렬
        unique_results = self._deduplicate_results(all_results)
        print(f"🔄 중복 제거 후: {len(unique_results)}개")
        
        sorted_results = sorted(unique_results, key=lambda x: x.relevance_score, reverse=True)
        print(f"📈 관련성 점수 기준 정렬 완료")

        max_results = int(os.getenv("MAX_EXTERNAL_RESULTS", "20"))
        final_results = sorted_results[:max_results]
        
        print(f"📊 최종 반환 결과: {len(final_results)}개 (상위 {max_results}개)")
        print("=" * 60)
        
        return final_results

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
    print(f"🚀 EXTERNAL_SEARCH_LAYER 시작 - {context.company_info.name}")
    print("=" * 60)
    
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

    print(f"✅ EXTERNAL_SEARCH_LAYER 완료 - {len(external_results)}개 결과")
    print("=" * 60)
    
    return context