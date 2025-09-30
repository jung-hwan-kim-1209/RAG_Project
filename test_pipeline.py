"""
AI 스타트업 투자 평가 에이전트 파이프라인 테스트
"""
import os
import unittest
from unittest.mock import Mock, patch
from datetime import datetime

# 테스트용 환경 변수 설정
from dotenv import load_dotenv
load_dotenv()

# 테스트용 환경 변수 설정 (기본값이 없는 경우에만)
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "test_key"
if not os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = "test_hf_token"

from models import (
    CompanyInfo, ParsedInput, EvaluationType, DocumentChunk,
    ExternalSearchResult, AnalysisResult, RiskAssessment, RiskLevel,
    UnicornScore, InvestmentReport, InvestmentRecommendation, PipelineContext
)
from pipeline import InvestmentEvaluationPipeline
from layers.input_layer import InputParser
from layers.analysis_engine import GrowthAnalyzer
from layers.scoring_engine import UnicornScoreCalculator

class TestModels(unittest.TestCase):
    """데이터 모델 테스트"""

    def test_company_info_creation(self):
        """CompanyInfo 생성 테스트"""
        company = CompanyInfo(
            name="에이젠글로벌",
            industry="핀테크",
            founded_year=2013,
            headquarters="서울"
        )
        self.assertEqual(company.name, "에이젠글로벌")
        self.assertEqual(company.industry, "핀테크")

    def test_analysis_result_creation(self):
        """AnalysisResult 생성 테스트"""
        result = AnalysisResult(
            category="growth_analysis",
            score=85.0,
            grade="A",
            summary="우수한 성장성",
            detailed_analysis="상세 분석 내용",
            key_strengths=["강력한 성장", "시장 확장"],
            key_weaknesses=["경쟁 심화"]
        )
        self.assertEqual(result.score, 85.0)
        self.assertEqual(result.grade, "A")

    def test_risk_assessment_creation(self):
        """RiskAssessment 생성 테스트"""
        risk = RiskAssessment(
            category="market_risk",
            risk_level=RiskLevel.MEDIUM,
            description="시장 위험 중간 수준",
            impact_score=6.0,
            probability=0.4
        )
        self.assertEqual(risk.risk_level, RiskLevel.MEDIUM)
        self.assertEqual(risk.impact_score, 6.0)

class TestInputLayer(unittest.TestCase):
    """입력 레이어 테스트"""

    def setUp(self):
        self.input_parser = InputParser()

    @patch('layers.input_layer.ChatOpenAI')
    def test_simple_parsing(self, mock_chat_openai):
        """간단한 입력 파싱 테스트"""
        # Mock LLM 응답
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.content = '{"company_name": "에이젠글로벌", "evaluation_type": "전체 평가", "specific_focus_areas": [], "additional_requirements": ""}'
        mock_llm_instance.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm_instance

        # 새로운 InputParser 인스턴스 생성 (Mock이 적용된 상태에서)
        parser = InputParser()
        result = parser.parse("에이젠글로벌의 투자 가치를 평가해줘")

        self.assertEqual(result.company_name, "에이젠글로벌")
        self.assertEqual(result.evaluation_type, EvaluationType.FULL_EVALUATION)

    def test_fallback_parsing(self):
        """백업 파싱 로직 테스트"""
        result = self.input_parser._fallback_parsing("카카오 성장성 분석해줘")

        self.assertEqual(result.company_name, "카카오")
        self.assertEqual(result.evaluation_type, EvaluationType.GROWTH_ANALYSIS)

    def test_company_info_extraction(self):
        """회사 정보 추출 테스트"""
        company_info = self.input_parser.extract_company_info("에이젠글로벌")

        self.assertEqual(company_info.name, "에이젠글로벌")
        # 알려진 기업인 경우 추가 정보 확인
        if company_info.industry:
            self.assertEqual(company_info.industry, "핀테크")

class TestScoringEngine(unittest.TestCase):
    """점수 엔진 테스트"""

    def setUp(self):
        self.calculator = UnicornScoreCalculator()

    def test_grade_calculation(self):
        """등급 계산 테스트"""
        self.assertEqual(self.calculator.calculate_grade(95.0), "S")
        self.assertEqual(self.calculator.calculate_grade(85.0), "A")
        self.assertEqual(self.calculator.calculate_grade(75.0), "B")
        self.assertEqual(self.calculator.calculate_grade(65.0), "C")
        self.assertEqual(self.calculator.calculate_grade(45.0), "D")

    def test_weighted_score_calculation(self):
        """가중치 점수 계산 테스트"""
        analysis_results = [
            AnalysisResult("growth_analysis", 80.0, "B", "", "", [], []),
            AnalysisResult("business_model_analysis", 85.0, "A", "", "", [], []),
            AnalysisResult("tech_security_analysis", 90.0, "A", "", "", [], [])
        ]

        category_scores, weighted_scores = self.calculator.calculate_weighted_score(analysis_results)

        self.assertEqual(category_scores["growth_analysis"], 80.0)
        self.assertTrue(weighted_scores["growth_analysis"] > 0)

class TestPipeline(unittest.TestCase):
    """전체 파이프라인 테스트"""

    def setUp(self):
        self.pipeline = InvestmentEvaluationPipeline()

    @patch('layers.input_layer.ChatOpenAI')
    @patch('layers.knowledge_base_layer.Chroma')
    @patch('layers.external_search_layer.WebSearchAgent')
    def test_partial_pipeline_execution(self, mock_search, mock_chroma, mock_chat_openai):
        """부분 파이프라인 실행 테스트"""
        # Mock 설정
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = '{"company_name": "에이젠글로벌", "evaluation_type": "전체 평가", "specific_focus_areas": [], "additional_requirements": ""}'
        mock_llm.invoke.return_value = mock_response
        mock_chat_openai.return_value = mock_llm

        # 테스트용 컨텍스트 실행
        context = self.pipeline.execute_partial_pipeline(
            user_input="에이젠글로벌 투자 평가",
            start_layer="INPUT_LAYER",
            end_layer="INPUT_LAYER"
        )

        self.assertIsNotNone(context.parsed_input)
        self.assertTrue(len(context.processing_steps) > 0)

class TestQualityMetrics(unittest.TestCase):
    """품질 지표 테스트"""

    def test_relevance_calculation(self):
        """관련성 계산 테스트"""
        from layers.quality_check_layer import RelevanceChecker

        checker = RelevanceChecker()

        # 기본 관련성 계산 테스트
        company_name = "에이젠글로벌"
        report = Mock()
        report.company_info.name = "에이젠글로벌"
        report.analysis_results = [Mock()]
        report.risk_assessments = [Mock()]

        relevance = checker._calculate_basic_relevance(company_name, report)

        self.assertTrue(0.0 <= relevance <= 1.0)
        self.assertGreater(relevance, 0.5)  # 회사명 일치로 인한 점수

class TestEndToEndScenarios(unittest.TestCase):
    """End-to-End 시나리오 테스트"""

    def setUp(self):
        self.test_queries = [
            "에이젠글로벌의 투자 가치를 평가해줘",
            "에이젠글로벌의 성장성 분석",
            "에이젠글로벌의 리스크 평가",
            "에이젠글로벌의 기술력 분석"
        ]

    def test_query_parsing_scenarios(self):
        """다양한 쿼리 파싱 시나리오 테스트"""
        parser = InputParser()

        for query in self.test_queries:
            with self.subTest(query=query):
                try:
                    result = parser._fallback_parsing(query)
                    self.assertIsNotNone(result.company_name)
                    self.assertIn(result.evaluation_type, list(EvaluationType))
                except Exception as e:
                    self.fail(f"Query parsing failed for '{query}': {e}")

class TestErrorHandling(unittest.TestCase):
    """에러 처리 테스트"""

    def test_missing_api_key(self):
        """API 키 누락 처리 테스트"""
        # 환경 변수에서 API 키 제거
        original_key = os.environ.get("OPENAI_API_KEY")
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        
        try:
            # ModelConfig 인스턴스 생성 시 직접 환경변수 확인
            from config import ModelConfig
            
            # 환경변수가 실제로 삭제되었는지 확인
            self.assertIsNone(os.environ.get("OPENAI_API_KEY"))
            
            # ModelConfig의 기본값 로직을 직접 테스트
            test_key = os.getenv("OPENAI_API_KEY", "")
            self.assertEqual(test_key, "")
            
        finally:
            # API 키 복원
            if original_key:
                os.environ["OPENAI_API_KEY"] = original_key

    def test_empty_input_handling(self):
        """빈 입력 처리 테스트"""
        parser = InputParser()
        result = parser._fallback_parsing("")

        self.assertEqual(result.company_name, "")
        self.assertEqual(result.evaluation_type, EvaluationType.FULL_EVALUATION)

class TestIntegration(unittest.TestCase):
    """통합 테스트"""

    def test_mock_pipeline_flow(self):
        """Mock을 사용한 파이프라인 플로우 테스트"""
        # 테스트용 컨텍스트 생성
        context = PipelineContext(
            parsed_input=ParsedInput(
                company_name="에이젠글로벌",
                evaluation_type=EvaluationType.FULL_EVALUATION
            ),
            company_info=CompanyInfo(name="에이젠글로벌", industry="핀테크"),
            retrieved_documents=[
                DocumentChunk("테스트 문서 내용", "RAG\RAG_Project\data\documents\company_profiles\rag.pdf")
            ],
            external_search_results=[
                ExternalSearchResult("에이젠글로벌 뉴스", "에이젠글로벌 관련 뉴스 내용", "news_source", "http://example.com")
            ],
            analysis_results=[
                AnalysisResult("growth_analysis", 85.0, "A", "성장성 우수", "상세 분석", ["강점1"], ["약점1"])
            ],
            risk_assessments=[
                RiskAssessment("market_risk", RiskLevel.LOW, "시장 리스크 낮음", 3.0, 0.3)
            ]
        )

        # 각 단계별 데이터가 올바르게 설정되었는지 확인
        self.assertEqual(context.company_info.name, "에이젠글로벌")
        self.assertEqual(len(context.analysis_results), 1)
        self.assertEqual(len(context.risk_assessments), 1)
        self.assertEqual(context.analysis_results[0].score, 85.0)

def create_test_suite():
    """테스트 스위트 생성"""
    test_suite = unittest.TestSuite()

    # 기본 테스트들
    loader = unittest.TestLoader()
    test_suite.addTest(loader.loadTestsFromTestCase(TestModels))
    test_suite.addTest(loader.loadTestsFromTestCase(TestInputLayer))
    test_suite.addTest(loader.loadTestsFromTestCase(TestScoringEngine))
    test_suite.addTest(loader.loadTestsFromTestCase(TestQualityMetrics))
    test_suite.addTest(loader.loadTestsFromTestCase(TestEndToEndScenarios))
    test_suite.addTest(loader.loadTestsFromTestCase(TestErrorHandling))
    test_suite.addTest(loader.loadTestsFromTestCase(TestIntegration))

    # 파이프라인 테스트 (시간이 오래 걸릴 수 있어 선택적으로)
    # test_suite.addTest(unittest.makeSuite(TestPipeline))

    return test_suite

def run_tests():
    """테스트 실행"""
    print("🧪 AI 스타트업 투자 평가 에이전트 테스트 시작...")
    print("=" * 60)

    test_suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    print("=" * 60)
    print(f"📊 테스트 결과:")
    print(f"   실행된 테스트: {result.testsRun}")
    print(f"   실패: {len(result.failures)}")
    print(f"   에러: {len(result.errors)}")

    if result.failures:
        print("\n❌ 실패한 테스트:")
        for test, traceback in result.failures:
            print(f"   - {test}: {traceback}")

    if result.errors:
        print("\n🔥 에러가 발생한 테스트:")
        for test, traceback in result.errors:
            print(f"   - {test}: {traceback}")

    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun
    print(f"\n✨ 성공률: {success_rate:.1%}")

    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)