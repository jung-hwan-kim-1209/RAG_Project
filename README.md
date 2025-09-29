# 🦄 AI 스타트업 투자 평가 에이전트

한국어 기반 AI 스타트업 투자 가치 평가 시스템입니다. 10단계 파이프라인을 통해 종합적인 투자 분석 리포트를 제공합니다.

## 🌟 주요 기능

- **지능형 입력 파싱**: 자연어 질의를 구조화된 평가 요청으로 변환
- **다중 소스 데이터 활용**: Vector DB(ChromaDB/FAISS) + 실시간 웹 검색
- **7개 영역 병렬 분석**: 성장성, 비즈니스모델, 기술력, 재무건전성, 팀역량, 규제적합성, 제휴네트워크
- **유니콘 확률 계산**: AI 기반 스타트업 성공 확률 예측
- **리스크 평가**: 6개 카테고리 리스크 분석 (시장, 규제, 경쟁, 재무, 기술, 팀)
- **품질 검증**: 관련성, 근거 품질, 객관성 자동 검증
- **다양한 출력 형식**: 콘솔, JSON, CSV 등 지원

## 🏗️ 시스템 아키텍처

```
📥 INPUT LAYER (입력 파싱)
    ↓
🗃️ KNOWLEDGE BASE LAYER (Vector DB 검색)
    ↓
📋 DOCUMENT RETRIEVAL LAYER (문서 필터링)
    ↓
🌐 EXTERNAL SEARCH LAYER (실시간 정보 수집)
    ↓
⚡ ANALYSIS ENGINE (7개 영역 병렬 분석)
    ↓
📊 SCORING & RANKING ENGINE (점수 계산)
    ↓
⚠️ RISK ASSESSMENT LAYER (리스크 평가)
    ↓
📄 REPORT GENERATION LAYER (리포트 생성)
    ↓
✅ QUALITY CHECK LAYER (품질 검증)
    ↓
📤 OUTPUT LAYER (최종 출력)
```

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone <repository-url>
cd Rag_Project

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 설정

```bash
# 환경 설정 파일 복사
cp .env.example .env

# .env 파일 편집
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. 초기 설정

```bash
# 데이터베이스 초기화
python cli.py setup

# 또는 직접 실행
python -c "from cli import setup; setup('./data')"
```

### 4. 기본 사용법

```bash
# 기본 평가
python cli.py evaluate "토스의 투자 가치를 평가해줘"

# JSON 형식 출력
python cli.py evaluate "카카오 성장성 분석" --format json

# 파일 저장
python cli.py evaluate "배달의민족 리스크 평가" --save --output report.json

# 빠른 평가 (외부 검색 없이)
python cli.py quick "쿠팡 투자 분석"
```

## 📋 CLI 명령어

| 명령어 | 설명 | 예시 |
|--------|------|------|
| `evaluate` | 투자 가치 평가 실행 | `python cli.py evaluate "토스 평가"` |
| `quick` | 빠른 평가 (외부 검색 없이) | `python cli.py quick "카카오 분석"` |
| `setup` | 초기 설정 및 DB 구축 | `python cli.py setup` |
| `add-documents` | 문서 추가 | `python cli.py add-documents ./docs` |
| `search` | 문서 검색 | `python cli.py search "토스"` |
| `status` | 시스템 상태 확인 | `python cli.py status` |
| `config` | 현재 설정 확인 | `python cli.py config` |
| `demo` | 데모 실행 | `python cli.py demo` |

## 💡 사용 예시

### 기본 평가

```python
from pipeline import run_investment_evaluation

result = run_investment_evaluation(
    user_input="토스의 투자 가치를 평가해줘",
    output_format="console"
)
print(result)
```

### 프로그래밍 인터페이스

```python
from pipeline import create_pipeline
from models import PipelineContext, CompanyInfo, ParsedInput

# 파이프라인 생성
pipeline = create_pipeline()

# 실행
result = pipeline.execute_pipeline(
    user_input="카카오 성장성 분석",
    output_format="json",
    save_to_file=True,
    output_path="./reports/kakao_analysis.json"
)
```

## 📊 출력 예시

```
================================================================================
🦄 AI 스타트업 투자 평가 리포트: 토스
================================================================================

📊 EXECUTIVE SUMMARY
----------------------------------------
종합 점수: 87.5/100 (A급)
유니콘 확률: 78.3%
투자 추천: 투자 추천
신뢰도: 85.2%

혁신적인 핀테크 플랫폼으로 강력한 성장 동력을 보유하고 있으며,
탄탄한 기술력과 우수한 팀 역량을 바탕으로 지속가능한 성장이
기대됩니다.

📈 영역별 점수카드
----------------------------------------
growth_analysis          85.0점 (A급)
business_model_analysis  88.5점 (A급)
tech_security_analysis   90.0점 (S급)
financial_health_analysis 85.0점 (A급)
team_evaluation         87.0점 (A급)
regulatory_analysis     80.0점 (B급)
partnership_analysis    85.5점 (A급)

⚠️ 리스크 평가
----------------------------------------
🟢 market_risk: 낮음
🟡 regulatory_risk: 보통
🟡 competitive_risk: 보통
🟢 financial_risk: 낮음
🟢 technology_risk: 낮음
🟢 team_risk: 낮음

💰 투자 권장사항
----------------------------------------
강력한 기술력과 시장 지배력을 바탕으로 한 투자 추천.
규제 리스크는 모니터링 필요하나 전반적으로 우수한
투자 기회로 평가됩니다.
================================================================================
```

## 🛠️ 고급 설정

### 문서 추가

```bash
# 특정 타입 문서 추가
python cli.py add-documents ./company_docs --doc-type company

# IR 보고서 추가
python cli.py add-documents ./ir_reports --doc-type ir
```

### 설정 커스터마이징

`config.py`에서 다음 설정들을 조정할 수 있습니다:

- **분석 가중치**: 각 영역별 중요도 조정
- **모델 설정**: Temperature, Max tokens 등
- **임계값**: 등급 기준점, 품질 검증 기준 등

## 📁 프로젝트 구조

```
Rag_Project/
├── cli.py                 # CLI 인터페이스
├── pipeline.py            # 메인 파이프라인
├── config.py              # 설정 관리
├── models.py              # 데이터 모델
├── requirements.txt       # 의존성
├── layers/                # 파이프라인 레이어들
│   ├── input_layer.py
│   ├── knowledge_base_layer.py
│   ├── document_retrieval_layer.py
│   ├── external_search_layer.py
│   ├── analysis_engine.py
│   ├── scoring_engine.py
│   ├── risk_assessment_layer.py
│   ├── report_generation_layer.py
│   ├── quality_check_layer.py
│   └── output_layer.py
└── data/                  # 데이터 저장소
    ├── documents/
    │   ├── ir_reports/
    │   ├── market_reports/
    │   ├── company_profiles/
    │   └── financials/
    ├── chroma_db/
    └── faiss_index/
```

## 🔧 개발자 가이드

### 새로운 분석기 추가

```python
from layers.analysis_engine import BaseAnalyzer

class CustomAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__("custom_analysis")

    def analyze(self, company_info, documents, external_results):
        # 분석 로직 구현
        return AnalysisResult(...)
```

### 커스텀 리스크 평가기

```python
from layers.risk_assessment_layer import BaseRiskEvaluator

class CustomRiskEvaluator(BaseRiskEvaluator):
    def __init__(self):
        super().__init__("custom_risk")

    def evaluate(self, company_info, documents, external_results, analysis_results):
        # 리스크 평가 로직
        return RiskAssessment(...)
```

## 📈 성능 최적화

- **병렬 처리**: 분석 엔진과 리스크 평가가 병렬 실행
- **캐싱**: Vector DB 결과 캐싱으로 속도 향상
- **배치 처리**: 여러 회사 동시 평가 지원
- **부분 실행**: 특정 레이어만 실행 가능

## 🔒 보안 고려사항

- API 키는 환경 변수로 관리
- 민감한 문서는 로컬 Vector DB에 저장
- 외부 API 호출 시 rate limiting 적용
- 데이터 암호화 옵션 제공

## 🤝 기여 방법

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 라이선스

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 지원

- 이슈 제기: [GitHub Issues](https://github.com/your-repo/issues)
- 이메일: support@your-domain.com
- 문서: [Wiki](https://github.com/your-repo/wiki)

## 🙏 감사의 말

- OpenAI GPT 모델
- ChromaDB & FAISS 벡터 데이터베이스
- LangChain 프레임워크
- Click CLI 라이브러리

---

**🦄 당신의 다음 유니콘 투자를 찾아보세요!**