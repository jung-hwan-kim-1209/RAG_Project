# AI Startup Investment Evaluation Agent

본 프로젝트는 금융 핀테크 AI스타트업에 대한 투자 가능성을 자동으로 평가하는 에이전트를 설계하고 구현한 실습 프로젝트입니다.

## Overview

- **Objective**: AI 스타트업의 기술력, 시장성, 리스크 등을 기준으로 투자 적합성 분석
- **Method**: Multi-Layer RAG Pipeline + Parallel Analysis Engine
- **Tools**: LangChain, HuggingFace Embeddings, Vector Databases, External APIs

## Features

- **📄 PDF 자료 기반 정보 추출**: IR 자료, 시장 보고서, 회사 프로필 등 다양한 문서에서 정보 추출
- **🔍 지능형 문서 검색**: FAISS/ChromaDB를 활용한 벡터 유사도 검색
- **🌐 실시간 외부 정보 수집**: 네이버 뉴스, 구글 뉴스, 투자 정보 등 최신 데이터 수집
- **📊 4개 영역 병렬 분석**: 성장성, 비즈니스 모델, 기술력/보안성, 재무건전성
- **⚠️ 리스크 평가**: 6개 리스크 영역에 대한 종합적 위험도 분석
- **🎯 투자 추천**: 유니콘 확률 계산 및 투자 추천/보류/회피 판단
- **✅ 품질 검증**: 관련성, 근거 품질, 객관성 자동 검증

## Tech Stack

| Category   | Details                      |
|------------|------------------------------|
| **Framework** | LangChain, Python 3.8+ |
| **LLM** | GPT-4o via OpenAI API |
| **Embeddings** | BAAI/bge-m3 via HuggingFace |
| **Vector DB** | FAISS, ChromaDB |
| **External APIs** | Naver News API |
| **Testing** | unittest, pytest |

## Architecture

<img width="801" height="683" alt="image" src="https://github.com/user-attachments/assets/8dc1a78c-d25e-49bb-a387-4b2a464bf08a" />

## Directory Structure

```
RAG_Project/
├── data/                          # 데이터 저장소
│   ├── documents/                 # PDF 문서들
│   │   ├── ir_reports/           # IR 자료
│   │   ├── market_reports/       # 시장 보고서
│   │   ├── company_profiles/     # 회사 프로필
│   │   └── financials/           # 재무 자료
│   ├── chroma_db/                # ChromaDB 저장소
│   └── faiss_index.*             # FAISS 인덱스
├── layers/                        # 핵심 레이어들
│   ├── input_layer.py            # 입력 파싱
│   ├── knowledge_base_layer.py   # 지식 베이스
│   ├── document_retrieval_layer.py # 문서 검색
│   ├── external_search_layer.py  # 외부 검색
│   ├── analysis_engine.py        # 분석 엔진
│   ├── risk_assessment_layer.py  # 리스크 평가
│   ├── scoring_engine.py         # 점수 계산
│   ├── report_generation_layer.py # 리포트 생성
│   ├── quality_check_layer.py    # 품질 검증
│   └── output_layer.py           # 출력 처리
├── models.py                      # 데이터 모델
├── config.py                      # 설정 관리
├── pipeline.py                    # 파이프라인 오케스트레이션
├── cli.py                         # CLI 인터페이스
├── test_pipeline.py               # 테스트 코드
├── requirements.txt               # 의존성
└── README.md                      # 프로젝트 문서
```

## Installation

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd RAG_Project

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 변수들을 설정하세요:

```env
# OpenAI API
OPENAI_API_KEY=your_openai_api_key
MODEL_NAME=gpt-4o
MODEL_TEMPERATURE=0.1

# HuggingFace
HF_TOKEN=your_huggingface_token
EMBEDDING_MODEL=BAAI/bge-m3

# External APIs
NAVER_CLIENT_ID=your_naver_client_id
NAVER_CLIENT_SECRET=your_naver_client_secret
SERPAPI_API_KEY=your_serpapi_key

# Vector Database
CHROMA_PERSIST_DIRECTORY=./data/chroma_db
FAISS_INDEX_PATH=./data/faiss_index
TOP_K_RESULTS=10

# Analysis Weights
GROWTH_WEIGHT=0.30
BUSINESS_MODEL_WEIGHT=0.25
TECH_SECURITY_WEIGHT=0.25
FINANCIAL_HEALTH_WEIGHT=0.20
```

### 3. 데이터베이스 초기화

```bash
# CLI를 통한 초기 설정
python cli.py setup

# 또는 문서 추가
python cli.py add-documents ./data/documents/ir_reports --doc-type ir
```

## Usage

### CLI 사용법

```bash
# 기본 투자 평가
python cli.py evaluate "토스 투자 평가해줘"

# JSON 형식으로 출력
python cli.py evaluate "카카오 성장성 분석" --format json

# 파일로 저장
python cli.py evaluate "배달의민족 전체 평가" --save --output results.json

# 외부 검색 건너뛰기
python cli.py evaluate "네이버 기술 분석" --skip-external

# 상태 확인
python cli.py status

# 설정 확인
python cli.py config
```

### Python API 사용법

```python
from pipeline import run_investment_evaluation

# 기본 사용
result = run_investment_evaluation("토스 투자 평가해줘")
print(result)

# 옵션 설정
result = run_investment_evaluation(
    "카카오 성장성 분석",
    output_format="json",
    save_to_file=True,
    output_path="kakao_analysis.json"
)
```

## Analysis Framework

### 4개 핵심 분석 영역

1. **성장성 분석 (Growth Analysis)**
   - 매출 성장률
   - 시장 확장 가능성
   - 고객 증가율
   - 제품/서비스 확장성
   - 시장 점유율 증가 잠재력

2. **비즈니스 모델 분석 (Business Model Analysis)**
   - 수익 모델의 지속가능성
   - 고객 획득 비용 vs 고객 생애 가치
   - 시장 진입 장벽
   - 경쟁 우위 요소
   - 수익화 구조의 명확성

3. **기술력/보안성 분석 (Tech/Security Analysis)**
   - 핵심 기술의 차별성
   - 특허 및 지적재산권
   - 개발팀의 기술 역량
   - 보안 체계 및 데이터 보호
   - 기술 혁신성 및 미래 대응력

4. **재무건전성 분석 (Financial Health Analysis)**
   - 현금 보유 현황 및 운영 자금
   - 매출 성장률 및 수익성
   - 투자 유치 이력 및 밸류에이션
   - 비용 구조 및 효율성
   - 재무 리스크 요소

### 6개 리스크 영역

- **시장 리스크**: 시장 변화, 경쟁 심화
- **규제 리스크**: 정책 변화, 규제 강화
- **경쟁 리스크**: 신규 진입자, 기술 대체
- **재무 리스크**: 자금 조달, 현금 흐름
- **기술 리스크**: 기술 노후화, 보안 사고
- **팀 리스크**: 핵심 인재 이탈, 리더십

## Output Format

### 투자 평가 리포트 예시

```
🦄 AI 스타트업 투자 평가 리포트

📊 기본 정보
회사명: 토스
업종: 핀테크
평가일: 2024-01-15

🎯 종합 평가
총점: 85/100 (A등급)
유니콘 확률: 65%
투자 추천: 투자 추천

📈 영역별 점수
• 성장성: 88점 (A)
• 비즈니스 모델: 82점 (A)
• 기술력/보안성: 85점 (A)
• 재무건전성: 80점 (B)

⚠️ 주요 리스크
• 규제 리스크: 보통
• 경쟁 리스크: 낮음
• 기술 리스크: 낮음



## Performance

- **처리 시간**: 평균 2-3분 (외부 검색 포함)
- **정확도**: 85% 이상 (품질 검증 기준)
- **동시 처리**: 최대 4개 분석기 병렬 실행


## Future Improvements

- [ ] 다국어 지원 (영어, 중국어)
- [ ] 실시간 시장 데이터 연동
- [ ] 웹 인터페이스 개발
- [ ] 모바일 앱 개발
- [ ] 고급 시각화 대시보드
- [ ] 투자 포트폴리오 관리 기능

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request



*이 프로젝트는 교육 및 연구 목적으로 개발되었습니다. 실제 투자 결정에 사용하기 전에 전문가의 조언을 구하시기 바랍니다.*
