"""
AI 스타트업 투자 평가 에이전트 CLI 인터페이스
"""
import click
import os
from pathlib import Path
from dotenv import load_dotenv

from pipeline import run_investment_evaluation, create_pipeline
from layers.knowledge_base_layer import create_knowledge_base_layer

# 환경 변수 로드
load_dotenv()

@click.group()
@click.version_option("1.0.0")
def cli():
    """ AI 스타트업 투자 평가 에이전트"""
    pass

@cli.command()
@click.argument('company_query')
@click.option('--format', '-f', default='console',
              type=click.Choice(['console', 'json', 'summary']),
              help='출력 형식')
@click.option('--save', '-s', is_flag=True, help='파일로 저장')
@click.option('--output', '-o', help='출력 파일 경로')
@click.option('--pdf', '-p', help='PDF 보고서 저장 경로')
@click.option('--skip-external', is_flag=True, help='외부 검색 건너뛰기')
@click.option('--retries', default=1, help='최대 재시도 횟수')
@click.option('--verbose', '-v', is_flag=True, help='상세 로그 출력')
def evaluate(company_query, format, save, output, pdf, skip_external, retries, verbose):
    """스타트업 투자 가치 평가

    예시:
        투자평가 "토스의 투자 가치를 평가해줘"
        투자평가 "카카오 성장성 분석" --format json
        투자평가 "배달의민족 리스크 분석" --save --output report.json
        투자평가 "토스의 투자 가치를 평가해줘" --pdf report.pdf
    """

    # 로깅 레벨 설정
    if verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        click.echo("[오류] OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.", err=True)
        click.echo("[정보] .env 파일에 OPENAI_API_KEY=your_api_key 를 설정해주세요.", err=True)
        return
    
    if not os.getenv("HF_TOKEN"):
        click.echo("[오류] HF_TOKEN 환경 변수가 설정되지 않았습니다.", err=True)
        click.echo("[정보] .env 파일에 HF_TOKEN=your_hf_token 를 설정해주세요.", err=True)
        return

    try:
        click.echo(f"[시작] {company_query} 투자 평가를 시작합니다...")

        # 파이프라인 실행
        result = run_investment_evaluation(
            user_input=company_query,
            output_format=format,
            save_to_file=save,
            output_path=output,
            pdf_output_path=pdf,
            skip_external_search=skip_external,
            max_retries=retries
        )

        click.echo(result)

        if save and output:
            click.echo(f" 리포트가 {output}에 저장되었습니다.")

        if pdf:
            click.echo(f" PDF 보고서가 {pdf}에 저장되었습니다.")

    except Exception as e:
        click.echo(f"[오류] 오류 발생: {str(e)}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)

@cli.command()
@click.option('--data-dir', default='./data', help='데이터 디렉토리 경로')
def setup(data_dir):
    """초기 설정 및 데이터베이스 구축"""

    click.echo(" AI 투자 평가 에이전트 초기 설정을 시작합니다...")

    try:
        # 데이터 디렉토리 생성
        data_path = Path(data_dir)

        # 필요한 디렉토리들 생성
        directories = [
            data_path / "documents" / "ir_reports",
            data_path / "documents" / "market_reports",
            data_path / "documents" / "company_profiles",
            data_path / "documents" / "financials",
            data_path / "chroma_db",
            data_path / "faiss_index"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            click.echo(f"[생성] 디렉토리 생성: {directory}")

        # Vector DB 초기화
        click.echo("Vector Database 초기화 중...")
        knowledge_base = create_knowledge_base_layer()
        knowledge_base.setup_database()

        click.echo("[완료] 초기 설정 완료!")
        click.echo(f" 데이터 디렉토리: {data_path.absolute()}")
        click.echo("[정보] 문서들을 해당 디렉토리에 추가한 후 사용하세요.")

    except Exception as e:
        click.echo(f"[오류] 설정 오류: {str(e)}", err=True)

@cli.command()
@click.argument('documents_path')
@click.option('--doc-type', default='all',
              type=click.Choice(['ir', 'market', 'company', 'financial', 'all']),
              help='문서 타입')
def add_documents(documents_path, doc_type):
    """문서를 Vector Database에 추가"""

    click.echo(f"[문서추가] 문서 추가 중: {documents_path}")

    try:
        knowledge_base = create_knowledge_base_layer()
        documents = knowledge_base.vector_db_manager.load_documents_from_directory(documents_path)

        if documents:
            knowledge_base.vector_db_manager.add_documents_to_chroma(documents)
            knowledge_base.vector_db_manager.add_documents_to_faiss(documents)
            click.echo(f"[완료] {len(documents)}개 문서가 추가되었습니다.")
        else:
            click.echo("[경고] 추가할 문서를 찾을 수 없습니다.")

    except Exception as e:
        click.echo(f"[오류] 문서 추가 오류: {str(e)}", err=True)

@cli.command()
@click.argument('company_name')
@click.option('--top-k', default=10, help='검색할 문서 개수')
def search(company_name, top_k):
    """회사 관련 문서 검색"""

    try:
        knowledge_base = create_knowledge_base_layer()
        results = knowledge_base.search_knowledge_base(
            query=f"{company_name} 투자 분석",
            company_name=company_name,
            k=top_k
        )

        click.echo(f"[검색] {company_name} 관련 문서 {len(results)}개 발견:")

        for i, result in enumerate(results, 1):
            click.echo(f"\n{i}. {result.source}")
            click.echo(f"   유사도: {result.similarity_score:.3f}")
            click.echo(f"   내용: {result.content[:100]}...")

    except Exception as e:
        click.echo(f"[오류] 검색 오류: {str(e)}", err=True)

@cli.command()
@click.argument('company_query')
def quick(company_query):
    """빠른 평가 (외부 검색 없이)"""

    try:
        click.echo(f" {company_query} 빠른 평가 중...")

        result = run_investment_evaluation(
            user_input=company_query,
            output_format="console",
            skip_external_search=True,
            max_retries=0
        )

        click.echo(result)

    except Exception as e:
        click.echo(f"[오류] 오류 발생: {str(e)}", err=True)

@cli.command()
def status():
    """시스템 상태 확인"""

    click.echo("시스템 상태 확인 중...")

    # API 키 확인
    api_key_status = "[설정됨]" if os.getenv("OPENAI_API_KEY") else "[미설정]"
    hf_token_status = "[설정됨]" if os.getenv("HF_TOKEN") else "[미설정]"
    click.echo(f"OpenAI API Key: {api_key_status}")
    click.echo(f"HuggingFace Token: {hf_token_status}")

    # 데이터 디렉토리 확인
    data_dirs = [
        "./data/documents/ir_reports",
        "./data/documents/market_reports",
        "./data/documents/company_profiles",
        "./data/documents/financials",
        "./data/chroma_db",
        "./data/faiss_index"
    ]

    click.echo("\n[상태] 데이터 디렉토리:")
    for directory in data_dirs:
        exists = "[존재]" if Path(directory).exists() else "[누락]"
        click.echo(f"  {exists} {directory}")

    # Vector DB 상태 확인
    try:
        knowledge_base = create_knowledge_base_layer()

        # ChromaDB 문서 수 확인
        if knowledge_base.vector_db_manager.chroma_db:
            chroma_count = knowledge_base.vector_db_manager.chroma_db._collection.count()
            click.echo(f"\n[데이터베이스] ChromaDB: {chroma_count}개 문서")
        else:
            click.echo(f"\n[데이터베이스] ChromaDB: 초기화되지 않음")

    except Exception as e:
        click.echo(f"\n[오류] Vector DB 상태 확인 오류: {e}")

@cli.command()
def config():
    """현재 설정 확인"""

    from config import get_config
    import json

    config_data = get_config()

    click.echo("[설정] 현재 설정:")
    click.echo(f"모델: {config_data['model'].model_name}")
    click.echo(f"Temperature: {config_data['model'].temperature}")
    click.echo(f"Vector DB: ChromaDB + FAISS")
    click.echo(f"임베딩 모델: {config_data['vector_db'].embedding_model}")
    click.echo(f"Top-K 결과: {config_data['vector_db'].top_k_results}")

    click.echo("\n[설정] 분석 가중치 (4개 분석기):")
    weights = config_data['analysis_weights']
    click.echo(f"  성장성: {weights.growth_weight:.0%}")
    click.echo(f"  비즈니스모델: {weights.business_model_weight:.0%}")
    click.echo(f"  기술/보안: {weights.tech_security_weight:.0%}")
    click.echo(f"  재무건전성: {weights.financial_health_weight:.0%}")

@cli.command()
def demo():
    """데모 실행"""

    demo_queries = [
        "토스의 투자 가치를 평가해줘",
        "카카오 성장성 분석",
        "배달의민족 리스크 평가"
    ]

    click.echo(" 데모 실행 - 샘플 쿼리들:")

    for i, query in enumerate(demo_queries, 1):
        click.echo(f"\n{i}. {query}")

        if click.confirm("이 쿼리를 실행하시겠습니까?"):
            try:
                result = run_investment_evaluation(
                    user_input=query,
                    output_format="console",
                    skip_external_search=True,  # 데모에서는 빠른 실행
                    max_retries=0
                )
                click.echo(result)

                if not click.confirm("다음 데모를 계속하시겠습니까?"):
                    break

            except Exception as e:
                click.echo(f"[오류] 데모 실행 오류: {e}")

if __name__ == '__main__':
    cli()