"""
Layer 2: KNOWLEDGE BASE LAYER
Vector DB(Chroma/FAISS)에서 관련 문서 검색하는 레이어
"""
import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from huggingface_hub import InferenceClient
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import faiss   # 파일 제일 위에 넣기
import pickle  # 같이 상단으로 올려두기

from models import DocumentChunk, PipelineContext
from config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuggingFaceEmbeddings:
    """HuggingFace Inference API를 사용한 임베딩 클래스"""
    
    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.client = InferenceClient(
            provider="hf-inference",
            api_key=api_key,
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서들을 임베딩으로 변환"""
        embeddings = []
        for text in texts:
            try:
                # BAAI/bge-m3 모델 사용
                embedding = self.client.feature_extraction(text, model=self.model_name)
                embeddings.append(embedding.tolist() if hasattr(embedding, 'tolist') else embedding)
            except Exception as e:
                logger.error(f"임베딩 생성 오류: {e}")
                # 오류 시 0 벡터 반환 (BAAI/bge-m3는 1024차원)
                embeddings.append([0.0] * 1024)
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """단일 쿼리를 임베딩으로 변환"""
        try:
            embedding = self.client.feature_extraction(text, model=self.model_name)
            return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        except Exception as e:
            logger.error(f"쿼리 임베딩 생성 오류: {e}")
            return [0.0] * 1024

class VectorDBManager:
    """Vector Database 관리 클래스"""

    def __init__(self):
        self.config = get_config()
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN 환경변수가 설정되지 않았습니다.")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config["vector_db"].embedding_model,
            api_key=hf_token
        )
        self.chroma_db = None
        self.faiss_db = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("TEXT_CHUNK_SIZE", "1000")),
            chunk_overlap=int(os.getenv("TEXT_CHUNK_OVERLAP", "200")),
            length_function=len
        )

    def initialize_chroma(self) -> None:
        """ChromaDB 초기화"""
        try:
            persist_directory = self.config["vector_db"].chroma_persist_directory
            os.makedirs(persist_directory, exist_ok=True)

            # ChromaDB용 커스텀 임베딩 함수 생성
            def chroma_embedding_function(texts):
                return self.embeddings.embed_documents(texts)

            self.chroma_db = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,  # 클래스 객체를 직접 넘김
                collection_name=self.config["vector_db"].collection_name
            )
            logger.info("ChromaDB 초기화 완료")
        except Exception as e:
            logger.error(f"ChromaDB 초기화 실패: {e}")

    def initialize_faiss(self) -> None:
        """FAISS 초기화"""
        try:
            index_path = self.config["vector_db"].faiss_index_path
            faiss_file = f"{index_path}.faiss"
            pkl_file = f"{index_path}.pkl"
            
            if os.path.exists(faiss_file) and os.path.exists(pkl_file):
                import faiss
                import pickle
                
                # FAISS 인덱스 로드
                index = faiss.read_index(faiss_file)
                
                # 메타데이터 로드
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                self.faiss_db = {
                    "index": index,
                    "texts": data["texts"],
                    "metadatas": data["metadatas"]
                }
                logger.info("FAISS 인덱스 로드 완료")
            else:
                logger.info("FAISS 인덱스가 존재하지 않음")
        except Exception as e:
            logger.error(f"FAISS 초기화 실패: {e}")

    def load_documents_from_directory(self, directory_path: str, file_types: List[str] = None) -> List[Dict[str, Any]]:
        """디렉토리에서 문서들을 로드"""
        if file_types is None:
            file_types = [".pdf", ".txt", ".md"]

        documents = []
        directory = Path(directory_path)

        if not directory.exists():
            logger.warning(f"디렉토리가 존재하지 않음: {directory_path}")
            return documents

        for file_path in directory.rglob("*"):
            if file_path.suffix.lower() in file_types:
                try:
                    docs = self._load_single_document(str(file_path))
                    documents.extend(docs)
                except Exception as e:
                    logger.error(f"문서 로드 실패: {file_path}, 오류: {e}")

        return documents

    def _load_single_document(self, file_path: str) -> List[Dict[str, Any]]:
        """개별 문서 로드"""
        file_extension = Path(file_path).suffix.lower()

        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        documents = loader.load()
        split_docs = self.text_splitter.split_documents(documents)

        result = []
        for doc in split_docs:
            # 유니코드 특수 문자 제거 (이모지 및 Private Use Area 문자)
            clean_content = self._clean_unicode(doc.page_content)

            result.append({
                "content": clean_content,
                "metadata": {
                    **doc.metadata,
                    "source": file_path,
                    "file_type": file_extension
                }
            })

        return result

    def _clean_unicode(self, text: str) -> str:
        """유니코드 특수 문자 제거"""
        import re

        # 이모지만 제거 (Private Use Area 제외 - 한글 보존)
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags
            u"\U00002700-\U000027BF"  # Dingbats
            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols
            u"\U0001FA70-\U0001FAFF"  # Extended-A
            u"\U00002600-\U000026FF"  # Miscellaneous Symbols
            u"\U0001F700-\U0001F77F"  # Alchemical Symbols
            "]+", flags=re.UNICODE)

        # Private Use Area에서 특정 bullet point 문자만 제거
        text = emoji_pattern.sub('', text)
        text = text.replace('\uf0b7', '')  # bullet point
        text = text.replace('\uf0a7', '')  # square bullet

        return text

    def add_documents_to_chroma(self, documents: List[Dict[str, Any]]) -> None:
        """ChromaDB에 문서 추가"""
        if not self.chroma_db:
            self.initialize_chroma()

        try:
            texts = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]

            self.chroma_db.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            logger.info(f"ChromaDB에 {len(documents)}개 문서 추가 완료")
        except Exception as e:
            logger.error(f"ChromaDB 문서 추가 실패: {e}")

    def add_documents_to_faiss(self, documents: List[Dict[str, Any]]) -> None:
        """FAISS에 문서 추가"""
        try:
            texts = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            
            # HuggingFace 임베딩으로 벡터 생성
            embeddings = self.embeddings.embed_documents(texts)
            embeddings_array = np.array(embeddings).astype('float32')

            if self.faiss_db is None:
                # FAISS 인덱스 직접 생성
                import faiss
                dimension = embeddings_array.shape[1]
                index = faiss.IndexFlatIP(dimension)  # Inner Product (cosine similarity)
                
                # 정규화 (cosine similarity를 위해)
                faiss.normalize_L2(embeddings_array)
                index.add(embeddings_array)
                
                # FAISS 인덱스와 메타데이터를 저장
                index_path = self.config["vector_db"].faiss_index_path
                os.makedirs(os.path.dirname(index_path), exist_ok=True)
                faiss.write_index(index, f"{index_path}.faiss")
                
                # 메타데이터 저장
                import pickle
                with open(f"{index_path}.pkl", 'wb') as f:
                    pickle.dump({"texts": texts, "metadatas": metadatas}, f)
                
                # self.faiss_db에 저장 (검색용)
                self.faiss_db = {"index": index, "texts": texts, "metadatas": metadatas}
            else:
                # 기존 인덱스에 추가
                faiss.normalize_L2(embeddings_array)
                self.faiss_db["index"].add(embeddings_array)
                self.faiss_db["texts"].extend(texts)
                self.faiss_db["metadatas"].extend(metadatas)
                
                # 업데이트된 인덱스 저장
                index_path = self.config["vector_db"].faiss_index_path
                faiss.write_index(self.faiss_db["index"], f"{index_path}.faiss")
                
                import pickle
                with open(f"{index_path}.pkl", 'wb') as f:
                    pickle.dump({"texts": self.faiss_db["texts"], "metadatas": self.faiss_db["metadatas"]}, f)

            logger.info(f"FAISS에 {len(documents)}개 문서 추가 완료")
        except Exception as e:
            logger.error(f"FAISS 문서 추가 실패: {e}")

    def search_chroma(self, query: str, k: int = None, filter_dict: Dict[str, Any] = None) -> List[DocumentChunk]:
        """ChromaDB에서 검색"""
        if not self.chroma_db:
            return []

        k = k or self.config["vector_db"].top_k_results

        try:
            # ChromaDB의 similarity_search_with_score는 커스텀 임베딩 함수와 호환되지 않을 수 있음
            # 대신 직접 구현
            query_embedding = self.embeddings.embed_query(query)
            
            # ChromaDB에서 검색 (임베딩 함수가 자동으로 호출됨)
            results = self.chroma_db.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )

            chunks = []
            for doc, score in results:
                chunk = DocumentChunk(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    metadata=doc.metadata,
                    similarity_score=score
                )
                chunks.append(chunk)

            return chunks
        except Exception as e:
            logger.error(f"ChromaDB 검색 실패: {e}")
            # 대안으로 FAISS 검색 사용
            return self.search_faiss(query, k)

    def search_faiss(self, query: str, k: int = None, company_name: str = "") -> List[DocumentChunk]:
        """FAISS에서 검색 (회사명 필터링 지원)"""
        if not self.faiss_db:
            return []

        k = k or self.config["vector_db"].top_k_results

        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embeddings.embed_query(query)
            query_array = np.array([query_embedding]).astype('float32')

            # 정규화
            import faiss
            faiss.normalize_L2(query_array)

            # 검색 실행 (더 많이 가져온 후 필터링)
            search_k = k * 3 if company_name else k
            scores, indices = self.faiss_db["index"].search(query_array, search_k)

            chunks = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and idx < len(self.faiss_db["texts"]):
                    metadata = self.faiss_db["metadatas"][idx]

                    # 회사명 필터링 (메타데이터 또는 소스 경로에서 확인)
                    if company_name:
                        source = metadata.get("source", "")
                        content = self.faiss_db["texts"][idx]

                        # 1) 메타데이터에 회사명 필드가 있는 경우
                        if "company" in metadata and metadata["company"] != company_name:
                            continue

                        # 2) 회사명 매칭 맵 (한글 <-> 영문)
                        company_mapping = {
                            "핀다": ["finda", "핀다"],
                            "finda": ["finda", "핀다"],
                            "8퍼센트": ["8percent", "8퍼센트"],
                            "8percent": ["8percent", "8퍼센트"],
                            "뱅크샐러드": ["banksalad", "뱅크샐러드"],
                            "banksalad": ["banksalad", "뱅크샐러드"],
                            "하이카": ["hicar", "hicarcompany", "하이카", "하이카컴퍼니"],
                            "하이카컴퍼니": ["hicar", "hicarcompany", "하이카", "하이카컴퍼니"],
                            "hicar": ["hicar", "hicarcompany", "하이카", "하이카컴퍼니"],
                            "hicarcompany": ["hicar", "hicarcompany", "하이카", "하이카컴퍼니"]
                        }

                        # 3) 매칭할 회사명들 생성
                        search_names = [company_name.lower()]
                        for key, values in company_mapping.items():
                            if company_name in values or company_name.lower() in [v.lower() for v in values]:
                                search_names.extend([v.lower() for v in values])
                                break

                        # 4) 파일 경로나 내용에 관련 회사명이 있는지 확인
                        source_lower = source.lower()
                        content_lower = content[:500].lower()  # 처음 500자만 확인

                        if not any(name in source_lower or name in content_lower for name in search_names):
                            continue

                    chunk = DocumentChunk(
                        content=self.faiss_db["texts"][idx],
                        source=metadata.get("source", "unknown"),
                        metadata=metadata,
                        similarity_score=float(score)
                    )
                    chunks.append(chunk)

                    # 필요한 개수만큼 모았으면 종료
                    if len(chunks) >= k:
                        break

            return chunks
        except Exception as e:
            logger.error(f"FAISS 검색 실패: {e}")
            return []

    def setup_initial_database(self) -> None:
        """초기 데이터베이스 설정"""
        logger.info("초기 데이터베이스 설정 시작")

        # 각 문서 타입별 디렉토리에서 문서 로드
        document_paths = self.config["document_paths"]

        # IR 문서들
        ir_docs = self.load_documents_from_directory(document_paths.ir_documents)
        if ir_docs:
            self.add_documents_to_chroma(ir_docs)
            self.add_documents_to_faiss(ir_docs)

        # 시장 보고서
        market_docs = self.load_documents_from_directory(document_paths.market_reports)
        if market_docs:
            self.add_documents_to_chroma(market_docs)
            self.add_documents_to_faiss(market_docs)

        # 회사 프로필
        company_docs = self.load_documents_from_directory(document_paths.company_profiles)
        if company_docs:
            self.add_documents_to_chroma(company_docs)
            self.add_documents_to_faiss(company_docs)

        # 재무 문서
        financial_docs = self.load_documents_from_directory(document_paths.financial_statements)
        if financial_docs:
            self.add_documents_to_chroma(financial_docs)
            self.add_documents_to_faiss(financial_docs)

        logger.info("초기 데이터베이스 설정 완료")

class KnowledgeBase:
    """지식 베이스 메인 클래스"""

    def __init__(self):
        self.vector_db_manager = VectorDBManager()
        self.vector_db_manager.initialize_chroma()
        self.vector_db_manager.initialize_faiss()

    def search_knowledge_base(
        self,
        query: str,
        company_name: str = "",
        use_chroma: bool = True,
        use_faiss: bool = True,
        k: int = None,
        similarity_threshold: float = 0.0
    ) -> List[DocumentChunk]:
        """지식 베이스에서 관련 문서 검색"""

        all_chunks = []

        # ChromaDB 검색 (필터 없이 일단 검색)
        if use_chroma:
            try:
                chroma_chunks = self.vector_db_manager.search_chroma(
                    query=query,
                    k=k,
                    filter_dict=None  # 필터 제거
                )
                all_chunks.extend(chroma_chunks)
                logger.info(f"ChromaDB에서 {len(chroma_chunks)}개 문서 검색")
            except Exception as e:
                logger.error(f"ChromaDB 검색 오류: {e}")

        # FAISS 검색 (회사명 필터링 적용)
        if use_faiss:
            try:
                faiss_chunks = self.vector_db_manager.search_faiss(
                    query=query,
                    k=k,
                    company_name=company_name
                )
                all_chunks.extend(faiss_chunks)
                logger.info(f"FAISS에서 {len(faiss_chunks)}개 문서 검색")
            except Exception as e:
                logger.error(f"FAISS 검색 오류: {e}")

        # 중복 제거
        unique_chunks = self._deduplicate_chunks(all_chunks)

        # 유사도 threshold 필터링
        filtered_chunks = [
            chunk for chunk in unique_chunks
            if chunk.similarity_score >= similarity_threshold
        ]

        # 정렬
        sorted_chunks = sorted(filtered_chunks, key=lambda x: x.similarity_score, reverse=True)

        # 상위 k개 반환
        final_k = k or self.vector_db_manager.config["vector_db"].top_k_results
        return sorted_chunks[:final_k]

    def _deduplicate_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """중복 청크 제거"""
        seen_content = set()
        unique_chunks = []

        for chunk in chunks:
            content_hash = hash(chunk.content[:100])  # 첫 100자로 해시 생성
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_chunks.append(chunk)

        return unique_chunks

    def setup_database(self) -> None:
        """데이터베이스 초기 설정"""
        self.vector_db_manager.setup_initial_database()

def create_knowledge_base_layer() -> KnowledgeBase:
    """Knowledge Base Layer 생성자"""
    return KnowledgeBase()

def process_knowledge_base_layer(context: PipelineContext) -> PipelineContext:
    """Knowledge Base Layer 처리 함수"""
    knowledge_base = create_knowledge_base_layer()

    # 검색 쿼리 생성
    company_name = context.company_info.name
    evaluation_type = context.parsed_input.evaluation_type.value

    search_query = f"{company_name} {evaluation_type} 투자 평가 분석"

    print(f"[검색] 내부 문서 검색 중: '{company_name}' 관련 문서...")

    # 지식 베이스에서 관련 문서 검색
    retrieved_chunks = knowledge_base.search_knowledge_base(
        query=search_query,
        company_name=company_name,
        similarity_threshold=0.0  # 모든 결과 포함
    )

    context.retrieved_documents = retrieved_chunks

    # CLI 출력: 참고한 내부 문서 목록
    print("\n" + "="*80)
    print(f"[문서] 참고한 내부 문서 ({len(retrieved_chunks)}개)")
    print("="*80)
    for i, chunk in enumerate(retrieved_chunks[:10], 1):  # 상위 10개만 출력
        print(f"\n[{i}] 출처: {chunk.source}")
        print(f"    유사도: {chunk.similarity_score:.3f}")
        preview = chunk.content[:150].replace('\n', ' ')
        print(f"    내용: {preview}...")
    if len(retrieved_chunks) > 10:
        print(f"\n... 외 {len(retrieved_chunks) - 10}개 더 참고")
    print("="*80 + "\n")

    # 처리 단계 기록
    context.processing_steps.append(
        f"KNOWLEDGE_BASE_LAYER: {len(retrieved_chunks)}개 관련 문서 검색 완료"
    )

    return context

# 테스트 코드
if __name__ == "__main__":
    print("[테스트] Knowledge Base Layer 테스트 시작...")
    
    try:
        # VectorDBManager 테스트
        print("1. VectorDBManager 초기화 테스트...")
        manager = VectorDBManager()
        print("[완료] VectorDBManager 초기화 성공")
        
        # HuggingFace 임베딩 테스트
        print("2. HuggingFace 임베딩 테스트...")
        test_text = "테스트 문서입니다."
        embedding = manager.embeddings.embed_query(test_text)
        print(f"[완료] 임베딩 생성 성공: {len(embedding)}차원")
        
        # KnowledgeBase 테스트
        print("3. KnowledgeBase 초기화 테스트...")
        kb = KnowledgeBase()
        print("[완료] KnowledgeBase 초기화 성공")
        
        print("\n[완료] 모든 테스트 통과!")
        
    except Exception as e:
        print(f"[오류] 테스트 실패: {e}")
        import traceback
        traceback.print_exc()