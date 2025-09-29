"""
Layer 2: KNOWLEDGE BASE LAYER
Vector DB(Chroma/FAISS)에서 관련 문서 검색하는 레이어
"""
import os
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from models import DocumentChunk, PipelineContext
from config import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDBManager:
    """Vector Database 관리 클래스"""

    def __init__(self):
        self.config = get_config()
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=self.config["vector_db"].embedding_model
        )
        self.chroma_db = None
        self.faiss_db = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def initialize_chroma(self) -> None:
        """ChromaDB 초기화"""
        try:
            persist_directory = self.config["vector_db"].chroma_persist_directory
            os.makedirs(persist_directory, exist_ok=True)

            self.chroma_db = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                collection_name=self.config["vector_db"].collection_name
            )
            logger.info("ChromaDB 초기화 완료")
        except Exception as e:
            logger.error(f"ChromaDB 초기화 실패: {e}")

    def initialize_faiss(self) -> None:
        """FAISS 초기화"""
        try:
            index_path = self.config["vector_db"].faiss_index_path
            if os.path.exists(index_path):
                self.faiss_db = FAISS.load_local(index_path, self.embeddings)
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
            result.append({
                "content": doc.page_content,
                "metadata": {
                    **doc.metadata,
                    "source": file_path,
                    "file_type": file_extension
                }
            })

        return result

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

            if self.faiss_db is None:
                self.faiss_db = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas
                )
            else:
                new_db = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas
                )
                self.faiss_db.merge_from(new_db)

            # FAISS 인덱스 저장
            index_path = self.config["vector_db"].faiss_index_path
            os.makedirs(os.path.dirname(index_path), exist_ok=True)
            self.faiss_db.save_local(index_path)

            logger.info(f"FAISS에 {len(documents)}개 문서 추가 완료")
        except Exception as e:
            logger.error(f"FAISS 문서 추가 실패: {e}")

    def search_chroma(self, query: str, k: int = None, filter_dict: Dict[str, Any] = None) -> List[DocumentChunk]:
        """ChromaDB에서 검색"""
        if not self.chroma_db:
            return []

        k = k or self.config["vector_db"].top_k_results

        try:
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
            return []

    def search_faiss(self, query: str, k: int = None) -> List[DocumentChunk]:
        """FAISS에서 검색"""
        if not self.faiss_db:
            return []

        k = k or self.config["vector_db"].top_k_results

        try:
            results = self.faiss_db.similarity_search_with_score(
                query=query,
                k=k
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
        k: int = None
    ) -> List[DocumentChunk]:
        """지식 베이스에서 관련 문서 검색"""

        all_chunks = []

        # ChromaDB 검색
        if use_chroma:
            filter_dict = {}
            if company_name:
                filter_dict["company"] = company_name

            chroma_chunks = self.vector_db_manager.search_chroma(
                query=query,
                k=k,
                filter_dict=filter_dict if filter_dict else None
            )
            all_chunks.extend(chroma_chunks)

        # FAISS 검색
        if use_faiss:
            faiss_chunks = self.vector_db_manager.search_faiss(query=query, k=k)
            all_chunks.extend(faiss_chunks)

        # 중복 제거 및 정렬
        unique_chunks = self._deduplicate_chunks(all_chunks)
        sorted_chunks = sorted(unique_chunks, key=lambda x: x.similarity_score, reverse=True)

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

    # 지식 베이스에서 관련 문서 검색
    retrieved_chunks = knowledge_base.search_knowledge_base(
        query=search_query,
        company_name=company_name
    )

    context.retrieved_documents = retrieved_chunks

    # 처리 단계 기록
    context.processing_steps.append(
        f"KNOWLEDGE_BASE_LAYER: {len(retrieved_chunks)}개 관련 문서 검색 완료"
    )

    return context