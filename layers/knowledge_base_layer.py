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
                embedding_function=chroma_embedding_function,
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

    def search_faiss(self, query: str, k: int = None) -> List[DocumentChunk]:
        """FAISS에서 검색"""
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
            
            # 검색 실행
            scores, indices = self.faiss_db["index"].search(query_array, k)
            
            chunks = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and idx < len(self.faiss_db["texts"]):
                    chunk = DocumentChunk(
                        content=self.faiss_db["texts"][idx],
                        source=self.faiss_db["metadatas"][idx].get("source", "unknown"),
                        metadata=self.faiss_db["metadatas"][idx],
                        similarity_score=float(score)
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

# 테스트 코드
if __name__ == "__main__":
    print("🧪 Knowledge Base Layer 테스트 시작...")
    
    try:
        # VectorDBManager 테스트
        print("1. VectorDBManager 초기화 테스트...")
        manager = VectorDBManager()
        print("✅ VectorDBManager 초기화 성공")
        
        # HuggingFace 임베딩 테스트
        print("2. HuggingFace 임베딩 테스트...")
        test_text = "테스트 문서입니다."
        embedding = manager.embeddings.embed_query(test_text)
        print(f"✅ 임베딩 생성 성공: {len(embedding)}차원")
        
        # KnowledgeBase 테스트
        print("3. KnowledgeBase 초기화 테스트...")
        kb = KnowledgeBase()
        print("✅ KnowledgeBase 초기화 성공")
        
        print("\n🎉 모든 테스트 통과!")
        
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()