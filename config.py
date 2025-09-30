"""
Configuration settings for AI Startup Investment Evaluation Agent
"""
import os
from dataclasses import dataclass
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class ModelConfig:
    """LLM model configuration"""
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    model_name: str = os.getenv("MODEL_NAME", "gpt-4o")
    temperature: float = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
    max_tokens: int = int(os.getenv("MODEL_MAX_TOKENS", "2000"))

@dataclass
class VectorDBConfig:
    """Vector database configuration"""
    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    faiss_index_path: str = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    collection_name: str = os.getenv("COLLECTION_NAME", "startup_docs")
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", "10"))

@dataclass
class AnalysisWeights:
    """Scoring weights for different analysis areas"""
    growth_weight: float = float(os.getenv("GROWTH_WEIGHT", "0.20"))
    business_model_weight: float = float(os.getenv("BUSINESS_MODEL_WEIGHT", "0.15"))
    tech_security_weight: float = float(os.getenv("TECH_SECURITY_WEIGHT", "0.15"))
    financial_health_weight: float = float(os.getenv("FINANCIAL_HEALTH_WEIGHT", "0.15"))
    team_weight: float = float(os.getenv("TEAM_WEIGHT", "0.15"))
    regulatory_weight: float = float(os.getenv("REGULATORY_WEIGHT", "0.10"))
    partnership_weight: float = float(os.getenv("PARTNERSHIP_WEIGHT", "0.10"))

@dataclass
class ScoringConfig:
    """Scoring and grading configuration"""
    max_score: int = int(os.getenv("MAX_SCORE", "100"))
    grade_thresholds: Dict[str, int] = None
    unicorn_probability_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.grade_thresholds is None:
            self.grade_thresholds = {
                "S": int(os.getenv("GRADE_S_THRESHOLD", "90")),  # Exceptional
                "A": int(os.getenv("GRADE_A_THRESHOLD", "80")),  # Strong
                "B": int(os.getenv("GRADE_B_THRESHOLD", "70")),  # Good
                "C": int(os.getenv("GRADE_C_THRESHOLD", "60")),  # Average
                "D": 0    # Poor
            }

        if self.unicorn_probability_weights is None:
            self.unicorn_probability_weights = {
                "market_size": float(os.getenv("MARKET_SIZE_WEIGHT", "0.25")),
                "growth_rate": float(os.getenv("GROWTH_RATE_WEIGHT", "0.20")),
                "technology": float(os.getenv("TECHNOLOGY_WEIGHT", "0.15")),
                "team": float(os.getenv("TEAM_WEIGHT_PROBABILITY", "0.15")),
                "business_model": float(os.getenv("BUSINESS_MODEL_WEIGHT_PROBABILITY", "0.15")),
                "funding": float(os.getenv("FUNDING_WEIGHT", "0.10"))
            }

@dataclass
class DocumentPaths:
    """Document storage paths"""
    ir_documents: str = os.getenv("IR_DOCUMENTS_PATH", "./data/documents/ir_reports")
    market_reports: str = os.getenv("MARKET_REPORTS_PATH", "./data/documents/market_reports")
    company_profiles: str = os.getenv("COMPANY_PROFILES_PATH", "./data/documents/company_profiles")
    financial_statements: str = os.getenv("FINANCIAL_STATEMENTS_PATH", "./data/documents/financials")

# Global configuration instance
CONFIG = {
    "model": ModelConfig(),
    "vector_db": VectorDBConfig(),
    "analysis_weights": AnalysisWeights(),
    "scoring": ScoringConfig(),
    "document_paths": DocumentPaths(),
    "risk_categories": [
        "market_risk",
        "regulatory_risk",
        "competitive_risk",
        "financial_risk",
        "technology_risk",
        "team_risk"
    ],
    "analysis_categories": [
        "growth_analysis",
        "business_model_analysis",
        "tech_security_analysis",
        "financial_health_analysis",
        "team_evaluation",
        "regulatory_analysis",
        "partnership_analysis"
    ]
}

def get_config():
    """Get global configuration"""
    return CONFIG

def update_config(section: str, **kwargs):
    """Update configuration section"""
    if section in CONFIG:
        if hasattr(CONFIG[section], '__dict__'):
            for key, value in kwargs.items():
                if hasattr(CONFIG[section], key):
                    setattr(CONFIG[section], key, value)