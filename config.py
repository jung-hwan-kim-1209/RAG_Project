"""
Configuration settings for AI Startup Investment Evaluation Agent
"""
import os
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ModelConfig:
    """LLM model configuration"""
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    model_name: str = "gpt-4"
    temperature: float = 0.2
    max_tokens: int = 2000

@dataclass
class VectorDBConfig:
    """Vector database configuration"""
    chroma_persist_directory: str = "./data/chroma_db"
    faiss_index_path: str = "./data/faiss_index"
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL_NAME",
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    collection_name: str = "startup_docs"
    top_k_results: int = 10
    huggingfacehub_api_token: str = os.getenv(
        "HUGGINGFACEHUB_API_TOKEN",
        os.getenv("HF_TOKEN", "")
    )

@dataclass
class AnalysisWeights:
    """Scoring weights for different analysis areas"""
    growth_weight: float = 0.20
    business_model_weight: float = 0.15
    tech_security_weight: float = 0.15
    financial_health_weight: float = 0.15
    team_weight: float = 0.15
    regulatory_weight: float = 0.10
    partnership_weight: float = 0.10

@dataclass
class ScoringConfig:
    """Scoring and grading configuration"""
    max_score: int = 100
    grade_thresholds: Dict[str, int] = None
    unicorn_probability_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.grade_thresholds is None:
            self.grade_thresholds = {
                "S": 90,  # Exceptional
                "A": 80,  # Strong
                "B": 70,  # Good
                "C": 60,  # Average
                "D": 0    # Poor
            }

        if self.unicorn_probability_weights is None:
            self.unicorn_probability_weights = {
                "market_size": 0.25,
                "growth_rate": 0.20,
                "technology": 0.15,
                "team": 0.15,
                "business_model": 0.15,
                "funding": 0.10
            }

@dataclass
class DocumentPaths:
    """Document storage paths"""
    ir_documents: str = "./data/documents/ir_reports"
    market_reports: str = "./data/documents/market_reports"
    company_profiles: str = "./data/documents/company_profiles"
    financial_statements: str = "./data/documents/financials"

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
