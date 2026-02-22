"""
AEEE Configuration Manager - Centralized configuration with environment awareness
Handles all environment variables, Firebase initialization, and ML model configurations
"""
import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import firebase_admin
from firebase_admin import credentials, firestore, auth
from google.cloud.firestore_v1.base_client import BaseClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment types for the AEEE system"""
    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "prod"

@dataclass
class ModelConfig:
    """Configuration for ML models"""
    embedding_dim: int = 256
    hidden_layers: int = 3
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    batch_size: int = 32
    epochs: int = 100

@dataclass
class FirebaseConfig:
    """Firebase configuration"""
    project_id: str = ""
    credential_path: str = ""
    collection_prefix: str = "aeee_"
    
    def validate(self) -> bool:
        """Validate Firebase configuration"""
        if not self.project_id:
            logger.error("Firebase project_id is required")
            return False
        if not self.credential_path:
            logger.error("Firebase credential_path is required")
            return False
        return True

class AEEEConfig:
    """Main configuration manager for AEEE"""
    
    _instance: Optional['AEEEConfig'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize configuration with environment detection"""
        if not self._initialized:
            self.environment = self._detect_environment()
            self.model_config = ModelConfig()
            self.firebase_config = FirebaseConfig()
            self._load_environment_vars()
            self._validate_config()
            self._initialized = True
    
    def _detect_environment(self) -> Environment:
        """Detect current environment"""
        env_var = os.getenv("AEEE_ENV", "dev").lower()
        try:
            return Environment(env_var)
        except ValueError:
            logger.warning(f"Unknown environment {env_var}, defaulting to DEVELOPMENT")
            return Environment.DEVELOPMENT
    
    def _load_environment_vars(self):
        """Load configuration from environment variables"""
        # Firebase
        self.firebase_config.project_id = os.getenv("FIREBASE_PROJECT_ID", "aeee-system")
        self.firebase_config.credential_path = os.getenv("FIREBASE_CREDENTIAL_PATH", "./firebase_credentials.json")
        
        # Model configurations
        self.model_config.embedding_dim = int(os.getenv("EMBEDDING_DIM", "256"))
        self.model_config.hidden_layers = int(os.getenv("HIDDEN_LAYERS", "3"))
        self.model_config.learning_rate = float(os.getenv("LEARNING_RATE", "0.001"))
        
        logger.info(f"Loaded configuration for {self.environment.value} environment")
    
    def _validate_config(self) -> bool:
        """Validate all configurations"""
        is_valid = True
        
        if not self.firebase_config.validate():
            is_valid = False
        
        if self.model_config.learning_rate <= 0:
            logger.error("Learning rate must be positive")
            is_valid = False
        
        if not is_valid:
            raise ValueError("Configuration validation failed")
        
        return is_valid
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "environment": self.environment.value,
            "model_config": asdict(self.model_config),
            "firebase_config": asdict(self.firebase_config)
        }
    
    def get_firebase_client(self) -> BaseClient:
        """Initialize and return Firebase Firestore client"""
        try:
            # Check if Firebase app already initialized
            if not firebase_admin._apps:
                if not os.path.exists(self.firebase_config.credential_path):
                    raise FileNotFoundError(
                        f"Firebase credentials not found at {self.firebase_config.credential_path}"
                    )
                
                cred = credentials.Certificate(self.firebase_config.credential_path)
                firebase_admin.initialize_app(cred, {
                    'projectId': self.firebase_config.project_id
                })
                logger.info("Firebase initialized successfully")
            
            return firestore.client()
        
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {str(e)}")
            raise

# Global configuration instance
config = AEEEConfig()