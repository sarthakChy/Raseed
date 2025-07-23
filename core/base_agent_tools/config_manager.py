import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class AgentConfig:
    # Vertex AI / GCP Configuration
    project_id: str = "massive-incline-466204-t5"
    location: str = "us-central1"
    gcs_bucket_name: str = "raseed_receipts"
    google_wallet_issuer_id: str = "3388000000022970942"
    
    # Model Configuration - Single model for all components
    model_name: str = "gemini-2.0-flash-001"
    
    # Database Configuration
    instance_connection_name: str = "massive-incline-466204-t5:asia-south1:raseed-pg-instance"
    db_user: str = "postgres"
    db_password: str = "raseed"
    db_name: str = "postgres"
    
    @classmethod
    def from_env(cls) -> 'AgentConfig':
        """Load configuration from environment variables with fallback to defaults."""
        return cls(
            # GCP Configuration
            project_id=os.getenv('GCP_PROJECT_ID', cls.project_id),
            location=os.getenv('GCP_LOCATION', cls.location),
            gcs_bucket_name=os.getenv('GCS_BUCKET_NAME', cls.gcs_bucket_name),
            google_wallet_issuer_id=os.getenv('GOOGLE_WALLET_ISSUER_ID', cls.google_wallet_issuer_id),
            
            # Model Configuration
            model_name=os.getenv('MODEL_NAME', cls.model_name),
            
            # Database Configuration
            instance_connection_name=os.getenv('INSTANCE_CONNECTION_NAME', cls.instance_connection_name),
            db_user=os.getenv('DB_USER', cls.db_user),
            db_password=os.getenv('DB_PASS', cls.db_password),
            db_name=os.getenv('DB_NAME', cls.db_name),
        )
    
    @property
    def database_url(self) -> str:
        """Generate database connection URL."""
        return f"postgresql://{self.db_user}:{self.db_password}@/{self.db_name}?host=/cloudsql/{self.instance_connection_name}"
    
    @property
    def gcs_bucket_url(self) -> str:
        """Generate GCS bucket URL."""
        return f"gs://{self.gcs_bucket_name}"