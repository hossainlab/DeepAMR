# AMR Prediction System - Complete Implementation Plan
## Deep Learning-Based Antibiotic Resistance Prediction Platform

**Project Duration:** 20-24 weeks  
**Target Market:** Bangladesh Healthcare System  
**Technology Stack:** Deep Learning, FastAPI, React, PostgreSQL  

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Phase 1: Foundation & Setup (Week 1-3)](#phase-1-foundation--setup-week-1-3)
3. [Phase 2: Data Collection & Preprocessing (Week 4-6)](#phase-2-data-collection--preprocessing-week-4-6)
4. [Phase 3: ML Model Development (Week 7-10)](#phase-3-ml-model-development-week-7-10)
5. [Phase 4: Backend Development (Week 11-13)](#phase-4-backend-development-week-11-13)
6. [Phase 5: Frontend Development (Week 14-16)](#phase-5-frontend-development-week-14-16)
7. [Phase 6: Testing & Optimization (Week 17-18)](#phase-6-testing--optimization-week-17-18)
8. [Phase 7: Clinical Validation (Week 19-20)](#phase-7-clinical-validation-week-19-20)
9. [Phase 8: Deployment & Launch (Week 21-24)](#phase-8-deployment--launch-week-21-24)
10. [Post-Launch: Scaling & Commercialization](#post-launch-scaling--commercialization)

---

## Project Overview

### Vision
Build a production-ready AMR prediction platform that can predict antibiotic resistance from bacterial whole genome sequences in 4-6 hours vs traditional culture methods (48-72 hours).

### Key Metrics
- **Accuracy Target:** 85-92% (better than traditional methods)
- **Processing Time:** < 6 hours per sample
- **Target Antibiotics:** 8-12 commonly used antibiotics
- **Target Bacteria:** E. coli, Klebsiella, Staph aureus, Acinetobacter

### Revenue Model
- **Per-test pricing:** ৳500-800 per prediction
- **Monthly subscriptions:** ৳15,000 - ৳1,50,000 based on volume
- **Year 1 Target:** 10 hospital partnerships, ৳50,00,000 revenue

---

## Phase 1: Foundation & Setup (Week 1-3)

### Week 1: Development Environment & Project Structure

#### Day 1-2: System Setup

**Prerequisites:**
```bash
# System requirements
- OS: Ubuntu 20.04/22.04 or macOS
- RAM: 16GB minimum (32GB recommended)
- GPU: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- Storage: 500GB SSD
- Python: 3.9 or 3.10
```

**Initial Setup:**
```bash
# 1. Install system dependencies
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3-pip
sudo apt install -y git curl wget build-essential
sudo apt install -y postgresql postgresql-contrib redis-server

# 2. Install CUDA (if using GPU)
# Follow: https://developer.nvidia.com/cuda-downloads

# 3. Create project directory
mkdir ~/amr-prediction
cd ~/amr-prediction

# 4. Initialize git repository
git init
git remote add origin <your-github-repo>

# 5. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 6. Upgrade pip
pip install --upgrade pip setuptools wheel
```

**Create Project Structure:**
```bash
# Create comprehensive directory structure
mkdir -p {data,models,notebooks,src,tests,frontend,deployment,docs}
mkdir -p data/{raw,processed,external,uploads}
mkdir -p data/raw/{ncbi,patric,card,resfinder}
mkdir -p models/{pretrained,checkpoints,final,evaluation}
mkdir -p src/{ml,api,preprocessing,utils}
mkdir -p src/ml/{feature_extraction,training,inference}
mkdir -p src/api/{routes,models,services,middleware}
mkdir -p deployment/{docker,kubernetes,scripts}
mkdir -p frontend/{src,public,components}
mkdir -p tests/{unit,integration,e2e}
mkdir -p docs/{api,user_guide,technical}
mkdir -p logs

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Data
data/raw/*
data/processed/*
*.fasta
*.fastq
*.h5
*.pkl

# Models
models/pretrained/*
models/checkpoints/*
*.pth
*.pt

# Logs
logs/*
*.log

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Environment
.env
.env.local

# Uploads
uploads/*
EOF

# Initialize README
cat > README.md << 'EOF'
# AMR Prediction Platform

Deep learning-based antibiotic resistance prediction from whole genome sequences.

## Features
- Predict resistance to 8-12 antibiotics
- 85-92% accuracy using ESM-2 protein language models
- Results in 4-6 hours vs 48-72 hours traditional methods
- REST API for integration
- Web-based user interface

## Tech Stack
- **ML:** PyTorch, HuggingFace Transformers, BioPython
- **Backend:** FastAPI, PostgreSQL, Redis, Celery
- **Frontend:** React, Material-UI
- **Deployment:** Docker, Kubernetes

## Getting Started
See [Installation Guide](docs/installation.md)

## License
Proprietary - All Rights Reserved
EOF
```

#### Day 3-4: Install Dependencies

**Create requirements.txt:**
```bash
cat > requirements.txt << 'EOF'
# Core ML/DL
torch==2.1.0
transformers==4.35.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0

# Bioinformatics
biopython==1.81
pysam==0.21.0

# Deep Learning utilities
tensorboard==2.14.0
wandb==0.15.12

# Backend
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
pydantic-settings==2.0.3
python-multipart==0.0.6

# Database
sqlalchemy==2.0.23
psycopg2-binary==2.9.9
alembic==1.12.1

# Task Queue
celery==5.3.4
redis==5.0.1

# Authentication & Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0

# File handling
aiofiles==23.2.1
openpyxl==3.1.2

# Monitoring & Logging
sentry-sdk==1.38.0
python-json-logger==2.0.7

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-cov==4.1.0
httpx==0.25.1

# Development
black==23.11.0
flake8==6.1.0
mypy==1.7.0
pre-commit==3.5.0

# Jupyter
jupyter==1.0.0
ipython==8.17.2
matplotlib==3.8.0
seaborn==0.13.0
plotly==5.18.0

# Utilities
tqdm==4.66.1
requests==2.31.0
pyyaml==6.0.1


# Install all dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support (if GPU available)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only:
# pip install torch torchvision torchaudio
```

**Verify Installation:**
```python
# Create verification script
cat > verify_setup.py << 'EOF'
import sys
import torch
import transformers
from Bio import SeqIO
import pandas as pd
import numpy as np

print("Python version:", sys.version)
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))
print("Transformers version:", transformers.__version__)
print("BioPython imported successfully")
print("Pandas version:", pd.__version__)
print("\n✅ All dependencies installed successfully!")
EOF

python verify_setup.py
```

#### Day 5-7: Download Pre-trained Models

**Model Download Script:**
```python
# src/utils/download_models.py
"""
Download pre-trained models from HuggingFace
"""
from transformers import AutoTokenizer, AutoModel
import torch
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelDownloader:
    def __init__(self, save_dir="models/pretrained"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def download_esm2(self, model_size="35M"):
        """
        Download ESM-2 protein language model
        
        Available sizes:
        - 8M: esm2_t6_8M_UR50D (6 layers, 8M params)
        - 35M: esm2_t12_35M_UR50D (12 layers, 35M params) - RECOMMENDED FOR TESTING
        - 150M: esm2_t30_150M_UR50D (30 layers, 150M params)
        - 650M: esm2_t33_650M_UR50D (33 layers, 650M params) - RECOMMENDED FOR PRODUCTION
        - 3B: esm2_t36_3B_UR50D (36 layers, 3B params)
        - 15B: esm2_t48_15B_UR50D (48 layers, 15B params)
        """
        model_mapping = {
            "8M": "facebook/esm2_t6_8M_UR50D",
            "35M": "facebook/esm2_t12_35M_UR50D",
            "150M": "facebook/esm2_t30_150M_UR50D",
            "650M": "facebook/esm2_t33_650M_UR50D",
            "3B": "facebook/esm2_t36_3B_UR50D",
            "15B": "facebook/esm2_t48_15B_UR50D"
        }
        
        if model_size not in model_mapping:
            raise ValueError(f"Invalid model size. Choose from: {list(model_mapping.keys())}")
        
        model_name = model_mapping[model_size]
        save_path = self.save_dir / f"esm2_{model_size}"
        
        logger.info(f"Downloading ESM-2 {model_size} model: {model_name}")
        logger.info("This may take several minutes depending on your internet speed...")
        
        try:
            # Download tokenizer
            logger.info("Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokenizer.save_pretrained(save_path)
            logger.info(f"✓ Tokenizer saved to {save_path}")
            
            # Download model
            logger.info("Downloading model weights...")
            model = AutoModel.from_pretrained(model_name)
            model.save_pretrained(save_path)
            logger.info(f"✓ Model saved to {save_path}")
            
            # Test model
            logger.info("Testing model...")
            test_sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
            
            inputs = tokenizer(test_sequence, return_tensors="pt")
            with torch.no_grad():
                outputs = model(**inputs)
            
            logger.info(f"✓ Model test successful! Output shape: {outputs.last_hidden_state.shape}")
            logger.info(f"✅ ESM-2 {model_size} downloaded successfully to {save_path}")
            
            return save_path
            
        except Exception as e:
            logger.error(f"❌ Error downloading model: {e}")
            raise
    
    def download_dnabert2(self):
        """Download DNABERT-2 for DNA sequence analysis"""
        model_name = "zhihan1996/DNABERT-2-117M"
        save_path = self.save_dir / "dnabert2"
        
        logger.info(f"Downloading DNABERT-2: {model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            tokenizer.save_pretrained(save_path)
            
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
            model.save_pretrained(save_path)
            
            logger.info(f"✅ DNABERT-2 downloaded successfully to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"❌ Error downloading DNABERT-2: {e}")
            raise
    
    def download_all(self, esm2_size="35M", include_dnabert=False):
        """Download all required models"""
        logger.info("="*60)
        logger.info("Starting model downloads...")
        logger.info("="*60)
        
        # Download ESM-2
        esm2_path = self.download_esm2(esm2_size)
        
        # Optionally download DNABERT-2
        dnabert_path = None
        if include_dnabert:
            dnabert_path = self.download_dnabert2()
        
        logger.info("="*60)
        logger.info("All downloads complete!")
        logger.info(f"ESM-2 {esm2_size}: {esm2_path}")
        if dnabert_path:
            logger.info(f"DNABERT-2: {dnabert_path}")
        logger.info("="*60)
        
        return {
            "esm2": esm2_path,
            "dnabert2": dnabert_path
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download pre-trained models")
    parser.add_argument("--esm2-size", default="35M", 
                       choices=["8M", "35M", "150M", "650M", "3B", "15B"],
                       help="ESM-2 model size (default: 35M for testing, use 650M for production)")
    parser.add_argument("--include-dnabert", action="store_true",
                       help="Also download DNABERT-2 model")
    
    args = parser.parse_args()
    
    downloader = ModelDownloader()
    downloader.download_all(esm2_size=args.esm2_size, include_dnabert=args.include_dnabert)
```

**Run the download:**
```bash
# Download small model for testing (recommended for Week 1)
python src/utils/download_models.py --esm2-size 35M

# For production (do this later when you have more compute):
# python src/utils/download_models.py --esm2-size 650M --include-dnabert
```

### Week 2: Database Setup & Configuration

#### Day 8-9: PostgreSQL Database Setup

**Database Schema:**
```sql
-- deployment/database/schema.sql
-- AMR Prediction Database Schema

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Organizations (hospitals, labs, clinics)
CREATE TABLE organizations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL CHECK (type IN ('hospital', 'diagnostic_lab', 'research_center', 'clinic')),
    address TEXT,
    city VARCHAR(100),
    country VARCHAR(100) DEFAULT 'Bangladesh',
    phone VARCHAR(20),
    email VARCHAR(255),
    license_number VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    subscription_tier VARCHAR(50) CHECK (subscription_tier IN ('free', 'basic', 'professional', 'enterprise')),
    subscription_start_date DATE,
    subscription_end_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('admin', 'doctor', 'lab_technician', 'researcher', 'viewer')),
    phone VARCHAR(20),
    license_number VARCHAR(100), -- Medical license for doctors
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    last_login TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Patients
CREATE TABLE patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    patient_id VARCHAR(100) NOT NULL, -- Hospital's internal patient ID
    name VARCHAR(255),
    age INTEGER,
    gender VARCHAR(20) CHECK (gender IN ('male', 'female', 'other', 'unknown')),
    phone VARCHAR(20),
    admission_date DATE,
    department VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(organization_id, patient_id)
);

-- Samples
CREATE TABLE samples (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    patient_id UUID REFERENCES patients(id) ON DELETE SET NULL,
    sample_id VARCHAR(100) NOT NULL, -- Lab's internal sample ID
    sample_type VARCHAR(100) CHECK (sample_type IN ('blood', 'urine', 'sputum', 'wound', 'csf', 'other')),
    collection_date TIMESTAMP,
    collection_site VARCHAR(100),
    clinical_info TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(organization_id, sample_id)
);

-- Bacterial Isolates
CREATE TABLE isolates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sample_id UUID REFERENCES samples(id) ON DELETE CASCADE,
    isolate_number VARCHAR(100),
    species VARCHAR(255), -- E. coli, Klebsiella pneumoniae, etc.
    gram_stain VARCHAR(20) CHECK (gram_stain IN ('positive', 'negative', 'unknown')),
    culture_result TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Genome Sequencing Runs
CREATE TABLE sequencing_runs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    isolate_id UUID REFERENCES isolates(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    run_id VARCHAR(100) NOT NULL,
    sequencing_platform VARCHAR(100), -- Illumina, Nanopore, etc.
    sequencing_date DATE,
    coverage FLOAT, -- Sequencing coverage depth
    quality_score FLOAT,
    genome_file_path TEXT, -- Path to FASTA/FASTQ file
    genome_file_size BIGINT, -- File size in bytes
    genome_checksum VARCHAR(64), -- MD5 or SHA256 hash
    status VARCHAR(50) DEFAULT 'uploaded' CHECK (status IN ('uploaded', 'processing', 'completed', 'failed')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- AMR Predictions (Main table)
CREATE TABLE predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    sequencing_run_id UUID REFERENCES sequencing_runs(id) ON DELETE CASCADE,
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Model information
    model_version VARCHAR(50) NOT NULL, -- v1.0, v1.1, etc.
    model_type VARCHAR(50) DEFAULT 'esm2_650M',
    
    -- Processing status
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    progress INTEGER DEFAULT 0, -- 0-100%
    
    -- Timing
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    processing_time_seconds INTEGER,
    
    -- Results summary
    total_antibiotics_tested INTEGER,
    resistant_count INTEGER,
    susceptible_count INTEGER,
    intermediate_count INTEGER,
    
    -- Quality metrics
    confidence_score FLOAT, -- Overall prediction confidence
    
    -- Error handling
    error_message TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Individual Antibiotic Resistance Results
CREATE TABLE resistance_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id UUID REFERENCES predictions(id) ON DELETE CASCADE,
    
    -- Antibiotic information
    antibiotic_name VARCHAR(100) NOT NULL,
    antibiotic_class VARCHAR(100), -- Beta-lactam, Fluoroquinolone, etc.
    
    -- Prediction
    prediction VARCHAR(20) NOT NULL CHECK (prediction IN ('Resistant', 'Susceptible', 'Intermediate')),
    confidence_score FLOAT, -- 0-1
    probability_resistant FLOAT, -- Raw model output
    
    -- Detected genetic markers
    detected_genes TEXT[], -- Array of detected AMR genes
    detected_mutations TEXT[], -- Specific mutations
    resistance_mechanism TEXT, -- e.g., "Beta-lactamase production"
    
    -- Clinical interpretation
    clinical_interpretation TEXT,
    recommendation TEXT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- AMR Genes Detected
CREATE TABLE detected_genes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    prediction_id UUID REFERENCES predictions(id) ON DELETE CASCADE,
    
    gene_name VARCHAR(100) NOT NULL,
    gene_family VARCHAR(100),
    accession VARCHAR(100),
    
    -- Detection details
    identity_percentage FLOAT, -- Sequence identity %
    coverage_percentage FLOAT, -- Gene coverage %
    
    -- Associated resistances
    conferred_resistance TEXT[], -- Which antibiotics this gene confers resistance to
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Audit Log
CREATE TABLE audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    organization_id UUID REFERENCES organizations(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL, -- 'create_prediction', 'view_result', 'export_report', etc.
    resource_type VARCHAR(50), -- 'prediction', 'user', 'organization', etc.
    resource_id UUID,
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API Keys (for programmatic access)
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    key_prefix VARCHAR(20) NOT NULL, -- First few chars for identification
    name VARCHAR(255), -- User-friendly name
    
    permissions JSONB, -- {"read": true, "write": true, "delete": false}
    
    is_active BOOLEAN DEFAULT true,
    last_used_at TIMESTAMP,
    expires_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Subscription Usage Tracking
CREATE TABLE usage_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    
    metric_date DATE NOT NULL,
    predictions_count INTEGER DEFAULT 0,
    storage_used_bytes BIGINT DEFAULT 0,
    api_calls_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(organization_id, metric_date)
);

-- Indexes for performance
CREATE INDEX idx_users_organization ON users(organization_id);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_patients_org_patient_id ON patients(organization_id, patient_id);
CREATE INDEX idx_samples_org_sample_id ON samples(organization_id, sample_id);
CREATE INDEX idx_samples_patient ON samples(patient_id);
CREATE INDEX idx_isolates_sample ON isolates(sample_id);
CREATE INDEX idx_sequencing_isolate ON sequencing_runs(isolate_id);
CREATE INDEX idx_sequencing_status ON sequencing_runs(status);
CREATE INDEX idx_predictions_org ON predictions(organization_id);
CREATE INDEX idx_predictions_user ON predictions(user_id);
CREATE INDEX idx_predictions_status ON predictions(status);
CREATE INDEX idx_predictions_created ON predictions(created_at DESC);
CREATE INDEX idx_resistance_prediction ON resistance_results(prediction_id);
CREATE INDEX idx_resistance_antibiotic ON resistance_results(antibiotic_name);
CREATE INDEX idx_genes_prediction ON detected_genes(prediction_id);
CREATE INDEX idx_audit_user ON audit_log(user_id);
CREATE INDEX idx_audit_org ON audit_log(organization_id);
CREATE INDEX idx_audit_created ON audit_log(created_at DESC);

-- Triggers for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_organizations_updated_at BEFORE UPDATE ON organizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sequencing_runs_updated_at BEFORE UPDATE ON sequencing_runs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_predictions_updated_at BEFORE UPDATE ON predictions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Sample seed data for development
INSERT INTO organizations (name, type, city, subscription_tier, subscription_start_date, subscription_end_date)
VALUES 
    ('Dhaka Medical College Hospital', 'hospital', 'Dhaka', 'professional', CURRENT_DATE, CURRENT_DATE + INTERVAL '1 year'),
    ('Popular Diagnostic Centre', 'diagnostic_lab', 'Dhaka', 'enterprise', CURRENT_DATE, CURRENT_DATE + INTERVAL '1 year'),
    ('Square Hospital Ltd', 'hospital', 'Dhaka', 'professional', CURRENT_DATE, CURRENT_DATE + INTERVAL '1 year');
```

**Database Setup Script:**
```bash
# deployment/database/setup.sh
#!/bin/bash

# Database configuration
DB_NAME="amr_prediction"
DB_USER="amr_user"
DB_PASSWORD="secure_password_here"  # Change this!

echo "Setting up PostgreSQL database..."

# Create database and user
sudo -u postgres psql << EOF
CREATE DATABASE $DB_NAME;
CREATE USER $DB_USER WITH ENCRYPTED PASSWORD '$DB_PASSWORD';
GRANT ALL PRIVILEGES ON DATABASE $DB_NAME TO $DB_USER;
\q
EOF

# Run schema
sudo -u postgres psql -d $DB_NAME -f schema.sql

echo "✅ Database setup complete!"
echo "Database: $DB_NAME"
echo "User: $DB_USER"
echo ""
echo "Add to .env file:"
echo "DATABASE_URL=postgresql://$DB_USER:$DB_PASSWORD@localhost:5432/$DB_NAME"
```

**Run setup:**
```bash
chmod +x deployment/database/setup.sh
./deployment/database/setup.sh
```

#### Day 10-11: Redis & Celery Setup

**Redis Installation:**
```bash
# Install Redis
sudo apt install redis-server

# Start Redis
sudo systemctl start redis
sudo systemctl enable redis

# Test Redis
redis-cli ping  # Should return PONG
```

**Celery Configuration:**
```python
# src/api/celery_app.py
"""
Celery configuration for background tasks
"""
from celery import Celery
from kombu import Exchange, Queue
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize Celery
celery_app = Celery(
    "amr_prediction",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0")
)

# Celery configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='Asia/Dhaka',
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    task_soft_time_limit=3300,  # 55 minutes soft limit
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
)

# Define queues
celery_app.conf.task_queues = (
    Queue('default', Exchange('default'), routing_key='default'),
    Queue('prediction', Exchange('prediction'), routing_key='prediction.#'),
    Queue('preprocessing', Exchange('preprocessing'), routing_key='preprocessing.#'),
)

# Task routes
celery_app.conf.task_routes = {
    'src.api.tasks.predict_resistance': {'queue': 'prediction'},
    'src.api.tasks.preprocess_genome': {'queue': 'preprocessing'},
}
```

**Celery Tasks:**
```python
# src/api/tasks.py
"""
Background tasks for AMR prediction
"""
from celery import Task
from src.api.celery_app import celery_app
from src.ml.inference.predictor import AMRPredictor
import logging

logger = logging.getLogger(__name__)

class PredictionTask(Task):
    """Base class for prediction tasks with progress tracking"""
    
    def __init__(self):
        self._predictor = None
    
    @property
    def predictor(self):
        if self._predictor is None:
            self._predictor = AMRPredictor()
        return self._predictor

@celery_app.task(bind=True, base=PredictionTask, name='predict_resistance')
def predict_resistance(self, prediction_id: str, genome_path: str, antibiotics: list):
    """
    Run AMR prediction for a genome
    
    Args:
        prediction_id: UUID of prediction record
        genome_path: Path to genome FASTA file
        antibiotics: List of antibiotics to test
    """
    try:
        # Update status
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Loading genome...'}
        )
        
        # Run prediction
        results = self.predictor.predict(
            genome_path=genome_path,
            antibiotics=antibiotics,
            progress_callback=lambda p: self.update_state(
                state='PROGRESS',
                meta={'current': p, 'total': 100, 'status': f'Processing... {p}%'}
            )
        )
        
        return {
            'status': 'completed',
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        self.update_state(
            state='FAILURE',
            meta={'error': str(e)}
        )
        raise
```

#### Day 12-14: Environment Configuration

**Create .env file:**
```bash
# .env
# Application
APP_NAME="AMR Prediction Platform"
APP_VERSION="1.0.0"
ENVIRONMENT="development"  # development, staging, production
DEBUG=true
SECRET_KEY="your-secret-key-here-change-this-in-production"

# Database
DATABASE_URL="postgresql://amr_user:secure_password_here@localhost:5432/amr_prediction"
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=0

# Redis
REDIS_URL="redis://localhost:6379/0"

# Celery
CELERY_BROKER_URL="redis://localhost:6379/0"
CELERY_RESULT_BACKEND="redis://localhost:6379/0"

# File Storage
UPLOAD_DIR="/home/youruser/amr-prediction/data/uploads"
MAX_UPLOAD_SIZE=5368709120  # 5GB in bytes
ALLOWED_EXTENSIONS=".fasta,.fastq,.fa,.fq,.fna"

# Model Paths
ESM2_MODEL_PATH="models/pretrained/esm2_35M"
DNABERT2_MODEL_PATH="models/pretrained/dnabert2"
CARD_DB_PATH="data/external/card"

# Authentication
JWT_SECRET_KEY="your-jwt-secret-key-change-in-production"
JWT_ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# API Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# Email (for notifications)
SMTP_HOST="smtp.gmail.com"
SMTP_PORT=587
SMTP_USER="your-email@gmail.com"
SMTP_PASSWORD="your-app-password"
EMAIL_FROM="noreply@amrprediction.com"

# Monitoring
SENTRY_DSN=""  # Add Sentry DSN for error tracking
LOG_LEVEL="INFO"

# Deployment
ALLOWED_HOSTS="localhost,127.0.0.1,your-domain.com"
CORS_ORIGINS="http://localhost:3000,http://localhost:8000"
```

**Configuration Module:**
```python
# src/api/config.py
"""
Application configuration
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List

class Settings(BaseSettings):
    # App
    APP_NAME: str = "AMR Prediction Platform"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str
    
    # Database
    DATABASE_URL: str
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 0
    
    # Redis
    REDIS_URL: str
    
    # File Storage
    UPLOAD_DIR: str
    MAX_UPLOAD_SIZE: int = 5368709120  # 5GB
    ALLOWED_EXTENSIONS: str = ".fasta,.fastq,.fa,.fq,.fna"
    
    # Models
    ESM2_MODEL_PATH: str
    DNABERT2_MODEL_PATH: str = ""
    CARD_DB_PATH: str
    
    # JWT
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # API
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # CORS
    CORS_ORIGINS: str = "http://localhost:3000"
    
    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(",")]
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
```

### Week 3: Testing Environment & Initial Prototypes

#### Day 15-16: Create Test Dataset

**Download Sample Genomes:**
```python
# notebooks/01_download_test_data.ipynb
"""
Download sample bacterial genomes for testing
"""
import requests
from Bio import Entrez, SeqIO
from pathlib import Path
import pandas as pd

# Set your email (required by NCBI)
Entrez.email = "your.email@example.com"

def download_ncbi_genome(accession, output_dir="data/raw/test_samples"):
    """Download genome from NCBI by accession number"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"Downloading {accession}...")
        handle = Entrez.efetch(db="nucleotide", id=accession, rettype="fasta", retmode="text")
        record = SeqIO.read(handle, "fasta")
        handle.close()
        
        # Save to file
        output_file = output_dir / f"{accession}.fasta"
        SeqIO.write(record, output_file, "fasta")
        print(f"✓ Saved to {output_file}")
        
        return str(output_file)
    except Exception as e:
        print(f"✗ Error downloading {accession}: {e}")
        return None

# Sample E. coli genomes with known AMR profiles
test_genomes = [
    {
        "accession": "CP000247.1",  # E. coli O157:H7 str. Sakai
        "species": "Escherichia coli",
        "strain": "O157:H7 Sakai",
        "expected_resistance": ["Beta-lactam"]
    },
    {
        "accession": "CP001671.1",  # K. pneumoniae
        "species": "Klebsiella pneumoniae",
        "strain": "342",
        "expected_resistance": ["Beta-lactam", "Fluoroquinolone"]
    }
]

# Download all test genomes
for genome in test_genomes:
    genome['file_path'] = download_ncbi_genome(genome['accession'])

# Save metadata
df = pd.DataFrame(test_genomes)
df.to_csv("data/raw/test_samples/metadata.csv", index=False)
print("\n✅ Test dataset created!")
print(df)
```

#### Day 17-18: Build Initial Prototype

**Simple Prediction Pipeline:**
```python
# notebooks/02_prototype_pipeline.ipynb
"""
Test the complete prediction pipeline
"""
from src.ml.feature_extraction.protein_features import ProteinFeatureExtractor
from Bio import SeqIO
import numpy as np

# Initialize feature extractor
extractor = ProteinFeatureExtractor(
    model_path="models/pretrained/esm2_35M"
)

# Load test genome
genome_path = "data/raw/test_samples/CP000247.1.fasta"
print(f"Loading genome: {genome_path}")

# Extract proteins
genome_seq = ""
for record in SeqIO.parse(genome_path, "fasta"):
    genome_seq += str(record.seq)
    
print(f"Genome length: {len(genome_seq):,} bp")

# Extract proteins
proteins = extractor.extract_proteins_from_genome(genome_seq)
print(f"Found {len(proteins)} proteins")

# Get embeddings for first protein
if len(proteins) > 0:
    test_protein = proteins[0]
    print(f"\nTest protein length: {len(test_protein)} aa")
    print(f"Sequence preview: {test_protein[:50]}...")
    
    # Get embedding
    embedding = extractor.get_protein_embedding(test_protein)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding mean: {embedding.mean():.4f}")
    print(f"Embedding std: {embedding.std():.4f}")
    
    print("\n✅ Feature extraction working!")
```

#### Day 19-21: Documentation & Planning

**Project Documentation:**
```markdown
# docs/architecture.md

# System Architecture

## Overview
The AMR Prediction Platform consists of several key components:

1. **ML Pipeline**: Feature extraction and prediction
2. **API Server**: FastAPI backend
3. **Database**: PostgreSQL for data storage
4. **Task Queue**: Celery + Redis for async processing
5. **Frontend**: React web application

## Data Flow

1. User uploads genome FASTA file via web interface
2. File is validated and stored
3. Background task is queued for prediction
4. ML pipeline:
   - Extracts proteins from genome
   - Generates embeddings using ESM-2
   - Detects AMR genes
   - Runs classifier for each antibiotic
5. Results are stored in database
6. User is notified and can view results

## Technology Decisions

### Why ESM-2?
- State-of-the-art protein language model
- Pre-trained on 250M protein sequences
- No need for MSA (multiple sequence alignment)
- Captures evolutionary information
- Better than CNN/RNN approaches

### Why FastAPI?
- Modern, fast Python web framework
- Automatic API documentation
- Native async support
- Type hints and validation
- Easy to deploy

### Why PostgreSQL?
- ACID compliance
- JSON support for flexible data
- Strong ecosystem
- Good for structured medical data

## Deployment Architecture

### Development
- Local machine
- SQLite or PostgreSQL
- Redis for local testing

### Production
- AWS EC2 or DigitalOcean Droplet
- RDS PostgreSQL
- ElastiCache Redis
- S3 for file storage
- Load balancer for scaling
```

---

## Phase 2: Data Collection & Preprocessing (Week 4-6)

### Week 4: Data Collection

#### Day 22-24: Download NCBI Pathogen Detection Data

**NCBI Data Download Script:**
```python
# src/preprocessing/ncbi_downloader.py
"""
Download bacterial genomes and AMR data from NCBI Pathogen Detection
"""
import pandas as pd
import requests
from Bio import Entrez
from pathlib import Path
import time
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NCBIPathogenDownloader:
    """Download data from NCBI Pathogen Detection database"""
    
    def __init__(self, email, output_dir="data/raw/ncbi"):
        Entrez.email = email
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.genomes_dir = self.output_dir / "genomes"
        self.genomes_dir.mkdir(exist_ok=True)
    
    def download_metadata(self, organisms=None, location="Bangladesh"):
        """
        Download metadata from NCBI Pathogen Detection
        
        Args:
            organisms: List of organisms (e.g., ['Escherichia coli', 'Klebsiella pneumoniae'])
            location: Geographic location filter
        """
        if organisms is None:
            organisms = [
                "Escherichia coli",
                "Klebsiella pneumoniae",
                "Staphylococcus aureus",
                "Acinetobacter baumannii",
                "Pseudomonas aeruginosa"
            ]
        
        logger.info(f"Downloading metadata for {len(organisms)} organisms...")
        
        all_metadata = []
        
        for organism in organisms:
            logger.info(f"Searching {organism}...")
            
            # Search query
            query = f"{organism}[Organism] AND {location}[Country]" if location else f"{organism}[Organism]"
            
            try:
                # Search for IDs
                handle = Entrez.esearch(db="biosample", term=query, retmax=1000)
                record = Entrez.read(handle)
                handle.close()
                
                id_list = record["IdList"]
                logger.info(f"Found {len(id_list)} samples for {organism}")
                
                if len(id_list) == 0:
                    continue
                
                # Fetch detailed info in batches
                batch_size = 100
                for i in range(0, len(id_list), batch_size):
                    batch_ids = id_list[i:i+batch_size]
                    
                    handle = Entrez.efetch(db="biosample", id=batch_ids, rettype="xml")
                    records = Entrez.read(handle)
                    handle.close()
                    
                    for record in records:
                        metadata = self._parse_biosample_record(record, organism)
                        if metadata:
                            all_metadata.append(metadata)
                    
                    time.sleep(0.5)  # Be nice to NCBI servers
                
            except Exception as e:
                logger.error(f"Error downloading {organism}: {e}")
                continue
        
        # Save metadata
        df = pd.DataFrame(all_metadata)
        output_file = self.output_dir / "metadata.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"✅ Downloaded {len(df)} samples")
        logger.info(f"Saved to {output_file}")
        
        return df
    
    def _parse_biosample_record(self, record, organism):
        """Parse BioSample record to extract relevant info"""
        try:
            biosample_id = record['Ids'][0]['content']
            
            # Extract attributes
            attributes = {}
            for attr in record.get('Attributes', []):
                attr_name = attr['@attribute_name']
                attr_value = attr.get('#text', '')
                attributes[attr_name] = attr_value
            
            # Check for AMR data
            amr_genotypes = attributes.get('AMR_genotypes', '')
            
            metadata = {
                'biosample_id': biosample_id,
                'organism': organism,
                'isolation_source': attributes.get('isolation_source', ''),
                'collection_date': attributes.get('collection_date', ''),
                'geographic_location': attributes.get('geo_loc_name', ''),
                'strain': attributes.get('strain', ''),
                'serovar': attributes.get('serovar', ''),
                'amr_genotypes': amr_genotypes,
                'has_amr_data': bool(amr_genotypes)
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error parsing record: {e}")
            return None
    
    def download_genomes(self, metadata_df, max_genomes=100):
        """
        Download genome assemblies
        
        Args:
            metadata_df: DataFrame with BioSample IDs
            max_genomes: Maximum number of genomes to download
        """
        # Filter for samples with AMR data
        amr_samples = metadata_df[metadata_df['has_amr_data'] == True].head(max_genomes)
        
        logger.info(f"Downloading {len(amr_samples)} genomes with AMR data...")
        
        downloaded = []
        
        for idx, row in tqdm(amr_samples.iterrows(), total=len(amr_samples)):
            biosample_id = row['biosample_id']
            
            try:
                # Find associated assembly
                handle = Entrez.esearch(db="assembly", term=f"{biosample_id}[BioSample]")
                record = Entrez.read(handle)
                handle.close()
                
                if len(record['IdList']) == 0:
                    continue
                
                assembly_id = record['IdList'][0]
                
                # Get assembly summary
                handle = Entrez.esummary(db="assembly", id=assembly_id)
                summary = Entrez.read(handle)
                handle.close()
                
                # Get FTP path
                ftp_path = summary['DocumentSummarySet']['DocumentSummary'][0]['FtpPath_RefSeq']
                
                if not ftp_path:
                    ftp_path = summary['DocumentSummarySet']['DocumentSummary'][0]['FtpPath_GenBank']
                
                if not ftp_path:
                    continue
                
                # Download genome FASTA
                assembly_name = ftp_path.split('/')[-1]
                fasta_url = f"{ftp_path}/{assembly_name}_genomic.fna.gz"
                
                output_file = self.genomes_dir / f"{biosample_id}.fna.gz"
                
                # Download
                response = requests.get(fasta_url.replace('ftp://', 'https://'))
                if response.status_code == 200:
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    
                    downloaded.append({
                        'biosample_id': biosample_id,
                        'assembly_id': assembly_id,
                        'file_path': str(output_file)
                    })
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error downloading {biosample_id}: {e}")
                continue
        
        # Save download log
        df_downloaded = pd.DataFrame(downloaded)
        df_downloaded.to_csv(self.output_dir / "downloaded_genomes.csv", index=False)
        
        logger.info(f"✅ Downloaded {len(downloaded)} genomes")
        
        return df_downloaded

# Usage
if __name__ == "__main__":
    downloader = NCBIPathogenDownloader(
        email="your.email@example.com",
        output_dir="data/raw/ncbi"
    )
    
    # Download metadata
    metadata = downloader.download_metadata(
        organisms=["Escherichia coli", "Klebsiella pneumoniae"],
        location="Bangladesh"  # Try Bangladesh first, then expand if needed
    )
    
    # Download genomes
    genomes = downloader.download_genomes(metadata, max_genomes=100)
```

#### Day 25-28: Download CARD and Additional Databases

**CARD Database Download:**
```bash
# Download CARD database
cd data/external
mkdir -p card
cd card

# Download latest CARD data
wget https://card.mcmaster.ca/latest/data
tar -xvf data

# Download protein sequences
wget https://card.mcmaster.ca/latest/protein_fasta_protein_homolog_model.fasta

# Download nucleotide sequences
wget https://card.mcmaster.ca/latest/nucleotide_fasta_protein_homolog_model.fasta

# Download AMR gene annotations
wget https://card.mcmaster.ca/latest/aro_index.tsv

echo "✅ CARD database downloaded"
```

**ResFinder Download:**
```bash
# Download ResFinder database
cd data/external
git clone https://bitbucket.org/genomicepidemiology/resfinder_db.git resfinder

echo "✅ ResFinder database downloaded"
```

### Week 5-6: Data Preprocessing

#### Day 29-35: Build Preprocessing Pipeline

**Complete Preprocessing Pipeline:**
```python
# src/preprocessing/data_preprocessor.py
"""
Complete data preprocessing pipeline
"""
import pandas as pd
import gzip
from Bio import SeqIO
from pathlib import Path
import json
from typing import List, Dict
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AMRDataPreprocessor:
    """Preprocess raw genome data for training"""
    
    def __init__(self, 
                 ncbi_dir="data/raw/ncbi",
                 card_dir="data/external/card",
                 output_dir="data/processed"):
        self.ncbi_dir = Path(ncbi_dir)
        self.card_dir = Path(card_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CARD annotations
        self.amr_annotations = self._load_card_annotations()
    
    def _load_card_annotations(self):
        """Load AMR gene annotations from CARD"""
        aro_file = self.card_dir / "aro_index.tsv"
        
        if not aro_file.exists():
            logger.warning("CARD annotations not found")
            return {}
        
        df = pd.read_csv(aro_file, sep='\t')
        
        annotations = {}
        for idx, row in df.iterrows():
            gene_family = row.get('AMR Gene Family', '')
            drug_class = row.get('Drug Class', '')
            resistance_mechanism = row.get('Resistance Mechanism', '')
            
            annotations[row['ARO Accession']] = {
                'gene_family': gene_family,
                'drug_class': drug_class,
                'mechanism': resistance_mechanism
            }
        
        logger.info(f"Loaded {len(annotations)} AMR gene annotations")
        return annotations
    
    def parse_amr_phenotype(self, amr_string: str) -> Dict:
        """
        Parse AMR phenotype string from NCBI
        
        Format: "AMINOGLYCOSIDE=RESISTANT;BETA-LACTAM=SUSCEPTIBLE"
        """
        if pd.isna(amr_string) or not amr_string:
            return {}
        
        phenotypes = {}
        pairs = amr_string.split(';')
        
        for pair in pairs:
            if '=' in pair:
                drug_class, status = pair.split('=')
                phenotypes[drug_class.strip()] = status.strip()
        
        return phenotypes
    
    def map_drug_class_to_antibiotics(self, drug_class: str) -> List[str]:
        """Map NCBI drug class to specific antibiotics"""
        mapping = {
            'AMINOGLYCOSIDE': ['Gentamicin', 'Amikacin', 'Tobramycin', 'Streptomycin'],
            'BETA-LACTAM': ['Amoxicillin', 'Ampicillin', 'Ceftriaxone', 'Cefotaxime', 'Meropenem', 'Imipenem'],
            'FLUOROQUINOLONE': ['Ciprofloxacin', 'Levofloxacin', 'Ofloxacin'],
            'MACROLIDE': ['Azithromycin', 'Erythromycin'],
            'TETRACYCLINE': ['Tetracycline', 'Doxycycline'],
            'SULFONAMIDE': ['Trimethoprim-sulfamethoxazole', 'Sulfamethoxazole'],
            'PHENICOL': ['Chloramphenicol'],
            'GLYCOPEPTIDE': ['Vancomycin'],
            'POLYMYXIN': ['Colistin'],
            'CARBAPENEM': ['Meropenem', 'Imipenem', 'Ertapenem']
        }
        
        return mapping.get(drug_class.upper(), [])
    
    def create_training_dataset(self):
        """Create final training dataset"""
        logger.info("Creating training dataset...")
        
        # Load metadata
        metadata_file = self.ncbi_dir / "metadata.csv"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
        
        metadata = pd.read_csv(metadata_file)
        logger.info(f"Loaded {len(metadata)} metadata records")
        
        # Load downloaded genomes info
        genomes_file = self.ncbi_dir / "downloaded_genomes.csv"
        if genomes_file.exists():
            genomes_df = pd.read_csv(genomes_file)
            metadata = metadata.merge(genomes_df, on='biosample_id', how='inner')
            logger.info(f"Matched {len(metadata)} records with genome files")
        
        # Expand to individual antibiotic-level records
        training_data = []
        
        for idx, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Processing samples"):
            # Parse AMR phenotypes
            phenotypes = self.parse_amr_phenotype(row['amr_genotypes'])
            
            if not phenotypes:
                continue
            
            # Verify genome file exists
            genome_path = row.get('file_path', '')
            if not genome_path or not Path(genome_path).exists():
                continue
            
            # Expand to individual antibiotics
            for drug_class, status in phenotypes.items():
                antibiotics = self.map_drug_class_to_antibiotics(drug_class)
                
                for antibiotic in antibiotics:
                    training_data.append({
                        'biosample_id': row['biosample_id'],
                        'organism': row['organism'],
                        'strain': row.get('strain', ''),
                        'isolation_source': row.get('isolation_source', ''),
                        'collection_date': row.get('collection_date', ''),
                        'geographic_location': row.get('geographic_location', ''),
                        'antibiotic': antibiotic,
                        'drug_class': drug_class,
                        'phenotype': status,
                        'resistance': 1 if status == 'RESISTANT' else 0,
                        'genome_path': genome_path
                    })
        
        # Create DataFrame
        df = pd.DataFrame(training_data)
        
        # Save
        output_file = self.output_dir / "training_dataset.csv"
        df.to_csv(output_file, index=False)
        
        logger.info(f"✅ Created training dataset with {len(df)} records")
        logger.info(f"Saved to {output_file}")
        
        # Print statistics
        logger.info("\nDataset Statistics:")
        logger.info(f"Total samples: {df['biosample_id'].nunique()}")
        logger.info(f"Organisms: {df['organism'].nunique()}")
        logger.info(f"Antibiotics: {df['antibiotic'].nunique()}")
        logger.info(f"\nResistance distribution:")
        logger.info(df.groupby('antibiotic')['resistance'].value_counts())
        
        return df
    
    def decompress_genomes(self):
        """Decompress .gz genome files"""
        genomes_dir = self.ncbi_dir / "genomes"
        
        gz_files = list(genomes_dir.glob("*.gz"))
        logger.info(f"Decompressing {len(gz_files)} genome files...")
        
        for gz_file in tqdm(gz_files):
            output_file = gz_file.with_suffix('')
            
            if output_file.exists():
                continue
            
            try:
                with gzip.open(gz_file, 'rb') as f_in:
                    with open(output_file, 'wb') as f_out:
                        f_out.write(f_in.read())
                
                # Remove .gz file to save space
                gz_file.unlink()
                
            except Exception as e:
                logger.error(f"Error decompressing {gz_file}: {e}")
        
        logger.info("✅ Decompression complete")

# Usage
if __name__ == "__main__":
    preprocessor = AMRDataPreprocessor()
    
    # Step 1: Decompress genomes
    preprocessor.decompress_genomes()
    
    # Step 2: Create training dataset
    df = preprocessor.create_training_dataset()
    
    print("\n✅ Preprocessing complete!")
    print(f"Training data saved to: data/processed/training_dataset.csv")
```

**Run preprocessing:**
```bash
python src/preprocessing/data_preprocessor.py
```

---

## Phase 3: ML Model Development (Week 7-10)

### Week 7: Feature Extraction

*[Continue with detailed implementation of feature extraction, model training, evaluation, and all remaining phases...]*

---

**Note:** This is a comprehensive plan but I've reached the length limit. Would you like me to:

1. Continue with the remaining phases (ML Model Development, Backend, Frontend, Testing, Deployment)
2. Create separate detailed files for each phase
3. Focus on any specific section you want more detail on

The complete implementation will be 150-200 pages covering all aspects from code to deployment to commercialization.