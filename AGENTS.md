# AGENTS.md

This file provides guidance to agentic coding agents working in this repository.

## Project Overview

DeepAMR is a machine learning platform for predicting antimicrobial resistance (AMR) from bacterial whole-genome sequences. The project has two main ML tasks:
1. **Organism Classification** - Multiclass classification of bacterial species from genome sequences
2. **AMR Prediction** - Multilabel classification predicting resistance to multiple drug classes

## Development Environment

- **Python**: 3.13 (specified in `.python-version`)
- **Virtual Environment**: `.venv/` directory
- **Frontend**: Next.js 16 with React 19 (in `frontend/`)
- **Bioinformatics Tools**: AMRFinderPlus (requires conda)

### Initial Setup

```bash
# Python environment (from project root)
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Frontend (from frontend/)
cd frontend && npm install

# AMRFinderPlus (optional, for AMR annotation)
conda install -c bioconda -c conda-forge ncbi-amrfinderplus
amrfinder -u  # Update database
```

## Common Commands

### Python/ML Development

```bash
# Preprocess NCBI data
python -m src.preprocessing.ncbi_preprocessor

# Run full AMR pipeline (annotation + preprocessing)
python scripts/run_ncbi_amr_pipeline.py

# Train models
python -m src.ml.model_trainer --data-dir data/processed/ncbi --prefix ncbi_organism --model random_forest
python -m src.ml.model_trainer --data-dir data/processed/ncbi --prefix ncbi_amr --model all

# List available ML models
python -m src.ml.model_trainer --list-models
```

### Build/Lint/Test Commands

```bash
# Python/ML Development
python -m pytest                    # Run all tests (when implemented)
python -m pytest tests/test_module.py  # Run specific test file
python -m pytest -k "test_function"  # Run specific test function

# Code quality (recommended to add to pyproject.toml)
ruff check .                        # Linting (fast)
ruff check --fix .                  # Auto-fix linting issues
ruff format .                       # Code formatting
mypy src/                           # Type checking
black --check .                     # Check formatting
black .                             # Format code
isort .                             # Sort imports

# Frontend Development
cd frontend
npm run dev      # Start dev server at localhost:3000
npm run build    # Production build
npm run lint     # Run ESLint
```

## Architecture

### Data Pipeline

```
data/raw/ncbi/
├── metadata/           # Per-organism CSV metadata
├── genomes/            # Compressed FASTA files (.fna.gz)
└── amr_annotations/    # AMRFinderPlus results

data/processed/ncbi/
├── ncbi_organism_*.npy  # Organism classification dataset
├── ncbi_amr_*.npy       # AMR prediction dataset
└── *_metadata.json      # Feature names, class names
```

### Source Code Structure

```
src/
├── data_collection/
│   ├── base.py              # BaseCollector abstract class
│   ├── config.py            # CollectionConfig dataclass
│   ├── ncbi_collector.py    # NCBI data collection
│   └── amrfinder_annotator.py  # AMRFinderPlus integration
├── preprocessing/
│   └── ncbi_preprocessor.py # K-mer feature extraction & data preparation
└── ml/
    ├── model_trainer.py     # Sklearn-based training pipeline
    ├── amr_classifier.py    # PyTorch neural network
    └── feature_extractor.py # Feature extraction utilities
```

### Key Classes

- **`NCBIPreprocessor`** (`src/preprocessing/ncbi_preprocessor.py`): Main preprocessing class. Extracts k-mer (k=6) features from genomes, handles train/val/test splits.
- **`AMRModelTrainer`** (`src/ml/model_trainer.py`): Training pipeline supporting multiple sklearn models (random_forest, gradient_boosting, mlp, etc.) for both multiclass and multilabel tasks.
- **`ModelFactory`** (`src/ml/model_trainer.py`): Factory for creating models with sensible defaults.
- **`BaseCollector`** (`src/data_collection/base.py`): Abstract base class for all data collectors.

### ML Approach

- **Feature Extraction**: K-mer (k=6) frequency features from genome sequences
  - 500 most frequent k-mers selected across all genomes
  - Sampling strategy for large genomes (100k positions)
- **Data Splits**: 70% train, 10% validation, 20% test (random_state=42)
- **Organism Classification**: Stratified split, 8 bacterial species
- **AMR Prediction**: Multilabel (11 drug classes), uses `OneVsRestClassifier` wrapper

### Output Files

- Models saved to `models/` as `.joblib` (sklearn) or `.pt` (PyTorch)
- Results saved as `*_results.json` with metrics and feature importance

## Environment Variables

Set `NCBI_EMAIL` for NCBI API access when collecting data:
```bash
export NCBI_EMAIL="your.email@example.com"
```

## Data Sources

- **NCBI Pathogen Detection**: Primary genome/metadata source
- **AMRFinderPlus**: AMR gene detection for label generation
- **CARD/ResFinder**: Reference AMR databases (in `data/external/`)

## Code Style Guidelines

### Python Code Style

**Imports & Formatting:**
- Use standard library imports first, then third-party, then local imports
- Use `from typing import Optional, List, Dict, Tuple, Union` for type hints
- Import typing constructs using `|` syntax for Union types (Python 3.10+)
- Use `pathlib.Path` instead of `os.path`
- Import logging as `import logging` and configure as `logger = logging.getLogger(__name__)`

**Naming Conventions:**
- Classes: `PascalCase` (e.g., `NCBIPreprocessor`, `BaseCollector`)
- Functions/variables: `snake_case` (e.g., `load_processed_data`, `metadata_dir`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_K_MER_SIZE`)
- Private methods: `_leading_underscore` (e.g., `_validate_config`)

**Type Hints:**
- Always include return types in function signatures
- Use `Optional[T]` for nullable types
- Use `List[str]`, `Dict[str, int]` for collections
- Use `Path` from pathlib for file paths
- Use `pd.DataFrame` for pandas dataframes

**Error Handling:**
- Use logging instead of print statements
- Raise appropriate exceptions with descriptive messages
- Validate inputs in public methods
- Use try/except blocks for external API calls

**Docstrings:**
- Use triple quotes at module and class level
- Describe purpose and key parameters for complex functions
- Follow Google-style docstring format

**Class Structure:**
- Use dataclasses for configuration objects (`@dataclass`)
- Use abstract base classes for shared interfaces (`ABC`)
- Include `__init__` with type hints and default values
- Use property decorators for computed attributes

### Frontend Code Style

**React/Next.js:**
- Use functional components with TypeScript
- Use hooks (`useState`, `useEffect`) following React patterns
- Import Radix UI components directly from their packages
- Use Tailwind CSS classes for styling (no inline styles)
- Use shadcn/ui components for consistent UI

**File Organization:**
- Keep components in component-specific folders
- Use `.tsx` extension for React components with TypeScript
- Export types and interfaces separately

### Testing Guidelines

**Python Tests:**
- Place tests in `tests/` directory mirroring `src/` structure
- Use descriptive test names: `test_function_behavior_when_condition`
- Use pytest fixtures for common setup
- Test both happy path and error conditions

**Frontend Tests:**
- Use Jest and React Testing Library
- Test user interactions and component rendering
- Mock API calls in unit tests

### Data Handling

**File Paths:**
- Always use `Path` objects from pathlib
- Use `Path.mkdir(parents=True, exist_ok=True)` for directory creation
- Validate file existence before reading

**Data Processing:**
- Use numpy arrays for ML features (`.npy` format)
- Include JSON metadata alongside processed data
- Use pandas DataFrames for tabular data processing
- Log data shapes and key statistics during processing

### ML/AI Patterns

**Model Training:**
- Use sklearn for traditional ML models
- Use PyTorch for deep learning models
- Save sklearn models with `joblib.dump()`
- Save PyTorch models with `.pt` extension
- Include model hyperparameters in saved metadata

**Feature Engineering:**
- Use sklearn transformers for preprocessing
- Store feature names and preprocessing steps
- Validate feature dimensions before model training

## Notes

- Processed data uses numpy arrays (`.npy`) with accompanying JSON metadata
- Frontend uses shadcn/ui components with Tailwind CSS
- Models directory has `checkpoints/`, `final/`, `pretrained/`, `huggingface/` subdirs
- GitIgnore includes common Python, Node.js, and ML artifacts
- Project supports both traditional ML (sklearn) and deep learning (PyTorch) approaches