# DeepAMR: Deep Learning for Antimicrobial Resistance Prediction

A comprehensive machine learning platform for predicting antimicrobial resistance (AMR) from whole-genome sequences using deep learning approaches.

---

## Table of Contents

1. [Overview](#overview)
2. [Data Sources](#data-sources)
3. [Data Collection](#data-collection)
4. [Data Preprocessing](#data-preprocessing)
5. [Feature Extraction](#feature-extraction)
6. [Dataset Summary](#dataset-summary)
7. [Reproducibility](#reproducibility)
8. [Project Structure](#project-structure)
9. [Installation](#installation)
10. [Usage](#usage)
11. [Citation](#citation)
12. [License](#license)

---

## Overview

Antimicrobial resistance (AMR) is a critical global health threat. This project develops machine learning models to predict AMR phenotypes directly from bacterial whole-genome sequences (WGS), enabling rapid identification of resistant isolates without time-consuming culture-based susceptibility testing.

### Key Features

- Multi-species AMR prediction across clinically relevant pathogens
- K-mer based feature extraction from whole-genome sequences
- Multi-label classification for simultaneous prediction of resistance to multiple drug classes
- Reproducible data collection and preprocessing pipeline
- Integration with NCBI's AMRFinderPlus for ground truth labeling

---

## Data Sources

### Primary Data Source: NCBI Pathogen Detection Database

- **URL**: https://www.ncbi.nlm.nih.gov/pathogens/
- **Description**: NCBI's Pathogen Detection system identifies and catalogs antimicrobial resistance genes and mutations in bacterial pathogens from submitted genome sequences.
- **Data Retrieved**: January 2025

### AMR Gene Detection: NCBI AMRFinderPlus

- **URL**: https://github.com/ncbi/amr
- **Version**: Latest available via Bioconda
- **Description**: AMRFinderPlus identifies AMR genes, resistance-associated point mutations, and select other genes using NCBI's curated Reference Gene Database.

### Supplementary Databases (for reference)

| Database | URL | Description |
|----------|-----|-------------|
| CARD | https://card.mcmaster.ca/ | Comprehensive Antibiotic Resistance Database |
| ResFinder | https://cge.food.dtu.dk/services/ResFinder/ | Resistance gene identification |
| PATRIC | https://www.patricbrc.org/ | Pathosystems Resource Integration Center |

---

## Data Collection

### Target Organisms

We collected whole-genome sequences for 20 clinically relevant bacterial species:

| Species | Clinical Relevance |
|---------|-------------------|
| *Acinetobacter baumannii* | Hospital-acquired infections, MDR |
| *Campylobacter jejuni* | Foodborne gastroenteritis |
| *Citrobacter freundii* | Opportunistic infections |
| *Clostridioides difficile* | Healthcare-associated diarrhea |
| *Enterococcus faecalis* | Endocarditis, UTIs |
| *Enterococcus faecium* | VRE infections |
| *Escherichia coli* | UTIs, sepsis, foodborne illness |
| *Haemophilus influenzae* | Respiratory infections, meningitis |
| *Klebsiella pneumoniae* | Pneumonia, bloodstream infections |
| *Listeria monocytogenes* | Foodborne listeriosis |
| *Mycobacterium tuberculosis* | Tuberculosis |
| *Neisseria gonorrhoeae* | Gonorrhea |
| *Neisseria meningitidis* | Meningitis |
| *Pseudomonas aeruginosa* | Hospital-acquired infections |
| *Salmonella enterica* | Foodborne salmonellosis |
| *Serratia marcescens* | Opportunistic infections |
| *Staphylococcus aureus* | Skin infections, MRSA |
| *Streptococcus pneumoniae* | Pneumonia, meningitis |
| *Streptococcus pyogenes* | Pharyngitis, necrotizing fasciitis |
| *Vibrio cholerae* | Cholera |

### Collection Methodology

#### Step 1: BioSample Identification

```python
# Query NCBI BioSample database for each target organism
query = f'"{organism}"[Organism]'
# Retrieved up to 500 samples per organism
```

#### Step 2: Metadata Retrieval

For each BioSample, we retrieved:
- `biosample_id`: Unique NCBI BioSample identifier
- `accession`: BioSample accession number (SAMN*)
- `organism`: Species name
- `strain`: Strain designation
- `isolation_source`: Sample origin (e.g., blood, wound, urine)
- `host`: Host organism
- `collection_date`: Date of sample collection
- `geo_loc_name`: Geographic location
- `amr_genotypes`: Known AMR genotypes (when available)
- `amr_phenotypes`: Laboratory-confirmed AMR phenotypes (when available)

#### Step 3: Assembly Identification

For each BioSample, we queried the NCBI Assembly database to identify associated genome assemblies:
- Prioritized RefSeq assemblies over GenBank
- Recorded assembly accession and status

#### Step 4: Genome Download

Downloaded genome sequences in FASTA format (gzip compressed):
- Source: NCBI FTP servers
- Format: `{biosample_id}.fna.gz`
- Content: Complete or draft genome assembly

### Collection Script

```bash
# Run data collection
python -c "
from src.data_collection.ncbi_collector import NCBICollector
from src.data_collection.config import CollectionConfig

config = CollectionConfig(
    ncbi_email='your.email@example.com',
    target_organisms=[
        'Escherichia coli',
        'Klebsiella pneumoniae',
        'Staphylococcus aureus',
        # ... additional organisms
    ],
    max_samples_per_organism=500,
    max_downloads=1000
)

collector = NCBICollector(config)
collector.collect()
"
```

---

## Data Preprocessing

### AMR Gene Detection Pipeline

Since laboratory-confirmed AMR phenotype data is sparse in public databases, we use AMRFinderPlus to detect AMR genes and infer resistance phenotypes.

#### Step 1: AMRFinderPlus Installation

```bash
# Install via Conda
conda install -c bioconda -c conda-forge ncbi-amrfinderplus

# Update AMR database
amrfinder -u
```

#### Step 2: AMR Gene Detection

For each genome, AMRFinderPlus identifies:
- AMR genes conferring resistance
- Point mutations associated with resistance
- Drug class affected by each gene/mutation

```bash
# Run AMRFinderPlus on a genome
amrfinder -n genome.fna -o output.tsv --plus --organism Escherichia
```

#### Step 3: AMR Label Generation

AMR gene detections are converted to a multi-label binary matrix:

```
biosample_id | aminoglycoside | beta-lactam | tetracycline | ...
-------------|----------------|-------------|--------------|----
54802194     | 1              | 1           | 0            | ...
54529271     | 1              | 1           | 1            | ...
```

- **1** = At least one AMR gene detected for this drug class
- **0** = No AMR genes detected for this drug class

### Feature Extraction Pipeline

#### K-mer Feature Extraction

We extract k-mer (k=6) frequency features from genome sequences:

1. **K-mer Counting**: Count occurrences of all 6-mers in each genome
2. **Sampling Strategy**: For large genomes (>100,000 k-mers), sample 100,000 positions uniformly to ensure tractable computation
3. **Feature Selection**: Select top 500 most frequent k-mers across all genomes
4. **Normalization**: Normalize counts by the number of k-mers sampled

```python
# Feature extraction parameters
k = 6                    # K-mer size
max_features = 500       # Number of k-mer features
sample_size = 100000     # Positions sampled per genome
```

#### Rationale for K-mer Features

- **Species-agnostic**: K-mer frequencies capture sequence composition without requiring alignment
- **Computationally efficient**: Linear time complexity O(n) for k-mer counting
- **Biologically meaningful**: K-mer signatures reflect GC content, codon usage, and repetitive elements
- **Proven effectiveness**: K-mer features have been successfully used in bacterial species classification and AMR prediction

### Data Splitting

Data is split into training, validation, and test sets:

| Split | Proportion | Purpose |
|-------|------------|---------|
| Training | 70% | Model training |
| Validation | 10% | Hyperparameter tuning |
| Test | 20% | Final evaluation |

- **Stratification**: Applied for organism classification (multiclass)
- **Random split**: Applied for AMR prediction (multilabel)
- **Random seed**: 42 (for reproducibility)

---

## Feature Extraction

### K-mer Feature Details

| Parameter | Value | Description |
|-----------|-------|-------------|
| K-mer size (k) | 6 | Hexamer nucleotide sequences |
| Vocabulary size | 4^6 = 4,096 | All possible 6-mers |
| Selected features | 500 | Top k-mers by frequency |
| Normalization | Frequency | Counts divided by total k-mers |

### Feature Matrix Dimensions

| Dataset | Samples | Features |
|---------|---------|----------|
| Organism Classification | 862 | 500 |
| AMR Prediction | 862 | 500 |

### Sample K-mer Features

Top 10 most frequent k-mers (representative):
1. AAAAAA (poly-A)
2. TTTTTT (poly-T)
3. AAAAAT
4. ATTTTT
5. TAAAAA
6. TTTTTA
7. TTTTTC
8. TTTAAA
9. GAAAAA
10. AAAATA

---

## Dataset Summary

### Overall Statistics

| Metric | Value |
|--------|-------|
| Total metadata records | 10,000 |
| Genomes with sequences | 862 |
| Unique organisms | 20 |
| Min genome length | 2,509,495 bp |
| Max genome length | 7,554,668 bp |
| Mean genome length | 4,679,661 bp |
| Median genome length | 4,332,278 bp |

### Organism Classification Dataset

**Task**: Multiclass classification (8 classes)

| Organism | Samples |
|----------|---------|
| *Acinetobacter baumannii* | Variable |
| *Enterococcus faecalis* | Variable |
| *Enterococcus faecium* | Variable |
| *Escherichia coli* | Variable |
| *Klebsiella pneumoniae* | Variable |
| *Pseudomonas aeruginosa* | Variable |
| *Salmonella enterica* | Variable |
| *Staphylococcus aureus* | Variable |

**Output files**:
- `data/processed/ncbi/ncbi_organism_X_{train,val,test}.npy`
- `data/processed/ncbi/ncbi_organism_y_{train,val,test}.npy`
- `data/processed/ncbi/ncbi_organism_metadata.json`

### AMR Prediction Dataset

**Task**: Multilabel classification (11 drug classes)

| Drug Class | Resistant Samples | Prevalence |
|------------|-------------------|------------|
| Aminoglycoside | 643 | 74.6% |
| Beta-lactam | 597 | 69.3% |
| Tetracycline | 505 | 58.6% |
| Macrolide | 399 | 46.3% |
| Sulfonamide | 391 | 45.4% |
| Phenicol | 298 | 34.6% |
| Fosfomycin | 218 | 25.3% |
| Trimethoprim | 216 | 25.1% |
| Quinolone | 133 | 15.4% |
| Glycopeptide | 52 | 6.0% |
| Rifampicin | 30 | 3.5% |

**Note**: Drug classes with <10 resistant samples were excluded (colistin, fusidic acid, nitroimidazole, oxazolidinone).

**Output files**:
- `data/processed/ncbi/ncbi_amr_X_{train,val,test}.npy`
- `data/processed/ncbi/ncbi_amr_y_{train,val,test}.npy`
- `data/processed/ncbi/ncbi_amr_metadata.json`

### Data Split Summary

| Split | Samples | Proportion |
|-------|---------|------------|
| Training | 602 | 70% |
| Validation | 87 | 10% |
| Test | 173 | 20% |

---

## Reproducibility

### Environment Setup

```bash
# Create conda environment
conda create -n deepamr python=3.10
conda activate deepamr

# Install dependencies
pip install numpy pandas scikit-learn biopython tqdm requests

# Install AMRFinderPlus (for AMR annotation)
conda install -c bioconda -c conda-forge ncbi-amrfinderplus
amrfinder -u  # Update database
```

### Reproducing the Dataset

#### Option 1: Using Existing AMR Annotations

If AMR annotations already exist (`data/raw/ncbi/amr_annotations/amr_labels.csv`):

```python
from src.preprocessing.ncbi_preprocessor import NCBIPreprocessor

# Initialize preprocessor
preprocessor = NCBIPreprocessor(
    ncbi_dir='data/raw/ncbi',
    output_dir='data/processed/ncbi'
)

# Prepare organism classification data
organism_data = preprocessor.prepare_organism_classification_data(
    k=6,
    max_features=500,
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    min_samples_per_class=5
)
preprocessor.save_processed_data(organism_data, prefix='ncbi_organism')

# Prepare AMR prediction data
amr_data = preprocessor.prepare_amr_prediction_data(
    amr_labels_file='data/raw/ncbi/amr_annotations/amr_labels.csv',
    k=6,
    max_features=500,
    test_size=0.2,
    val_size=0.1,
    random_state=42,
    min_samples_per_drug=10
)
preprocessor.save_processed_data(amr_data, prefix='ncbi_amr')
```

#### Option 2: Full Pipeline from Scratch

```bash
# Run complete AMR annotation and preprocessing pipeline
python scripts/run_ncbi_amr_pipeline.py
```

This script:
1. Checks AMRFinderPlus installation
2. Runs AMRFinderPlus on all genomes (~5 min/genome)
3. Creates AMR labels from gene detections
4. Preprocesses data for ML modeling

### Random Seeds

All random operations use fixed seeds for reproducibility:

| Operation | Seed |
|-----------|------|
| K-mer sampling | 42 + sequence_index |
| Train/test split | 42 |
| Validation split | 42 |

### Software Versions

| Software | Version |
|----------|---------|
| Python | 3.10+ |
| NumPy | 1.24+ |
| Pandas | 2.0+ |
| Scikit-learn | 1.3+ |
| BioPython | 1.81+ |
| AMRFinderPlus | Latest |

---

## Project Structure

```
DeepAMR/
├── README.md                           # This documentation
├── data/
│   ├── raw/
│   │   └── ncbi/
│   │       ├── metadata/               # Per-organism metadata CSVs
│   │       │   ├── Escherichia_coli_metadata.csv
│   │       │   ├── Klebsiella_pneumoniae_metadata.csv
│   │       │   └── ...
│   │       ├── genomes/                # Compressed FASTA files
│   │       │   ├── 54802194.fna.gz
│   │       │   └── ...
│   │       ├── amr_annotations/        # AMRFinderPlus results
│   │       │   ├── all_amr_annotations.csv
│   │       │   ├── amr_labels.csv
│   │       │   └── drug_class_mapping.json
│   │       └── complete_metadata.csv
│   └── processed/
│       └── ncbi/
│           ├── ncbi_organism_X_train.npy
│           ├── ncbi_organism_X_val.npy
│           ├── ncbi_organism_X_test.npy
│           ├── ncbi_organism_y_train.npy
│           ├── ncbi_organism_y_val.npy
│           ├── ncbi_organism_y_test.npy
│           ├── ncbi_organism_metadata.json
│           ├── ncbi_amr_X_train.npy
│           ├── ncbi_amr_X_val.npy
│           ├── ncbi_amr_X_test.npy
│           ├── ncbi_amr_y_train.npy
│           ├── ncbi_amr_y_val.npy
│           ├── ncbi_amr_y_test.npy
│           └── ncbi_amr_metadata.json
├── src/
│   ├── data_collection/
│   │   ├── ncbi_collector.py           # NCBI data collection
│   │   ├── amrfinder_annotator.py      # AMRFinderPlus integration
│   │   └── config.py                   # Configuration
│   ├── preprocessing/
│   │   ├── ncbi_preprocessor.py        # Main preprocessing pipeline
│   │   └── amr_gene_detector.py        # K-mer based AMR detection
│   └── ml/
│       ├── model_trainer.py            # Model training
│       └── model_evaluator.py          # Evaluation metrics
├── scripts/
│   ├── run_ncbi_amr_pipeline.py        # Complete pipeline script
│   └── setup_amrfinder.sh              # AMRFinderPlus setup
└── notebooks/
    └── 03_resfinder_data_exploration.ipynb
```

---

## Installation

### Prerequisites

- Python 3.10 or higher
- Conda (recommended for AMRFinderPlus)
- At least 16 GB RAM (for k-mer extraction)
- ~10 GB disk space for genome data

### Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/DeepAMR.git
cd DeepAMR

# Create environment
conda create -n deepamr python=3.10
conda activate deepamr

# Install Python dependencies
pip install -r requirements.txt

# Install AMRFinderPlus (optional, for AMR annotation)
conda install -c bioconda -c conda-forge ncbi-amrfinderplus
amrfinder -u
```

### Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
biopython>=1.81
tqdm>=4.65.0
requests>=2.31.0
```

---

## Usage

### Loading Preprocessed Data

```python
import numpy as np
import json

# Load organism classification data
X_train = np.load('data/processed/ncbi/ncbi_organism_X_train.npy')
X_val = np.load('data/processed/ncbi/ncbi_organism_X_val.npy')
X_test = np.load('data/processed/ncbi/ncbi_organism_X_test.npy')
y_train = np.load('data/processed/ncbi/ncbi_organism_y_train.npy')
y_val = np.load('data/processed/ncbi/ncbi_organism_y_val.npy')
y_test = np.load('data/processed/ncbi/ncbi_organism_y_test.npy')

with open('data/processed/ncbi/ncbi_organism_metadata.json') as f:
    metadata = json.load(f)

print(f"Training samples: {X_train.shape[0]}")
print(f"Features: {X_train.shape[1]}")
print(f"Classes: {metadata['class_names']}")
```

### Training a Simple Model

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=metadata['class_names']))
```

### AMR Prediction (Multilabel)

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Load AMR data
X_train = np.load('data/processed/ncbi/ncbi_amr_X_train.npy')
y_train = np.load('data/processed/ncbi/ncbi_amr_y_train.npy')

# Train multilabel classifier
base_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
clf = MultiOutputClassifier(base_clf)
clf.fit(X_train, y_train)
```

---

## Citation

If you use this dataset or code in your research, please cite:

```bibtex
@software{deepamr2025,
  title = {DeepAMR: Deep Learning for Antimicrobial Resistance Prediction},
  author = {[Your Name]},
  year = {2025},
  url = {https://github.com/yourusername/DeepAMR}
}
```

### Data Sources

Please also cite the original data sources:

```bibtex
@article{feldgarden2021amrfinderplus,
  title={AMRFinderPlus and the Reference Gene Catalog facilitate examination of the genomic links among antimicrobial resistance, stress response, and virulence},
  author={Feldgarden, Michael and Brover, Vyacheslav and Gonzalez-Escalona, Narjol and others},
  journal={Scientific Reports},
  volume={11},
  number={1},
  pages={12728},
  year={2021},
  publisher={Nature Publishing Group}
}

@misc{ncbi_pathogen_detection,
  title = {NCBI Pathogen Detection},
  author = {{National Center for Biotechnology Information}},
  url = {https://www.ncbi.nlm.nih.gov/pathogens/},
  year = {2025}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Data License

- NCBI data is in the public domain
- AMRFinderPlus is a US Government work and is in the public domain

---

## Acknowledgments

- NCBI Pathogen Detection team for maintaining the database
- NCBI AMRFinderPlus developers for the AMR detection tool
- The scientific community for contributing genome sequences and AMR data

---

## Contact

For questions or issues, please open an issue on GitHub or contact:
- Email: [your.email@example.com]
- GitHub: [https://github.com/yourusername/DeepAMR]
