# Programmatic Data Collection Guide
## Automated AMR Data Collection from Multiple Sources

This guide provides complete, automated scripts to collect bacterial genome and AMR data programmatically.

---

## Table of Contents
1. [Overview](#overview)
2. [Setup](#setup)
3. [NCBI Data Collection](#ncbi-data-collection)
4. [CARD Database](#card-database)
5. [PATRIC Database](#patric-database)
6. [ResFinder Database](#resfinder-database)
7. [Additional Sources](#additional-sources)
8. [Master Collection Script](#master-collection-script)
9. [Data Validation](#data-validation)

---

## Overview

### What We'll Collect
- **Bacterial genomes** (FASTA/FASTQ files)
- **AMR phenotypes** (resistance/susceptible labels)
- **AMR genes** (known resistance genes)
- **Metadata** (species, strain, location, date)

### Data Sources
1. **NCBI Pathogen Detection** - Largest public database
2. **CARD** - Comprehensive AMR gene database
3. **PATRIC** - Bacterial bioinformatics database
4. **ResFinder** - AMR gene sequences
5. **ENA** - European genome archive

### Target Size
- **Training set:** 1,000-5,000 genomes
- **Test set:** 200-500 genomes
- **Validation set:** 200-500 genomes

---

## Setup

### Install Dependencies

```bash
# Create conda environment (recommended)
conda create -n amr-data python=3.10
conda activate amr-data

# Or use venv
python -m venv venv
source venv/bin/activate
```

```bash
# Install required packages
pip install biopython pandas requests tqdm
pip install ncbi-genome-download
pip install asyncio aiohttp aiofiles
pip install beautifulsoup4 lxml
pip install ftputil
pip install pyyaml
```

### Directory Structure

```bash
mkdir -p data/{raw,processed,external}
mkdir -p data/raw/{ncbi,patric,ena,combined}
mkdir -p data/external/{card,resfinder,aro}
mkdir -p logs
```

---

## NCBI Data Collection

### Method 1: Using Entrez API (Recommended)

```python
# src/data_collection/ncbi_collector.py
"""
Comprehensive NCBI data collector using Entrez API
"""
import os
import time
import gzip
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import pandas as pd
import requests
from Bio import Entrez, SeqIO
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/ncbi_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NCBIDataCollector:
    """Automated NCBI data collection"""
    
    def __init__(self, email: str, api_key: Optional[str] = None, output_dir: str = "data/raw/ncbi"):
        """
        Initialize NCBI collector
        
        Args:
            email: Your email (required by NCBI)
            api_key: NCBI API key (get from https://www.ncbi.nlm.nih.gov/account/)
            output_dir: Output directory
        """
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key  # Allows 10 requests/second instead of 3
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.genomes_dir = self.output_dir / "genomes"
        self.genomes_dir.mkdir(exist_ok=True)
        
        self.metadata_dir = self.output_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Rate limiting
        self.request_delay = 0.35 if api_key else 0.4  # seconds
    
    def search_pathogen_isolates(self, 
                                 organism: str,
                                 filters: Optional[Dict] = None,
                                 max_results: int = 1000) -> List[str]:
        """
        Search for pathogen isolates in NCBI
        
        Args:
            organism: Organism name (e.g., "Escherichia coli")
            filters: Additional filters (country, date, etc.)
            max_results: Maximum number of results
        
        Returns:
            List of BioSample IDs
        """
        logger.info(f"Searching NCBI for {organism}...")
        
        # Build search query
        query_parts = [f'"{organism}"[Organism]']
        
        if filters:
            if 'country' in filters:
                query_parts.append(f'"{filters["country"]}"[Country]')
            if 'collection_date_start' in filters:
                query_parts.append(f'{filters["collection_date_start"]}[Collection Date] : 3000[Collection Date]')
            if 'has_amr' in filters and filters['has_amr']:
                query_parts.append('AMR[Filter]')
        
        query = " AND ".join(query_parts)
        logger.info(f"Query: {query}")
        
        try:
            # Search BioSample database
            handle = Entrez.esearch(
                db="biosample",
                term=query,
                retmax=max_results,
                usehistory="y"
            )
            results = Entrez.read(handle)
            handle.close()
            
            biosample_ids = results["IdList"]
            logger.info(f"Found {len(biosample_ids)} BioSample records")
            
            time.sleep(self.request_delay)
            return biosample_ids
            
        except Exception as e:
            logger.error(f"Error searching NCBI: {e}")
            return []
    
    def fetch_biosample_metadata(self, biosample_ids: List[str]) -> pd.DataFrame:
        """
        Fetch detailed metadata for BioSample IDs
        
        Args:
            biosample_ids: List of BioSample IDs
        
        Returns:
            DataFrame with metadata
        """
        logger.info(f"Fetching metadata for {len(biosample_ids)} samples...")
        
        metadata_list = []
        batch_size = 100
        
        for i in tqdm(range(0, len(biosample_ids), batch_size), desc="Fetching metadata"):
            batch_ids = biosample_ids[i:i+batch_size]
            
            try:
                handle = Entrez.efetch(
                    db="biosample",
                    id=batch_ids,
                    rettype="xml"
                )
                records = Entrez.read(handle)
                handle.close()
                
                # Parse each record
                for record in records:
                    metadata = self._parse_biosample_record(record)
                    if metadata:
                        metadata_list.append(metadata)
                
                time.sleep(self.request_delay)
                
            except Exception as e:
                logger.error(f"Error fetching batch {i}: {e}")
                continue
        
        df = pd.DataFrame(metadata_list)
        logger.info(f"Retrieved {len(df)} metadata records")
        
        return df
    
    def _parse_biosample_record(self, record: Dict) -> Optional[Dict]:
        """Parse a BioSample XML record"""
        try:
            # Get BioSample ID
            biosample_id = record['Ids'][0]['content']
            
            # Extract attributes
            attributes = {}
            for attr in record.get('Attributes', []):
                attr_name = attr.get('@attribute_name', '')
                attr_value = attr.get('#text', '')
                attributes[attr_name] = attr_value
            
            # Extract links to other databases
            links = {}
            for db_link in record.get('Links', []):
                if 'Link' in db_link:
                    for link in db_link['Link']:
                        db_name = link.get('@label', '')
                        link_id = link.get('@target', '')
                        links[db_name] = link_id
            
            metadata = {
                'biosample_id': biosample_id,
                'accession': record.get('Accession', ''),
                'organism': attributes.get('organism', ''),
                'strain': attributes.get('strain', ''),
                'isolation_source': attributes.get('isolation_source', ''),
                'host': attributes.get('host', ''),
                'collection_date': attributes.get('collection_date', ''),
                'geo_loc_name': attributes.get('geo_loc_name', ''),
                'lat_lon': attributes.get('lat_lon', ''),
                'collected_by': attributes.get('collected_by', ''),
                'serovar': attributes.get('serovar', ''),
                'serotype': attributes.get('serotype', ''),
                
                # AMR data
                'amr_genotypes': attributes.get('AMR_genotypes', ''),
                'amr_phenotypes': attributes.get('AMR_phenotypes', ''),
                
                # Clinical data
                'disease': attributes.get('disease', ''),
                'patient_age': attributes.get('patient_age', ''),
                'patient_gender': attributes.get('patient_gender', ''),
                
                # Database links
                'sra_accession': links.get('SRA', ''),
                'assembly_accession': links.get('Assembly', ''),
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error parsing record: {e}")
            return None
    
    def find_genome_assemblies(self, biosample_ids: List[str]) -> pd.DataFrame:
        """
        Find genome assemblies for BioSample IDs
        
        Args:
            biosample_ids: List of BioSample IDs
        
        Returns:
            DataFrame with assembly information
        """
        logger.info(f"Finding genome assemblies for {len(biosample_ids)} samples...")
        
        assemblies = []
        
        for biosample_id in tqdm(biosample_ids, desc="Finding assemblies"):
            try:
                # Search for assemblies linked to this BioSample
                handle = Entrez.esearch(
                    db="assembly",
                    term=f"{biosample_id}[BioSample]"
                )
                results = Entrez.read(handle)
                handle.close()
                
                assembly_ids = results.get("IdList", [])
                
                if not assembly_ids:
                    continue
                
                # Get assembly details
                handle = Entrez.esummary(db="assembly", id=assembly_ids[0])
                summary = Entrez.read(handle)
                handle.close()
                
                assembly_info = summary['DocumentSummarySet']['DocumentSummary'][0]
                
                # Get FTP paths
                ftp_path_refseq = assembly_info.get('FtpPath_RefSeq', '')
                ftp_path_genbank = assembly_info.get('FtpPath_GenBank', '')
                
                assemblies.append({
                    'biosample_id': biosample_id,
                    'assembly_accession': assembly_info.get('AssemblyAccession', ''),
                    'assembly_name': assembly_info.get('AssemblyName', ''),
                    'assembly_status': assembly_info.get('AssemblyStatus', ''),
                    'coverage': assembly_info.get('Coverage', ''),
                    'ftp_path_refseq': ftp_path_refseq,
                    'ftp_path_genbank': ftp_path_genbank,
                })
                
                time.sleep(self.request_delay)
                
            except Exception as e:
                logger.error(f"Error finding assembly for {biosample_id}: {e}")
                continue
        
        df = pd.DataFrame(assemblies)
        logger.info(f"Found {len(df)} genome assemblies")
        
        return df
    
    def download_genome(self, ftp_path: str, biosample_id: str) -> Optional[str]:
        """
        Download genome from FTP
        
        Args:
            ftp_path: FTP path to genome
            biosample_id: BioSample ID for naming
        
        Returns:
            Path to downloaded file
        """
        if not ftp_path:
            return None
        
        try:
            # Get assembly name from FTP path
            assembly_name = ftp_path.split('/')[-1]
            
            # Construct FASTA file URL
            fasta_filename = f"{assembly_name}_genomic.fna.gz"
            fasta_url = f"{ftp_path}/{fasta_filename}"
            
            # Convert FTP to HTTPS
            if fasta_url.startswith('ftp://'):
                fasta_url = fasta_url.replace('ftp://', 'https://')
            
            # Output file
            output_file = self.genomes_dir / f"{biosample_id}.fna.gz"
            
            # Download
            response = requests.get(fasta_url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Save
            total_size = int(response.headers.get('content-length', 0))
            with open(output_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded {biosample_id}: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error downloading {biosample_id}: {e}")
            return None
    
    def download_genomes_batch(self, assemblies_df: pd.DataFrame, max_downloads: int = 100) -> pd.DataFrame:
        """
        Download multiple genomes
        
        Args:
            assemblies_df: DataFrame with assembly information
            max_downloads: Maximum number to download
        
        Returns:
            DataFrame with download status
        """
        logger.info(f"Downloading up to {max_downloads} genomes...")
        
        downloads = []
        count = 0
        
        for idx, row in tqdm(assemblies_df.iterrows(), total=min(len(assemblies_df), max_downloads)):
            if count >= max_downloads:
                break
            
            biosample_id = row['biosample_id']
            
            # Try RefSeq first, then GenBank
            ftp_path = row['ftp_path_refseq'] if row['ftp_path_refseq'] else row['ftp_path_genbank']
            
            if not ftp_path:
                continue
            
            file_path = self.download_genome(ftp_path, biosample_id)
            
            if file_path:
                downloads.append({
                    'biosample_id': biosample_id,
                    'assembly_accession': row['assembly_accession'],
                    'file_path': file_path,
                    'download_date': datetime.now().isoformat()
                })
                count += 1
            
            time.sleep(1)  # Be nice to NCBI servers
        
        df = pd.DataFrame(downloads)
        
        # Save download log
        df.to_csv(self.output_dir / 'downloaded_genomes.csv', index=False)
        logger.info(f"✅ Downloaded {len(df)} genomes")
        
        return df
    
    def collect_complete_dataset(self,
                                  organisms: List[str],
                                  filters: Optional[Dict] = None,
                                  max_per_organism: int = 500) -> pd.DataFrame:
        """
        Complete automated collection pipeline
        
        Args:
            organisms: List of organism names
            filters: Search filters
            max_per_organism: Max samples per organism
        
        Returns:
            Complete dataset DataFrame
        """
        logger.info("="*80)
        logger.info("Starting complete NCBI data collection")
        logger.info("="*80)
        
        all_metadata = []
        all_assemblies = []
        
        for organism in organisms:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing: {organism}")
            logger.info(f"{'='*80}")
            
            # Step 1: Search
            biosample_ids = self.search_pathogen_isolates(
                organism=organism,
                filters=filters,
                max_results=max_per_organism
            )
            
            if not biosample_ids:
                logger.warning(f"No samples found for {organism}")
                continue
            
            # Step 2: Fetch metadata
            metadata = self.fetch_biosample_metadata(biosample_ids)
            all_metadata.append(metadata)
            
            # Step 3: Find assemblies
            assemblies = self.find_genome_assemblies(biosample_ids)
            all_assemblies.append(assemblies)
            
            # Save intermediate results
            metadata.to_csv(self.metadata_dir / f'{organism.replace(" ", "_")}_metadata.csv', index=False)
            assemblies.to_csv(self.metadata_dir / f'{organism.replace(" ", "_")}_assemblies.csv', index=False)
        
        # Combine all data
        logger.info("\nCombining all metadata...")
        combined_metadata = pd.concat(all_metadata, ignore_index=True)
        combined_assemblies = pd.concat(all_assemblies, ignore_index=True)
        
        # Merge metadata with assembly info
        complete_df = combined_metadata.merge(
            combined_assemblies,
            on='biosample_id',
            how='left'
        )
        
        # Save complete metadata
        complete_df.to_csv(self.output_dir / 'complete_metadata.csv', index=False)
        logger.info(f"✅ Complete metadata saved: {len(complete_df)} records")
        
        # Step 4: Download genomes
        samples_with_assemblies = complete_df[complete_df['assembly_accession'].notna()]
        
        # Filter for samples with AMR data
        if 'amr_genotypes' in samples_with_assemblies.columns:
            amr_samples = samples_with_assemblies[samples_with_assemblies['amr_genotypes'].notna()]
            logger.info(f"Found {len(amr_samples)} samples with AMR data")
            download_df = amr_samples
        else:
            download_df = samples_with_assemblies
        
        # Download
        downloads = self.download_genomes_batch(download_df, max_downloads=min(len(download_df), 200))
        
        # Merge download info
        final_df = complete_df.merge(
            downloads[['biosample_id', 'file_path', 'download_date']],
            on='biosample_id',
            how='left'
        )
        
        final_df.to_csv(self.output_dir / 'final_dataset.csv', index=False)
        
        logger.info("\n" + "="*80)
        logger.info("COLLECTION COMPLETE!")
        logger.info("="*80)
        logger.info(f"Total records: {len(final_df)}")
        logger.info(f"Downloaded genomes: {final_df['file_path'].notna().sum()}")
        logger.info(f"Samples with AMR data: {final_df['amr_genotypes'].notna().sum()}")
        logger.info(f"Output directory: {self.output_dir}")
        
        return final_df


# Example usage
if __name__ == "__main__":
    # Initialize collector
    collector = NCBIDataCollector(
        email="your.email@example.com",  # CHANGE THIS!
        api_key=None,  # Get from https://www.ncbi.nlm.nih.gov/account/
        output_dir="data/raw/ncbi"
    )
    
    # Target organisms
    organisms = [
        "Escherichia coli",
        "Klebsiella pneumoniae",
        "Staphylococcus aureus",
        "Acinetobacter baumannii",
        "Pseudomonas aeruginosa"
    ]
    
    # Filters (optional)
    filters = {
        # 'country': 'Bangladesh',  # Uncomment to filter by country
        # 'collection_date_start': '2020',  # Samples from 2020 onwards
        'has_amr': True  # Only samples with AMR data
    }
    
    # Run complete collection
    dataset = collector.collect_complete_dataset(
        organisms=organisms,
        filters=filters,
        max_per_organism=500
    )
    
    print("\n✅ Data collection complete!")
    print(f"Dataset shape: {dataset.shape}")
    print(f"\nFirst few rows:")
    print(dataset.head())
```

### Method 2: Using SRA Toolkit (For Raw Sequencing Data)

```python
# src/data_collection/sra_collector.py
"""
Download raw sequencing data from SRA
"""
import subprocess
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class SRACollector:
    """Download raw sequencing data from SRA"""
    
    def __init__(self, output_dir="data/raw/sra"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if SRA toolkit is installed
        self._check_sra_toolkit()
    
    def _check_sra_toolkit(self):
        """Check if SRA toolkit is installed"""
        try:
            result = subprocess.run(['fastq-dump', '--version'], 
                                   capture_output=True, text=True)
            logger.info(f"SRA Toolkit version: {result.stdout.split()[2]}")
        except FileNotFoundError:
            raise Exception(
                "SRA Toolkit not found. Install with:\n"
                "  conda install -c bioconda sra-tools\n"
                "or download from: https://github.com/ncbi/sra-tools"
            )
    
    def download_sra(self, sra_accession: str, paired_end: bool = True) -> list:
        """
        Download SRA data and convert to FASTQ
        
        Args:
            sra_accession: SRA accession (e.g., SRR123456)
            paired_end: Whether data is paired-end
        
        Returns:
            List of downloaded FASTQ file paths
        """
        logger.info(f"Downloading {sra_accession}...")
        
        output_files = []
        
        try:
            # Download and convert to FASTQ
            cmd = [
                'fastq-dump',
                '--outdir', str(self.output_dir),
                '--gzip',  # Compress output
            ]
            
            if paired_end:
                cmd.append('--split-files')  # Split into R1 and R2
            
            cmd.append(sra_accession)
            
            # Run command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                if paired_end:
                    output_files = [
                        str(self.output_dir / f"{sra_accession}_1.fastq.gz"),
                        str(self.output_dir / f"{sra_accession}_2.fastq.gz")
                    ]
                else:
                    output_files = [str(self.output_dir / f"{sra_accession}.fastq.gz")]
                
                logger.info(f"✓ Downloaded {sra_accession}")
            else:
                logger.error(f"Error downloading {sra_accession}: {result.stderr}")
            
        except Exception as e:
            logger.error(f"Error: {e}")
        
        return output_files
    
    def download_batch(self, sra_accessions: list, paired_end: bool = True) -> pd.DataFrame:
        """Download multiple SRA datasets"""
        results = []
        
        for acc in tqdm(sra_accessions, desc="Downloading SRA"):
            files = self.download_sra(acc, paired_end)
            results.append({
                'sra_accession': acc,
                'files': files,
                'success': len(files) > 0
            })
        
        return pd.DataFrame(results)
```

---

## CARD Database

```python
# src/data_collection/card_collector.py
"""
Download CARD (Comprehensive Antibiotic Resistance Database)
"""
import requests
import tarfile
from pathlib import Path
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class CARDCollector:
    """Download and process CARD database"""
    
    BASE_URL = "https://card.mcmaster.ca/latest"
    
    def __init__(self, output_dir="data/external/card"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_card_data(self):
        """Download complete CARD database"""
        logger.info("Downloading CARD database...")
        
        files_to_download = {
            'card_data.tar.bz2': f"{self.BASE_URL}/data",
            'protein_fasta_protein_homolog_model.fasta': f"{self.BASE_URL}/protein_fasta_protein_homolog_model.fasta",
            'nucleotide_fasta_protein_homolog_model.fasta': f"{self.BASE_URL}/nucleotide_fasta_protein_homolog_model.fasta",
            'aro_index.tsv': f"{self.BASE_URL}/aro_index.tsv",
            'aro_categories.tsv': f"{self.BASE_URL}/aro_categories.tsv",
            'aro_categories_index.tsv': f"{self.BASE_URL}/aro_categories_index.tsv",
        }
        
        for filename, url in files_to_download.items():
            output_file = self.output_dir / filename
            
            if output_file.exists():
                logger.info(f"✓ {filename} already exists")
                continue
            
            try:
                logger.info(f"Downloading {filename}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                
                with open(output_file, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
                
                logger.info(f"✓ Downloaded {filename}")
                
            except Exception as e:
                logger.error(f"Error downloading {filename}: {e}")
        
        # Extract tar file
        tar_file = self.output_dir / 'card_data.tar.bz2'
        if tar_file.exists():
            logger.info("Extracting CARD data...")
            with tarfile.open(tar_file, 'r:bz2') as tar:
                tar.extractall(self.output_dir)
            logger.info("✓ Extraction complete")
        
        logger.info("✅ CARD database download complete!")
    
    def load_aro_index(self):
        """Load ARO (Antibiotic Resistance Ontology) index"""
        import pandas as pd
        
        aro_file = self.output_dir / 'aro_index.tsv'
        if not aro_file.exists():
            logger.error("ARO index not found. Run download_card_data() first.")
            return None
        
        df = pd.read_csv(aro_file, sep='\t')
        logger.info(f"Loaded {len(df)} ARO entries")
        
        return df


# Usage
if __name__ == "__main__":
    collector = CARDCollector()
    collector.download_card_data()
    aro_df = collector.load_aro_index()
    print(aro_df.head())
```

---

## PATRIC Database

```python
# src/data_collection/patric_collector.py
"""
Download data from PATRIC (Pathosystems Resource Integration Center)
"""
import requests
import pandas as pd
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


class PATRICCollector:
    """Download genomes and AMR data from PATRIC"""
    
    API_BASE = "https://www.patricbrc.org/api"
    
    def __init__(self, output_dir="data/raw/patric"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.genomes_dir = self.output_dir / "genomes"
        self.genomes_dir.mkdir(exist_ok=True)
    
    def search_genomes(self, 
                       organism: str,
                       has_amr: bool = True,
                       limit: int = 1000) -> pd.DataFrame:
        """
        Search for genomes in PATRIC
        
        Args:
            organism: Organism name
            has_amr: Filter for genomes with AMR data
            limit: Maximum results
        
        Returns:
            DataFrame with genome metadata
        """
        logger.info(f"Searching PATRIC for {organism}...")
        
        # Build query
        query = f"genome_name:{organism}"
        if has_amr:
            query += " AND amr_phenotype:*"
        
        url = f"{self.API_BASE}/genome"
        
        params = {
            'q': query,
            'limit': limit,
            'select': 'genome_id,genome_name,organism_name,genome_status,'
                     'genome_length,contigs,chromosomes,plasmids,'
                     'isolation_country,host_name,disease,collection_date,'
                     'completion_date,sequencing_centers'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            
            logger.info(f"Found {len(df)} genomes")
            return df
            
        except Exception as e:
            logger.error(f"Error searching PATRIC: {e}")
            return pd.DataFrame()
    
    def get_amr_phenotypes(self, genome_ids: List[str]) -> pd.DataFrame:
        """
        Get AMR phenotype data for genomes
        
        Args:
            genome_ids: List of PATRIC genome IDs
        
        Returns:
            DataFrame with AMR phenotypes
        """
        logger.info(f"Fetching AMR data for {len(genome_ids)} genomes...")
        
        all_amr_data = []
        
        for genome_id in tqdm(genome_ids, desc="Fetching AMR"):
            url = f"{self.API_BASE}/genome_amr"
            params = {
                'q': f'genome_id:{genome_id}',
                'limit': 1000
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                
                amr_data = response.json()
                all_amr_data.extend(amr_data)
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching AMR for {genome_id}: {e}")
                continue
        
        df = pd.DataFrame(all_amr_data)
        logger.info(f"Retrieved {len(df)} AMR records")
        
        return df
    
    def download_genome_fasta(self, genome_id: str) -> str:
        """Download genome FASTA file"""
        url = f"{self.API_BASE}/genome/{genome_id}/?select=fasta"
        output_file = self.genomes_dir / f"{genome_id}.fasta"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            with open(output_file, 'w') as f:
                f.write(response.text)
            
            logger.info(f"Downloaded {genome_id}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error downloading {genome_id}: {e}")
            return ""
    
    def collect_complete_dataset(self, organisms: List[str], max_per_organism: int = 500):
        """Complete collection from PATRIC"""
        all_genomes = []
        all_amr = []
        
        for organism in organisms:
            logger.info(f"\nProcessing {organism}...")
            
            # Search genomes
            genomes_df = self.search_genomes(organism, has_amr=True, limit=max_per_organism)
            
            if genomes_df.empty:
                continue
            
            # Get AMR data
            genome_ids = genomes_df['genome_id'].tolist()
            amr_df = self.get_amr_phenotypes(genome_ids[:100])  # Limit for demo
            
            all_genomes.append(genomes_df)
            all_amr.append(amr_df)
        
        # Combine
        combined_genomes = pd.concat(all_genomes, ignore_index=True)
        combined_amr = pd.concat(all_amr, ignore_index=True)
        
        # Save
        combined_genomes.to_csv(self.output_dir / 'genomes_metadata.csv', index=False)
        combined_amr.to_csv(self.output_dir / 'amr_phenotypes.csv', index=False)
        
        logger.info(f"✅ PATRIC collection complete!")
        logger.info(f"Genomes: {len(combined_genomes)}")
        logger.info(f"AMR records: {len(combined_amr)}")
        
        return combined_genomes, combined_amr


# Usage
if __name__ == "__main__":
    collector = PATRICCollector()
    
    organisms = [
        "Escherichia coli",
        "Klebsiella pneumoniae"
    ]
    
    genomes, amr = collector.collect_complete_dataset(organisms)
```

---

## ResFinder Database

```python
# src/data_collection/resfinder_collector.py
"""
Download ResFinder AMR gene database
"""
import subprocess
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ResFinderCollector:
    """Download ResFinder database"""
    
    REPO_URL = "https://bitbucket.org/genomicepidemiology/resfinder_db.git"
    
    def __init__(self, output_dir="data/external/resfinder"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def download_resfinder(self):
        """Clone ResFinder repository"""
        if (self.output_dir / '.git').exists():
            logger.info("ResFinder already downloaded. Updating...")
            subprocess.run(['git', 'pull'], cwd=self.output_dir)
        else:
            logger.info("Cloning ResFinder database...")
            subprocess.run([
                'git', 'clone', 
                self.REPO_URL,
                str(self.output_dir)
            ])
        
        logger.info("✅ ResFinder database ready!")


# Usage
if __name__ == "__main__":
    collector = ResFinderCollector()
    collector.download_resfinder()
```