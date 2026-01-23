# src/preprocessing/data_loader.py
import pandas as pd
from Bio import SeqIO
from pathlib import Path
import json

class AMRDataLoader:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        
    def load_ncbi_data(self):
        """Load NCBI pathogen detection data"""
        # NCBI provides metadata.tsv with AMR phenotypes
        metadata = pd.read_csv(self.data_dir / "ncbi_metadata.tsv", sep="\t")
        
        # Filter relevant columns
        df = metadata[[
            'BioSample', 'organism', 'AMR_genotypes', 
            'computed_serotype', 'isolation_source'
        ]]
        
        # Parse AMR phenotypes
        amr_data = []
        for idx, row in df.iterrows():
            if pd.notna(row['AMR_genotypes']):
                # Parse format: "AMINOGLYCOSIDE=RESISTANT;BETA-LACTAM=SUSCEPTIBLE"
                phenotypes = self.parse_amr_phenotypes(row['AMR_genotypes'])
                amr_data.append({
                    'sample_id': row['BioSample'],
                    'species': row['organism'],
                    'phenotypes': phenotypes,
                    'genome_path': f"genomes/{row['BioSample']}.fasta"
                })
        
        return pd.DataFrame(amr_data)
    
    def parse_amr_phenotypes(self, amr_string):
        """Parse AMR phenotype string"""
        phenotypes = {}
        if pd.isna(amr_string):
            return phenotypes
            
        pairs = amr_string.split(';')
        for pair in pairs:
            if '=' in pair:
                drug_class, status = pair.split('=')
                phenotypes[drug_class.strip()] = status.strip()
        return phenotypes
    
    def load_genome_sequence(self, fasta_path):
        """Load genome from FASTA file"""
        sequences = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            sequences.append(str(record.seq))
        return "".join(sequences)
    
    def create_training_dataset(self):
        """Create final training dataset"""
        # Load all data sources
        ncbi_data = self.load_ncbi_data()
        
        # Map drug classes to specific antibiotics
        drug_mapping = {
            'AMINOGLYCOSIDE': ['Gentamicin', 'Amikacin', 'Tobramycin'],
            'BETA-LACTAM': ['Amoxicillin', 'Ceftriaxone', 'Meropenem'],
            'FLUOROQUINOLONE': ['Ciprofloxacin', 'Levofloxacin'],
            'MACROLIDE': ['Azithromycin'],
            'TETRACYCLINE': ['Doxycycline'],
            'SULFONAMIDE': ['Trimethoprim-sulfamethoxazole']
        }
        
        # Expand to individual antibiotics
        expanded_data = []
        for idx, row in ncbi_data.iterrows():
            for drug_class, status in row['phenotypes'].items():
                if drug_class in drug_mapping:
                    for antibiotic in drug_mapping[drug_class]:
                        expanded_data.append({
                            'sample_id': row['sample_id'],
                            'species': row['species'],
                            'antibiotic': antibiotic,
                            'resistance': 1 if status == 'RESISTANT' else 0,
                            'genome_path': row['genome_path']
                        })
        
        df = pd.DataFrame(expanded_data)
        df.to_csv('data/processed/training_data.csv', index=False)
        print(f"Created training dataset with {len(df)} samples")
        return df

# Run data preprocessing
if __name__ == "__main__":
    loader = AMRDataLoader()
    df = loader.create_training_dataset()
    print(df.head())
    print(f"\nDataset statistics:")
    print(df.groupby(['species', 'antibiotic', 'resistance']).size())