# src/ml/feature_extractor.py
import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
import numpy as np
from typing import List, Dict
import re

class ProteinFeatureExtractor:
    """Extract features from protein sequences using ESM-2"""
    
    def __init__(self, model_path="models/pretrained/esm2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
    def extract_proteins_from_genome(self, genome_sequence: str) -> List[str]:
        """
        Extract protein sequences from genome
        Use Prodigal or simple ORF finder
        """
        # Simple ORF finder (for demo - use Prodigal in production)
        proteins = []
        
        # Find ORFs starting with ATG and ending with stop codons
        start_codons = ['ATG']
        stop_codons = ['TAA', 'TAG', 'TGA']
        
        for i in range(len(genome_sequence) - 3):
            codon = genome_sequence[i:i+3]
            if codon in start_codons:
                # Look for stop codon
                for j in range(i+3, len(genome_sequence)-3, 3):
                    stop_codon = genome_sequence[j:j+3]
                    if stop_codon in stop_codons:
                        orf = genome_sequence[i:j+3]
                        if len(orf) >= 300:  # Minimum 100 amino acids
                            protein = self.translate_dna_to_protein(orf)
                            if protein:
                                proteins.append(protein)
                        break
        
        return proteins[:50]  # Top 50 proteins to avoid too much data
    
    def translate_dna_to_protein(self, dna_seq: str) -> str:
        """Translate DNA to protein sequence"""
        codon_table = {
            'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
            'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
            'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
            'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
            'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
            'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
            'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
            'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
            'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
            'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
            'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
            'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
            'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
        }
        
        protein = []
        for i in range(0, len(dna_seq) - 2, 3):
            codon = dna_seq[i:i+3].upper()
            if codon in codon_table:
                aa = codon_table[codon]
                if aa == '*':
                    break
                protein.append(aa)
        
        return ''.join(protein) if len(protein) > 0 else None
    
    def get_protein_embedding(self, protein_seq: str) -> np.ndarray:
        """Get ESM-2 embedding for a protein sequence"""
        # Truncate if too long (ESM-2 has max length ~1000)
        if len(protein_seq) > 1000:
            protein_seq = protein_seq[:1000]
        
        # Tokenize
        inputs = self.tokenizer(protein_seq, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling over sequence length
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings.squeeze()
    
    def extract_genome_features(self, genome_path: str) -> np.ndarray:
        """Extract features from entire genome"""
        # Load genome
        genome_seq = ""
        for record in SeqIO.parse(genome_path, "fasta"):
            genome_seq += str(record.seq)
        
        # Extract proteins
        proteins = self.extract_proteins_from_genome(genome_seq)
        print(f"Extracted {len(proteins)} proteins from genome")
        
        if len(proteins) == 0:
            return np.zeros(320)  # Return zero vector if no proteins found
        
        # Get embeddings for all proteins
        embeddings = []
        for protein in proteins[:20]:  # Top 20 proteins
            try:
                emb = self.get_protein_embedding(protein)
                embeddings.append(emb)
            except Exception as e:
                print(f"Error processing protein: {e}")
                continue
        
        if len(embeddings) == 0:
            return np.zeros(320)
        
        # Aggregate embeddings (mean pooling)
        genome_embedding = np.mean(embeddings, axis=0)
        return genome_embedding


class AMRGeneDetector:
    """Detect known AMR genes using CARD database"""
    
    def __init__(self, card_db_path="data/external/card"):
        self.card_sequences = self.load_card_database(card_db_path)
        
    def load_card_database(self, card_path):
        """Load CARD AMR gene sequences"""
        card_genes = {}
        # Load from CARD FASTA file
        fasta_path = f"{card_path}/nucleotide_fasta_protein_homolog_model.fasta"
        
        try:
            for record in SeqIO.parse(fasta_path, "fasta"):
                # Parse gene name and antibiotic class
                gene_info = self.parse_card_header(record.description)
                card_genes[record.id] = {
                    'sequence': str(record.seq),
                    'gene_name': gene_info['gene_name'],
                    'drug_class': gene_info['drug_class']
                }
        except FileNotFoundError:
            print(f"CARD database not found at {fasta_path}")
            # Return empty dict for now
            return {}
        
        print(f"Loaded {len(card_genes)} AMR genes from CARD")
        return card_genes
    
    def parse_card_header(self, header: str) -> Dict:
        """Parse CARD FASTA header"""
        # Example: "ARO:3000026|mecA [Staphylococcus aureus]"
        parts = header.split('|')
        gene_name = parts[1].split('[')[0].strip() if len(parts) > 1 else "unknown"
        
        return {
            'gene_name': gene_name,
            'drug_class': 'beta-lactam'  # Simplified for now
        }
    
    def detect_amr_genes(self, genome_sequence: str) -> List[Dict]:
        """
        Detect AMR genes in genome using sequence similarity
        In production, use BLAST or MMseqs2
        """
        detected_genes = []
        
        # Simplified: check for exact substring matches
        # In production: use BLAST or diamond
        for gene_id, gene_info in self.card_sequences.items():
            if gene_info['sequence'] in genome_sequence:
                detected_genes.append({
                    'gene_id': gene_id,
                    'gene_name': gene_info['gene_name'],
                    'drug_class': gene_info['drug_class']
                })
        
        return detected_genes


class CombinedFeatureExtractor:
    """Combine protein embeddings and gene detection"""
    
    def __init__(self):
        self.protein_extractor = ProteinFeatureExtractor()
        self.gene_detector = AMRGeneDetector()
        
    def extract_features(self, genome_path: str) -> Dict:
        """Extract all features from genome"""
        # 1. Protein embeddings (320-dim from ESM-2)
        protein_features = self.protein_extractor.extract_genome_features(genome_path)
        
        # 2. Load genome for gene detection
        genome_seq = ""
        for record in SeqIO.parse(genome_path, "fasta"):
            genome_seq += str(record.seq)
        
        # 3. AMR gene detection
        detected_genes = self.gene_detector.detect_amr_genes(genome_seq)
        
        # 4. Create gene presence/absence vector
        gene_features = self.create_gene_feature_vector(detected_genes)
        
        # 5. Combine features
        combined_features = np.concatenate([protein_features, gene_features])
        
        return {
            'features': combined_features,
            'detected_genes': detected_genes,
            'feature_dim': len(combined_features)
        }
    
    def create_gene_feature_vector(self, detected_genes: List[Dict], num_genes=50) -> np.ndarray:
        """Create binary vector for gene presence/absence"""
        # Top 50 most important AMR genes
        important_genes = [
            'mecA', 'vanA', 'blaCTX-M', 'blaKPC', 'blaNDM', 'blaOXA',
            'ermB', 'tetM', 'aac', 'aph', 'sul1', 'sul2', 'dfrA'
        ]
        
        gene_vector = np.zeros(num_genes)
        detected_names = [g['gene_name'] for g in detected_genes]
        
        for i, gene in enumerate(important_genes[:num_genes]):
            if any(gene in name for name in detected_names):
                gene_vector[i] = 1
        
        return gene_vector