#!/usr/bin/env python3
"""
Fix NCBI data: combine metadata files and download genomes.

Usage:
    python scripts/fix_ncbi_data.py --fix-metadata
    python scripts/fix_ncbi_data.py --download-genomes --max-downloads 500
    python scripts/fix_ncbi_data.py --download-missing-organisms --email your@email.com
    python scripts/fix_ncbi_data.py --all --email your@email.com
"""

import argparse
import gzip
import logging
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

NCBI_DIR = project_root / "data" / "raw" / "ncbi"
METADATA_DIR = NCBI_DIR / "metadata"
GENOMES_DIR = NCBI_DIR / "genomes"


def fix_metadata():
    """Combine all individual metadata files into complete_metadata.csv."""
    logger.info("Fixing metadata files...")

    all_dfs = []
    for f in sorted(METADATA_DIR.glob("*.csv")):
        df = pd.read_csv(f)
        logger.info(f"  Loaded {f.stem}: {len(df)} records")
        all_dfs.append(df)

    if not all_dfs:
        logger.error("No metadata files found!")
        return None

    combined = pd.concat(all_dfs, ignore_index=True)

    # Save combined metadata
    output_file = NCBI_DIR / "complete_metadata.csv"
    combined.to_csv(output_file, index=False)
    logger.info(f"Saved combined metadata: {len(combined)} records to {output_file}")

    # Print summary
    print("\n=== Metadata Summary ===")
    print(f"Total records: {len(combined)}")
    print(f"Organisms: {combined['organism_query'].nunique()}")
    print(f"Records with assemblies: {combined['assembly_accession'].notna().sum()}")
    print(f"Records with FTP paths: {(combined['ftp_path_refseq'].notna() | combined['ftp_path_genbank'].notna()).sum()}")

    print("\nPer-organism breakdown:")
    for org in combined['organism_query'].unique():
        org_df = combined[combined['organism_query'] == org]
        assemblies = org_df['assembly_accession'].notna().sum()
        print(f"  {org}: {len(org_df)} samples, {assemblies} with assemblies")

    return combined


def download_genomes(max_downloads: int = 500, force: bool = False):
    """Download genomes for samples with assemblies."""
    GENOMES_DIR.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata_file = NCBI_DIR / "complete_metadata.csv"
    if not metadata_file.exists():
        logger.info("complete_metadata.csv not found, fixing first...")
        df = fix_metadata()
    else:
        df = pd.read_csv(metadata_file)

    # Filter to samples with FTP paths
    df_with_paths = df[
        (df['ftp_path_refseq'].notna() & (df['ftp_path_refseq'] != '')) |
        (df['ftp_path_genbank'].notna() & (df['ftp_path_genbank'] != ''))
    ].copy()

    logger.info(f"Found {len(df_with_paths)} samples with download paths")

    if len(df_with_paths) == 0:
        logger.warning("No samples have FTP paths for genome download!")
        return

    downloaded = 0
    skipped = 0
    failed = 0

    for _, row in tqdm(df_with_paths.iterrows(), total=min(len(df_with_paths), max_downloads), desc="Downloading genomes"):
        if downloaded >= max_downloads:
            break

        biosample_id = row['biosample_id']
        output_file = GENOMES_DIR / f"{biosample_id}.fna.gz"

        # Skip if already downloaded
        if output_file.exists() and not force:
            skipped += 1
            downloaded += 1
            continue

        # Get FTP path
        ftp_path = row['ftp_path_refseq'] if pd.notna(row['ftp_path_refseq']) and row['ftp_path_refseq'] else row['ftp_path_genbank']

        if not ftp_path or pd.isna(ftp_path):
            continue

        try:
            # Construct download URL
            assembly_name = ftp_path.split("/")[-1]
            fasta_url = f"{ftp_path}/{assembly_name}_genomic.fna.gz"
            fasta_url = fasta_url.replace("ftp://", "https://")

            # Download
            response = requests.get(fasta_url, stream=True, timeout=300)
            response.raise_for_status()

            with open(output_file, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            downloaded += 1
            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            logger.debug(f"Failed to download {biosample_id}: {e}")
            failed += 1

    print(f"\n=== Download Summary ===")
    print(f"Downloaded: {downloaded - skipped}")
    print(f"Skipped (already exists): {skipped}")
    print(f"Failed: {failed}")
    print(f"Total genomes in folder: {len(list(GENOMES_DIR.glob('*.fna.gz')))}")


def download_missing_organisms(email: str, api_key: str = None, max_samples: int = 500):
    """Download metadata and assemblies for missing organisms."""
    from src.data_collection import CollectionConfig, NCBICollector

    # Get list of downloaded organisms
    downloaded_orgs = set()
    for f in METADATA_DIR.glob("*.csv"):
        org_name = f.stem.replace("_metadata", "").replace("_", " ")
        downloaded_orgs.add(org_name)

    config = CollectionConfig(
        ncbi_email=email,
        ncbi_api_key=api_key,
        max_samples_per_organism=max_samples,
        max_downloads=0,  # Don't download genomes yet
    )

    # Check which organisms are missing
    missing = [org for org in config.target_organisms if org not in downloaded_orgs]

    if not missing:
        logger.info("All target organisms already have metadata!")
        return

    logger.info(f"Missing organisms ({len(missing)}):")
    for org in missing:
        logger.info(f"  - {org}")

    # Temporarily set target organisms to only missing ones
    original_targets = config.target_organisms.copy()
    config.target_organisms = missing

    try:
        collector = NCBICollector(config)
        collector.collect()
    finally:
        config.target_organisms = original_targets

    logger.info("Done downloading missing organisms!")


def show_status():
    """Show current status of NCBI data."""
    print("=== NCBI Data Status ===\n")

    # Metadata files
    print("Metadata files:")
    total_records = 0
    total_assemblies = 0

    for f in sorted(METADATA_DIR.glob("*.csv")):
        df = pd.read_csv(f)
        assemblies = df['assembly_accession'].notna().sum()
        total_records += len(df)
        total_assemblies += assemblies
        print(f"  {f.stem}: {len(df)} records, {assemblies} assemblies")

    print(f"\nTotal: {total_records} records, {total_assemblies} with assemblies")

    # Genomes
    genome_files = list(GENOMES_DIR.glob("*.fna.gz"))
    print(f"\nGenome files downloaded: {len(genome_files)}")

    if genome_files:
        total_size = sum(f.stat().st_size for f in genome_files) / (1024 * 1024)
        print(f"Total size: {total_size:.1f} MB")

    # Missing organisms
    from src.data_collection import CollectionConfig
    config = CollectionConfig()

    downloaded_orgs = set()
    for f in METADATA_DIR.glob("*.csv"):
        org_name = f.stem.replace("_metadata", "").replace("_", " ")
        downloaded_orgs.add(org_name)

    missing = [org for org in config.target_organisms if org not in downloaded_orgs]

    print(f"\nTarget organisms: {len(config.target_organisms)}")
    print(f"Downloaded: {len(downloaded_orgs)}")
    print(f"Missing: {len(missing)}")

    if missing:
        print("\nMissing organisms:")
        for org in missing:
            print(f"  - {org}")


def main():
    parser = argparse.ArgumentParser(description="Fix NCBI data and download genomes")

    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--fix-metadata", action="store_true", help="Combine metadata files")
    parser.add_argument("--download-genomes", action="store_true", help="Download genome files")
    parser.add_argument("--download-missing-organisms", action="store_true", help="Download missing organisms")
    parser.add_argument("--all", action="store_true", help="Run all fixes")

    parser.add_argument("--email", type=str, help="Email for NCBI (required for --download-missing-organisms)")
    parser.add_argument("--api-key", type=str, help="NCBI API key (optional)")
    parser.add_argument("--max-downloads", type=int, default=500, help="Max genomes to download")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples per organism")
    parser.add_argument("--force", action="store_true", help="Force re-download of existing files")

    args = parser.parse_args()

    if not any([args.status, args.fix_metadata, args.download_genomes, args.download_missing_organisms, args.all]):
        parser.print_help()
        print("\nRun with --status to see current data status")
        sys.exit(1)

    if args.status:
        show_status()
        return

    if args.fix_metadata or args.all:
        fix_metadata()

    if args.download_genomes or args.all:
        download_genomes(max_downloads=args.max_downloads, force=args.force)

    if args.download_missing_organisms or args.all:
        if not args.email:
            print("Error: --email is required for downloading missing organisms")
            sys.exit(1)
        download_missing_organisms(args.email, args.api_key, args.max_samples)


if __name__ == "__main__":
    main()
