#!/usr/bin/env python3
"""
Main script for downloading AMR data from all sources.

Usage:
    python scripts/download_data.py --all
    python scripts/download_data.py --ncbi --email your@email.com
    python scripts/download_data.py --card
    python scripts/download_data.py --patric
    python scripts/download_data.py --resfinder
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection import (
    CARDCollector,
    CollectionConfig,
    NCBICollector,
    PATRICCollector,
    ResFinderCollector,
)


def setup_logging(log_dir: Path) -> None:
    """Configure logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "download.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download AMR data from public databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Source selection
    parser.add_argument("--all", action="store_true", help="Download from all sources")
    parser.add_argument("--ncbi", action="store_true", help="Download from NCBI")
    parser.add_argument("--card", action="store_true", help="Download CARD database")
    parser.add_argument("--patric", action="store_true", help="Download from PATRIC")
    parser.add_argument("--resfinder", action="store_true", help="Download ResFinder")

    # Configuration
    parser.add_argument("--email", type=str, help="Email for NCBI (required for NCBI)")
    parser.add_argument("--api-key", type=str, help="NCBI API key (optional)")
    parser.add_argument(
        "--max-samples", type=int, default=500, help="Max samples per organism"
    )
    parser.add_argument(
        "--max-downloads", type=int, default=200, help="Max genome downloads"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Base data directory"
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([args.all, args.ncbi, args.card, args.patric, args.resfinder]):
        parser.print_help()
        print("\nError: Select at least one data source (--all, --ncbi, --card, --patric, --resfinder)")
        sys.exit(1)

    if (args.ncbi or args.all) and not args.email:
        print("Error: --email is required for NCBI downloads")
        print("  Get an API key at: https://www.ncbi.nlm.nih.gov/account/")
        sys.exit(1)

    # Setup
    setup_logging(project_root / "logs")
    logger = logging.getLogger("download_data")

    config = CollectionConfig(
        base_dir=args.data_dir,
        ncbi_email=args.email or "",
        ncbi_api_key=args.api_key,
        max_samples_per_organism=args.max_samples,
        max_downloads=args.max_downloads,
    )
    config.ensure_directories()

    logger.info("=" * 60)
    logger.info("AMR Data Collection Pipeline")
    logger.info("=" * 60)
    logger.info(f"Data directory: {config.base_dir.absolute()}")
    logger.info(f"Target organisms: {', '.join(config.target_organisms)}")

    results = {}

    # CARD Database
    if args.card or args.all:
        logger.info("\n" + "=" * 60)
        logger.info("Downloading CARD database...")
        logger.info("=" * 60)
        try:
            collector = CARDCollector(config)
            results["card"] = collector.collect()
            logger.info(f"CARD: {len(results['card'])} records")
        except Exception as e:
            logger.error(f"CARD collection failed: {e}")

    # ResFinder
    if args.resfinder or args.all:
        logger.info("\n" + "=" * 60)
        logger.info("Downloading ResFinder database...")
        logger.info("=" * 60)
        try:
            collector = ResFinderCollector(config)
            results["resfinder"] = collector.collect()
            logger.info(f"ResFinder: {len(results['resfinder'])} genes")
        except Exception as e:
            logger.error(f"ResFinder collection failed: {e}")

    # PATRIC
    if args.patric or args.all:
        logger.info("\n" + "=" * 60)
        logger.info("Downloading from PATRIC...")
        logger.info("=" * 60)
        try:
            collector = PATRICCollector(config)
            results["patric"] = collector.collect()
            logger.info(f"PATRIC: {len(results['patric'])} genomes")
        except Exception as e:
            logger.error(f"PATRIC collection failed: {e}")

    # NCBI (most time-consuming, do last)
    if args.ncbi or args.all:
        logger.info("\n" + "=" * 60)
        logger.info("Downloading from NCBI Pathogen Detection...")
        logger.info("=" * 60)
        try:
            collector = NCBICollector(config)
            results["ncbi"] = collector.collect()
            logger.info(f"NCBI: {len(results['ncbi'])} samples")
        except Exception as e:
            logger.error(f"NCBI collection failed: {e}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("COLLECTION SUMMARY")
    logger.info("=" * 60)
    for source, df in results.items():
        logger.info(f"  {source.upper()}: {len(df)} records")
    logger.info(f"\nData saved to: {config.base_dir.absolute()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
