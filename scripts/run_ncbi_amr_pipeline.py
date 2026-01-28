#!/usr/bin/env python3
"""Complete pipeline to annotate NCBI genomes with AMR and preprocess for modeling.

This script:
1. Checks AMRFinderPlus installation
2. Runs AMRFinderPlus on all NCBI genomes
3. Creates AMR labels from the results
4. Preprocesses data for ML modeling

Usage:
    conda activate amr_analysis  # or your conda env with amrfinderplus
    python scripts/run_ncbi_amr_pipeline.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    print("=" * 60)
    print("NCBI AMR Annotation and Preprocessing Pipeline")
    print("=" * 60)

    # Step 1: Check AMRFinderPlus installation
    print("\n[Step 1/4] Checking AMRFinderPlus installation...")
    from data_collection.amrfinder_annotator import AMRFinderAnnotator

    annotator = AMRFinderAnnotator()

    if not annotator.check_installation():
        print("\nERROR: AMRFinderPlus is not installed!")
        print("\nTo install:")
        print("  1. Install Miniconda: https://docs.conda.io/en/latest/miniconda.html")
        print("  2. Run: conda install -c bioconda -c conda-forge ncbi-amrfinderplus")
        print("  3. Update database: amrfinder -u")
        print("\nOr run: bash scripts/setup_amrfinder.sh")
        sys.exit(1)

    print("AMRFinderPlus is installed!")

    # Step 2: Run AMRFinderPlus on all genomes
    print("\n[Step 2/4] Running AMRFinderPlus on NCBI genomes...")
    print("This may take a while depending on the number of genomes...")

    results = annotator.run_on_all_genomes()

    if len(results) == 0:
        print("\nERROR: No AMR genes found in any genome.")
        print("This could mean:")
        print("  - Genomes don't contain detectable AMR genes")
        print("  - Genome files are corrupted or empty")
        print("  - AMRFinderPlus database needs updating (run: amrfinder -u)")
        sys.exit(1)

    print(f"Found {len(results)} AMR annotations!")

    # Step 3: Create AMR labels
    print("\n[Step 3/4] Creating AMR labels for ML...")
    labels_df, drug_mapping = annotator.create_amr_labels(min_samples_per_drug=5)

    if len(labels_df) == 0 or len(drug_mapping) == 0:
        print("\nWARNING: No drug classes with enough samples for training.")
        print("Try lowering min_samples_per_drug or adding more genomes.")
    else:
        print(f"Created labels for {len(labels_df)} samples")
        print(f"Drug classes: {list(drug_mapping.keys())}")

    # Step 4: Preprocess for modeling
    print("\n[Step 4/4] Preprocessing data for ML modeling...")
    from preprocessing.ncbi_preprocessor import NCBIPreprocessor

    preprocessor = NCBIPreprocessor()
    preprocessor.load_data()

    try:
        data = preprocessor.prepare_amr_prediction_data(
            amr_labels_file="data/raw/ncbi/amrfinder_results/amr_labels.csv",
            k=6,
            max_features=500,
            min_samples_per_drug=5,
        )
        preprocessor.save_processed_data(data, prefix="ncbi_amr")

        print("\n" + "=" * 60)
        print("Pipeline Complete!")
        print("=" * 60)
        print(f"\nProcessed data saved to: data/processed/ncbi/")
        print(f"\nDataset summary:")
        print(f"  Task: Multi-label AMR prediction")
        print(f"  Features: {data['X_train'].shape[1]} k-mer features")
        print(f"  Drug classes: {len(data['class_names'])}")
        print(f"  Training samples: {data['X_train'].shape[0]}")
        print(f"  Validation samples: {data['X_val'].shape[0]}")
        print(f"  Test samples: {data['X_test'].shape[0]}")
        print(f"\nDrug classes: {data['class_names']}")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nMake sure AMRFinderPlus completed successfully in Step 2.")
        sys.exit(1)
    except ValueError as e:
        print(f"\nERROR: {e}")
        print("\nTry:")
        print("  - Lowering min_samples_per_drug")
        print("  - Adding more genome files")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Next steps:")
    print("  1. Train a model using the processed data")
    print("  2. Run: python -m src.ml.amr_classifier")
    print("=" * 60)


if __name__ == "__main__":
    main()
