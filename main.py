#!/usr/bin/env python3
import argparse
import sys
import os

# Ensure src directory is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(
        description="Academic Paper Classification & Metrics Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract papers for an author from Semantic Scholar:
  python main.py extract --author 144019071 --output data/prof3.csv

  # Classify extracted papers (filter out surveys and compute indices):
  python main.py classify --input data/prof3.csv

  # Train the classifier:
  python main.py train --dataset data/dataset.csv

  # Tune hyperparameters using Optuna:
  python main.py train --dataset data/dataset.csv --tune --trials 3
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # ---- Extract Subparser ----
    parser_extract = subparsers.add_parser("extract", help="Extract publications for an author ID from Semantic Scholar")
    parser_extract.add_argument("--author", type=str, help="Semantic Scholar Author ID (e.g. 144019071)")
    parser_extract.add_argument("--output", type=str, help="Destination CSV path (default: data/prof3.csv)")
    parser_extract.add_argument("--api-key", type=str, help="Optional Semantic Scholar API key override")

    # ---- Classify Subparser ----
    parser_classify = subparsers.add_parser("classify", help="Filter out survey papers and calculate academic indices")
    parser_classify.add_argument("--input", type=str, help="Input CSV file of publications (default: first CSV found in data/)")
    parser_classify.add_argument("--output", type=str, help="Output CSV for original research papers (default: data/Non-Survey-Papers.csv)")
    parser_classify.add_argument("--surveys", type=str, help="Output CSV for excluded survey papers (default: data/Survey-Papers.csv)")
    parser_classify.add_argument("--model", type=str, default="./distilbert_survey_model", help="Path to fine-tuned model (default: ./distilbert_survey_model)")
    parser_classify.add_argument("--batch-size", type=int, default=32, help="Batch size for model inference (default: 32)")
    parser_classify.add_argument("--threshold", type=float, default=0.8, help="Classification probability threshold (default: 0.8)")

    # ---- Train Subparser ----
    parser_train = subparsers.add_parser("train", help="Train DistilBERT classifier or perform hyperparameter tuning")
    parser_train.add_argument("--dataset", type=str, default="data/dataset.csv", help="Path to training dataset CSV (default: data/dataset.csv)")
    parser_train.add_argument("--output-dir", type=str, default="./distilbert_survey_model", help="Directory to save model weights")
    parser_train.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser_train.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser_train.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser_train.add_argument("--tune", action="store_true", help="Perform hyperparameter search with Optuna instead of training")
    parser_train.add_argument("--trials", type=int, default=5, help="Number of tuning trials for Optuna (default: 5)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "extract":
        from extract_semantic import run_extraction
        run_extraction(
            author_id=args.author,
            output_path=args.output,
            api_key=args.api_key
        )

    elif args.command == "classify":
        from classifier import run_classification_pipeline
        # Handle default files in 'data/' directory
        input_csv = args.input
        if not input_csv:
            import glob
            csv_files = glob.glob(os.path.join("data", "*.csv"))
            # Filter out target outputs if any
            csv_files = [c for c in csv_files if "Non-Survey-Papers" not in c and "Survey-Papers" not in c and "dataset" not in c]
            if csv_files:
                input_csv = csv_files[0]
            else:
                input_csv = os.path.join("data", "prof3.csv")

        output_csv = args.output or os.path.join("data", "Non-Survey-Papers.csv")
        survey_csv = args.surveys or os.path.join("data", "Survey-Papers.csv")
        
        print(f"🎬 Starting classification pipeline with input: '{input_csv}'")
        run_classification_pipeline(
            input_csv=input_csv,
            output_csv=output_csv,
            survey_csv=survey_csv,
            model_path=args.model,
            batch_size=args.batch_size,
            threshold=args.threshold
        )

    elif args.command == "train":
        if args.tune:
            from train import run_tuning
            run_tuning(dataset_path=args.dataset, n_trials=args.trials)
        else:
            from train import run_training
            run_training(
                dataset_path=args.dataset,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr
            )

if __name__ == "__main__":
    main()
