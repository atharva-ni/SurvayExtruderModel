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
  # Extract papers for an author from Semantic Scholar (several IDs can be merged):
  python main.py extract --author 1713586,2266084696 --output data/niyato.csv

  # Extract a merged author profile from OpenAlex (needs OPENALEX_API_KEY for large profiles):
  python main.py extract --source openalex --name "Dusit Niyato" --output data/proauthor_openalex/niyato.csv

  # Classify extracted papers (filter out surveys and compute indices):
  python main.py classify --input data/proauthor_s2/niyato.csv

  # Train the classifier (also fits the threshold and learned hybrid):
  python main.py train --dataset data/real_dataset.csv

  # Tune hyperparameters using Optuna (train/validation splits only):
  python main.py train --dataset data/real_dataset.csv --tune --trials 3

  # Regenerate Table II and Table IV:
  python main.py evaluate --baseline-model ./distilbert_survey_model_synthetic

  # Build a real labeled training dataset from OpenAlex:
  python main.py build-dataset --output data/real_dataset.csv

  # Sample author-profile papers for a hand-labeled test set:
  python main.py make-eval-set --input "data/proauthor/*.csv"
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available sub-commands")

    # ---- Extract Subparser ----
    parser_extract = subparsers.add_parser("extract", help="Extract an author's publications (Semantic Scholar or OpenAlex)")
    parser_extract.add_argument("--source", type=str, default="semantic", choices=["semantic", "openalex"],
                                help="semantic (default) or openalex (merged author profiles)")
    parser_extract.add_argument("--author", type=str, help="Author ID; for Semantic Scholar several IDs may be comma-separated")
    parser_extract.add_argument("--name", type=str, help="OpenAlex only: search the author by name")
    parser_extract.add_argument("--output", type=str, help="Destination CSV path")
    parser_extract.add_argument("--api-key", type=str, help="Optional API key override")
    parser_extract.add_argument("--add-types", type=str, help="Semantic Scholar only: add publication types to an existing CSV")

    # ---- Classify Subparser ----
    parser_classify = subparsers.add_parser("classify", help="Filter out survey papers and calculate academic indices")
    parser_classify.add_argument("--input", type=str, required=True, help="Input CSV file of publications (from the extract command)")
    parser_classify.add_argument("--output", type=str, help="Output CSV for original research papers (default: data/Non-Survey-Papers.csv)")
    parser_classify.add_argument("--surveys", type=str, help="Output CSV for surveys and non-papers (default: data/Survey-Papers.csv)")
    parser_classify.add_argument("--model", type=str, default="./distilbert_survey_model", help="Path to fine-tuned model (default: ./distilbert_survey_model)")
    parser_classify.add_argument("--batch-size", type=int, default=32, help="Batch size for model inference (default: 32)")
    parser_classify.add_argument("--threshold", type=float, default=None, help="DistilBERT survey-probability threshold (default: tuned value saved with the model)")
    parser_classify.add_argument("--mode", type=str, default="learned", choices=["learned", "or", "model", "keyword"], help="learned hybrid (default), OR rule from the paper, DistilBERT only, or keywords only")
    parser_classify.add_argument("--exclude-magazine-overviews", action="store_true", help="Also exclude magazine articles flagged by the classifier that do not present themselves as surveys")

    # ---- Train Subparser ----
    parser_train = subparsers.add_parser("train", help="Train DistilBERT classifier or perform hyperparameter tuning")
    parser_train.add_argument("--dataset", type=str, default="data/real_dataset.csv", help="Path to training dataset CSV (default: data/real_dataset.csv)")
    parser_train.add_argument("--output-dir", type=str, default="./distilbert_survey_model", help="Directory to save model weights")
    parser_train.add_argument("--epochs", type=int, default=3, help="Number of training epochs (default: 3)")
    parser_train.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser_train.add_argument("--lr", type=float, default=2e-5, help="Learning rate (default: 2e-5)")
    parser_train.add_argument("--tune", action="store_true", help="Perform hyperparameter search with Optuna instead of training")
    parser_train.add_argument("--trials", type=int, default=5, help="Number of tuning trials for Optuna (default: 5)")

    # ---- Evaluate Subparser ----
    parser_evaluate = subparsers.add_parser("evaluate", help="Regenerate Table II (classification) and Table IV (author impact)")
    parser_evaluate.add_argument("--model", type=str, default="./distilbert_survey_model", help="Trained model directory")
    parser_evaluate.add_argument("--eval-set", type=str, default="data/eval_to_label.csv", help="Hand-labeled evaluation CSV")
    parser_evaluate.add_argument("--authors", type=str, default="data/proauthor/*.csv", help="Glob of author profile CSVs for Table IV")
    parser_evaluate.add_argument("--baseline-model", type=str, default=None, help="Optional older model to compare against")
    parser_evaluate.add_argument("--report-dir", type=str, default="reports", help="Where to write evaluation.md/json")

    # ---- Build Dataset Subparser ----
    parser_build = subparsers.add_parser("build-dataset", help="Build a real survey/non-survey training dataset from OpenAlex")
    parser_build.add_argument("--output", type=str, default="data/real_dataset.csv", help="Output dataset CSV (default: data/real_dataset.csv)")
    parser_build.add_argument("--hard-cases", type=str, default="data/hard_cases_to_label.csv", help="Output CSV for ambiguous papers to label by hand")
    parser_build.add_argument("--years", type=str, default="2010-2025", help="Publication year range (default: 2010-2025)")
    parser_build.add_argument("--pool-per-venue", type=int, default=1200, help="Research papers sampled per venue (default: 1200)")
    parser_build.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser_build.add_argument("--exclude", type=str, default="data/eval_to_label.csv", help="Test-set CSV whose papers are kept out of the dataset")

    # ---- Make Eval Set Subparser ----
    parser_eval = subparsers.add_parser("make-eval-set", help="Sample author-profile papers for a hand-labeled test set")
    parser_eval.add_argument("--input", type=str, default="data/proauthor/*.csv", help="Glob of author profile CSVs")
    parser_eval.add_argument("--training", type=str, default="data/real_dataset.csv", help="Training dataset whose papers are excluded")
    parser_eval.add_argument("--output", type=str, default="data/eval_to_label.csv", help="Output CSV to label (default: data/eval_to_label.csv)")
    parser_eval.add_argument("--n-title-kw", type=int, default=120, help="Papers with survey terms in the title")
    parser_eval.add_argument("--n-abstract-cue", type=int, default=120, help="Papers with survey phrasing only in the abstract")
    parser_eval.add_argument("--n-random", type=int, default=160, help="Other papers")
    parser_eval.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "extract":
        if args.source == "openalex":
            from extract_openalex import run_extraction
            run_extraction(
                author_id=args.author,
                name=args.name,
                output_path=args.output,
                api_key=args.api_key
            )
        elif args.add_types:
            from extract_semantic import add_publication_types
            add_publication_types(args.add_types, api_key=args.api_key)
        else:
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
        output_csv = args.output or os.path.join("data", "Non-Survey-Papers.csv")
        survey_csv = args.surveys or os.path.join("data", "Survey-Papers.csv")
        
        print(f"🎬 Starting classification pipeline with input: '{input_csv}'")
        run_classification_pipeline(
            input_csv=input_csv,
            output_csv=output_csv,
            survey_csv=survey_csv,
            model_path=args.model,
            batch_size=args.batch_size,
            threshold=args.threshold,
            mode=args.mode,
            exclude_magazine_overviews=args.exclude_magazine_overviews
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

    elif args.command == "evaluate":
        from evaluate import run_evaluation
        run_evaluation(
            model_path=args.model,
            eval_set=args.eval_set,
            authors_glob=args.authors,
            baseline_model=args.baseline_model,
            report_dir=args.report_dir
        )

    elif args.command == "build-dataset":
        from build_dataset import run_build
        run_build(
            output_csv=args.output,
            hard_cases_csv=args.hard_cases,
            years=args.years,
            pool_per_venue=args.pool_per_venue,
            seed=args.seed,
            exclude_csv=args.exclude
        )

    elif args.command == "make-eval-set":
        from make_eval_set import make_eval_set
        make_eval_set(
            input_glob=args.input,
            training_csv=args.training,
            output_csv=args.output,
            n_title_kw=args.n_title_kw,
            n_abstract_cue=args.n_abstract_cue,
            n_random=args.n_random,
            seed=args.seed
        )

if __name__ == "__main__":
    main()
