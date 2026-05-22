import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import DATA_DIR, OUTPUT_DIR, CV_SPLITS
from src.data.preprocess import load_and_prepare
from src.models.train import get_base_classifiers, fit_classifiers
from src.models.ensemble import build_voting_classifier, build_stacking_classifier
from src.models.evaluate import evaluate_classifier_with_proba, plot_confusion_matrix


def save_submission(passenger_ids, predictions, output_dir: Path):
    """Save predictions to CSV submission file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    submission = passenger_ids.to_frame(name="PassengerId")
    submission["Survived"] = predictions.astype(int)
    output_path = output_dir / "submission_voting.csv"
    submission.to_csv(output_path, index=False)
    print(f"Saved submission: {output_path}")
    return output_path


def create_model_plots(eval_result, model_name, y_train):
    """Create and return confusion matrix, ROC, and probability plots for a model."""
    plots = {}
    
    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    plot_confusion_matrix(eval_result["confusion_matrix"], ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plots[f"{model_name}_confusion_matrix"] = fig
    plt.close(fig)
    
    # ROC curve
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(eval_result["fpr"], eval_result["tpr"], label=f"AUC = {eval_result['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    plots[f"{model_name}_roc_curve"] = fig
    plt.close(fig)
    
    # Probability distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(eval_result["proba"][y_train == 0], bins=25, alpha=0.5, label="Not survived", color="red", density=True)
    ax.hist(eval_result["proba"][y_train == 1], bins=25, alpha=0.5, label="Survived", color="green", density=True)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Probability Distribution - {model_name}")
    ax.legend()
    plt.tight_layout()
    plots[f"{model_name}_probability"] = fig
    plt.close(fig)
    
    return plots


def save_figures_to_disk(figures_dict, output_dir: Path):
    """Save all matplotlib figures to disk and return file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_paths = {}
    
    for fig_name, fig in figures_dict.items():
        fig_path = output_dir / f"{fig_name}.png"
        fig.savefig(fig_path, dpi=100, bbox_inches='tight')
        figures_paths[fig_name] = str(fig_path)
    
    return figures_paths


def main(args=None):
    parser = argparse.ArgumentParser(description="Titanic Ensemble Model Pipeline")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Directory with train/test CSV files")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--cv", type=int, default=CV_SPLITS, help="Cross-validation folds")
    parser.add_argument("--skip-outliers", action="store_true", help="Skip outlier removal")
    parsed = parser.parse_args(args=args)

    print("="*80)
    print("TITANIC ENSEMBLE MODEL PIPELINE")
    print("="*80)
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    X_train, y_train, X_test, passenger_ids = load_and_prepare(parsed.data_dir, drop_outliers=not parsed.skip_outliers)
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")

    # Train base models
    print("\nTraining base models...")
    classifiers = get_base_classifiers()
    fitted_models = fit_classifiers(classifiers, X_train, y_train)

    results = {
        "base_models": {},
        "ensemble": {},
        "figures": {}
    }

    # Evaluate base models
    print("\n" + "="*80)
    print("BASE MODELS EVALUATION")
    print("="*80)
    
    for model_name, model in fitted_models.items():
        print(f"\n{model_name}:")
        eval_result = evaluate_classifier_with_proba(model, X_train, y_train, cv=parsed.cv)
        print(f"  Accuracy: {eval_result['accuracy_mean']:.4f} ± {eval_result['accuracy_std']:.4f}")
        
        # Generate plots
        plots = create_model_plots(eval_result, model_name, y_train)
        results["figures"].update(plots)
        results["base_models"][model_name] = eval_result

    # Build ensemble models
    voting = build_voting_classifier(fitted_models)
    stacking = build_stacking_classifier(fitted_models)

    # Evaluate ensemble models
    print("\n" + "="*80)
    print("ENSEMBLE MODELS EVALUATION")
    print("="*80)
    
    print("\nVoting Classifier:")
    voting_result = evaluate_classifier_with_proba(voting, X_train, y_train, cv=parsed.cv)
    print(f"  Accuracy: {voting_result['accuracy_mean']:.4f} ± {voting_result['accuracy_std']:.4f}")
    plots = create_model_plots(voting_result, "voting", y_train)
    results["figures"].update(plots)
    results["ensemble"]["voting"] = voting_result
    
    print("\nStacking Classifier:")
    stacking_result = evaluate_classifier_with_proba(stacking, X_train, y_train, cv=parsed.cv)
    print(f"  Accuracy: {stacking_result['accuracy_mean']:.4f} ± {stacking_result['accuracy_std']:.4f}")
    plots = create_model_plots(stacking_result, "stacking", y_train)
    results["figures"].update(plots)
    results["ensemble"]["stacking"] = stacking_result

    # Ensemble comparison plot
    print("\nCreating ensemble comparison...")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(voting_result["fpr"], voting_result["tpr"], label=f"Voting (AUC={voting_result['roc_auc']:.3f})", linewidth=2)
    ax.plot(stacking_result["fpr"], stacking_result["tpr"], label=f"Stacking (AUC={stacking_result['roc_auc']:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison - Ensemble Models")
    ax.legend(loc="lower right")
    ax.grid(True)
    plt.tight_layout()
    results["figures"]["ensemble_comparison_roc"] = fig
    plt.close(fig)

    # Generate predictions
    print("\n" + "="*80)
    print("GENERATING PREDICTIONS")
    print("="*80)
    voting.fit(X_train, y_train)
    predictions = voting.predict(X_test)
    save_submission(passenger_ids, predictions, Path(parsed.output_dir))
    
    results["predictions"] = predictions
    results["passenger_ids"] = passenger_ids

    # Save all figures
    print("\n" + "="*80)
    print("SAVING VISUALIZATIONS")
    print("="*80)
    output_dir = Path(parsed.output_dir)
    figures_paths = save_figures_to_disk(results["figures"], output_dir / "figures")
    results["figures_paths"] = figures_paths
    print(f"Saved {len(figures_paths)} figures to {output_dir / 'figures'}/")

    # Summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"Base models: {len(results['base_models'])}")
    print(f"Ensemble models: {len(results['ensemble'])}")
    print(f"Visualizations: {len(results['figures_paths'])}")
    print(f"Predictions: {results['predictions'].shape[0]} samples")
    
    return results


if __name__ == "__main__":
    results = main()
