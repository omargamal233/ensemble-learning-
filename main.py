import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

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
    return output_path


def create_model_plots(eval_result, model_name, y_train):
    """Create and return confusion matrix, ROC, and probability plots for a model."""
    plots = {}

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_confusion_matrix(eval_result["confusion_matrix"], ax=ax)
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plots[f"{model_name}_confusion_matrix"] = fig

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

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(eval_result["proba"][y_train == 0], bins=25, alpha=0.5, label="Not survived", color="red", density=True)
    ax.hist(eval_result["proba"][y_train == 1], bins=25, alpha=0.5, label="Survived", color="green", density=True)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    ax.set_title(f"Probability Distribution - {model_name}")
    ax.legend()
    plt.tight_layout()
    plots[f"{model_name}_probability"] = fig

    return plots


def save_figures_to_disk(figures_dict, output_dir: Path):
    """Save all matplotlib figures to disk and return file paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_paths = {}

    for fig_name, fig in figures_dict.items():
        fig_path = output_dir / f"{fig_name}.png"
        fig.savefig(fig_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        figures_paths[fig_name] = str(fig_path)

    return figures_paths


def is_streamlit_environment() -> bool:
    return os.environ.get("STREAMLIT_SERVER_RUNNING") == "1"


def run_pipeline(data_dir, output_dir, cv, skip_outliers, save_outputs=False):
    X_train, y_train, X_test, passenger_ids = load_and_prepare(data_dir, drop_outliers=not skip_outliers)

    classifiers = get_base_classifiers()
    fitted_models = fit_classifiers(classifiers, X_train, y_train)

    results = {
        "data_shapes": {
            "train_shape": X_train.shape,
            "test_shape": X_test.shape,
        },
        "base_models": {},
        "ensemble": {},
        "figures": {},
    }

    for model_name, model in fitted_models.items():
        eval_result = evaluate_classifier_with_proba(model, X_train, y_train, cv=cv)
        plots = create_model_plots(eval_result, model_name, y_train)
        results["figures"].update(plots)
        results["base_models"][model_name] = eval_result

    voting = build_voting_classifier(fitted_models)
    stacking = build_stacking_classifier(fitted_models)

    voting_result = evaluate_classifier_with_proba(voting, X_train, y_train, cv=cv)
    plots = create_model_plots(voting_result, "voting", y_train)
    results["figures"].update(plots)
    results["ensemble"]["voting"] = voting_result

    stacking_result = evaluate_classifier_with_proba(stacking, X_train, y_train, cv=cv)
    plots = create_model_plots(stacking_result, "stacking", y_train)
    results["figures"].update(plots)
    results["ensemble"]["stacking"] = stacking_result

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

    voting.fit(X_train, y_train)
    predictions = voting.predict(X_test)

    results["predictions"] = predictions
    results["passenger_ids"] = passenger_ids

    submission_df = passenger_ids.to_frame(name="PassengerId")
    submission_df["Survived"] = predictions.astype(int)
    results["submission_df"] = submission_df

    if save_outputs:
        out_path = Path(output_dir)
        results["submission_path"] = str(save_submission(passenger_ids, predictions, out_path))
        results["figures_paths"] = save_figures_to_disk(results["figures"], out_path / "figures")
    else:
        results["submission_path"] = None
        results["figures_paths"] = {}

    return results


def display_results(results):
    st.subheader("Dataset Shapes")
    st.write({
        "Training samples": results["data_shapes"]["train_shape"],
        "Test samples": results["data_shapes"]["test_shape"],
    })

    if results["submission_path"]:
        st.success(f"Submission saved to: {results['submission_path']}")

    st.subheader("Base Model Metrics")
    base_metrics = []
    for name, eval_result in results["base_models"].items():
        base_metrics.append({
            "model": name,
            "accuracy_mean": eval_result["accuracy_mean"],
            "accuracy_std": eval_result["accuracy_std"],
            "roc_auc": eval_result["roc_auc"],
        })
    st.table(pd.DataFrame(base_metrics).set_index("model"))

    for model_name, eval_result in results["base_models"].items():
        with st.expander(f"{model_name} details", expanded=False):
            st.write("**Classification report**")
            st.text(eval_result["report"])
            fig = results["figures"][f"{model_name}_confusion_matrix"]
            st.pyplot(fig)
            plt.close(fig)
            fig = results["figures"][f"{model_name}_roc_curve"]
            st.pyplot(fig)
            plt.close(fig)
            fig = results["figures"][f"{model_name}_probability"]
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Ensemble Metrics")
    ensemble_metrics = []
    for name, eval_result in results["ensemble"].items():
        ensemble_metrics.append({
            "model": name,
            "accuracy_mean": eval_result["accuracy_mean"],
            "accuracy_std": eval_result["accuracy_std"],
            "roc_auc": eval_result["roc_auc"],
        })
    st.table(pd.DataFrame(ensemble_metrics).set_index("model"))

    for model_name in ["voting", "stacking"]:
        eval_result = results["ensemble"][model_name]
        with st.expander(f"{model_name} ensemble details", expanded=False):
            st.write("**Classification report**")
            st.text(eval_result["report"])
            fig = results["figures"][f"{model_name}_confusion_matrix"]
            st.pyplot(fig)
            plt.close(fig)
            fig = results["figures"][f"{model_name}_roc_curve"]
            st.pyplot(fig)
            plt.close(fig)
            fig = results["figures"][f"{model_name}_probability"]
            st.pyplot(fig)
            plt.close(fig)

    st.subheader("Ensemble ROC Comparison")
    fig = results["figures"]["ensemble_comparison_roc"]
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Submission Download")
    csv_bytes = results["submission_df"].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download submission CSV",
        data=csv_bytes,
        file_name="submission_voting.csv",
        mime="text/csv",
    )


def main():
    st.set_page_config(page_title="Titanic Ensemble Pipeline", layout="wide")
    st.title("Titanic Ensemble Pipeline")
    st.write("Use this app to train and evaluate the Titanic ensemble prediction pipeline.")

    st.sidebar.header("Pipeline configuration")
    data_dir = st.sidebar.text_input("Data directory", value=str(DATA_DIR))
    output_dir = st.sidebar.text_input("Output directory", value=str(OUTPUT_DIR))
    cv = st.sidebar.slider("Cross-validation folds", min_value=2, max_value=10, value=CV_SPLITS, step=1)
    skip_outliers = st.sidebar.checkbox("Skip outlier removal", value=False)
    save_outputs = st.sidebar.checkbox("Save outputs to disk", value=True)

    if st.sidebar.button("Run pipeline"):
        with st.spinner("Running ensemble pipeline..."):
            try:
                results = run_pipeline(data_dir, output_dir, cv, skip_outliers, save_outputs=save_outputs)
                display_results(results)
            except Exception as exc:
                st.error("Pipeline execution failed. See exception details below.")
                st.exception(exc)

    st.sidebar.markdown("---")
    st.sidebar.write(
        "Make sure the training and test files are available in the selected data directory. "
        "The default path is `data/raw` relative to this repository."
    )


def cli_main(args=None):
    parser = argparse.ArgumentParser(description="Titanic Ensemble Model Pipeline")
    parser.add_argument("--data-dir", default=str(DATA_DIR), help="Directory with train/test CSV files")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--cv", type=int, default=CV_SPLITS, help="Cross-validation folds")
    parser.add_argument("--skip-outliers", action="store_true", help="Skip outlier removal")
    parsed = parser.parse_args(args=args)

    return run_pipeline(
        parsed.data_dir,
        parsed.output_dir,
        parsed.cv,
        parsed.skip_outliers,
        save_outputs=True,
    )


if __name__ == "__main__":
    import sys

    # Streamlit does not always set STREAMLIT_SERVER_RUNNING, so always
    # start the Streamlit UI when running via `streamlit run`.
    if "--cli" in sys.argv:
        cli_main(args=[arg for arg in sys.argv[1:] if arg != "--cli"])
    else:
        main()
