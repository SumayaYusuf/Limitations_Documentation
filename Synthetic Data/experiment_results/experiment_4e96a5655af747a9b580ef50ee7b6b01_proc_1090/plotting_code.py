import matplotlib.pyplot as plt
import numpy as np
import os

# Define constants from the experiment code for clarity and consistent access
NUM_STAGES = 4
NUM_MODALITIES = 5
FEATURE_DIM = 15

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)  # Ensure working directory exists

# Load experiment data
experiment_data = {}
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    print("Experiment data loaded successfully.")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    # If data cannot be loaded, further plotting attempts will fail, so we might exit or skip.
    # For this exercise, we'll continue with empty data, which will trigger messages in try-except blocks.

# Plot 1: Overall MAECMPR Across T2DM Progression Stages
try:
    plt.figure(figsize=(8, 5))
    maecmpr_values = []
    stages = np.arange(NUM_STAGES)

    for s_idx in stages:
        stage_key = f"stage_{s_idx}"
        if (
            stage_key in experiment_data
            and experiment_data[stage_key]["metrics"]["val"]
        ):
            maecmpr_values.append(experiment_data[stage_key]["metrics"]["val"][-1])
        else:
            maecmpr_values.append(np.nan)  # Append NaN if data is missing for a stage

    if not all(np.isnan(maecmpr_values)):  # Only plot if there's actual data
        plt.plot(stages, maecmpr_values, marker="o", linestyle="-", color="blue")
        plt.title("Overall MAECMPR Across T2DM Progression Stages")
        plt.xlabel("T2DM Progression Stage")
        plt.ylabel("Mean Absolute Error of Cross-Modal Prediction Residuals (MAECMPR)")
        plt.xticks(stages, [f"Stage {s}" for s in stages])
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(os.path.join(working_dir, "overall_maecmpr_across_stages.png"))
        print("Plot 'overall_maecmpr_across_stages.png' created.")
    else:
        print("Skipping 'overall_maecmpr_across_stages.png' plot due to missing data.")
    plt.close()
except Exception as e:
    print(f"Error creating MAECMPR Across Stages plot: {e}")
    plt.close()

# Plot 2: Training Loss Curves for a Selected Modality Across All Stages
try:
    plt.figure(figsize=(10, 6))
    target_modality_to_plot = 0  # We choose modality 0 as an example for demonstration
    max_epochs = 0
    plots_made = False

    for s_idx in range(NUM_STAGES):
        key = f"stage_{s_idx}_modality_{target_modality_to_plot}"
        if key in experiment_data and experiment_data[key]["losses"]["train"]:
            train_losses = experiment_data[key]["losses"]["train"]
            plt.plot(train_losses, label=f"Stage {s_idx}")
            max_epochs = max(max_epochs, len(train_losses))
            plots_made = True

    if plots_made:
        plt.title(
            f"Training Loss Curves for Predicting Modality {target_modality_to_plot} (Across T2DM Stages)"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Training MAE Loss")
        plt.legend(title="T2DM Stage")
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.xlim(0, max_epochs - 1 if max_epochs > 0 else 0)  # Adjust x-axis limit
        plt.savefig(
            os.path.join(
                working_dir,
                f"training_loss_modality_{target_modality_to_plot}_all_stages.png",
            )
        )
        print(
            f"Plot 'training_loss_modality_{target_modality_to_plot}_all_stages.png' created."
        )
    else:
        print(
            f"Skipping 'training_loss_modality_{target_modality_to_plot}_all_stages.png' plot due to missing data."
        )
    plt.close()
except Exception as e:
    print(f"Error creating Training Loss Curves plot: {e}")
    plt.close()

# Plot 3: Predictions vs. Ground Truth for Stage 0, Modality 0, Feature 0
try:
    plt.figure(figsize=(7, 7))
    stage_idx_plot3 = 0
    modality_idx_plot3 = 0
    feature_idx_plot3 = 0  # We choose the first feature dimension
    key_plot3 = f"stage_{stage_idx_plot3}_modality_{modality_idx_plot3}"

    if (
        key_plot3 in experiment_data
        and experiment_data[key_plot3]["predictions"]
        and experiment_data[key_plot3]["ground_truth"]
        and len(experiment_data[key_plot3]["predictions"][0]) > 0
    ):  # Ensure data exists and is not empty

        predictions = experiment_data[key_plot3]["predictions"][0][:, feature_idx_plot3]
        ground_truth = experiment_data[key_plot3]["ground_truth"][0][
            :, feature_idx_plot3
        ]

        plt.scatter(ground_truth, predictions, alpha=0.6, s=10)
        # Plot y=x line for ideal prediction
        min_val = min(ground_truth.min(), predictions.min())
        max_val = max(ground_truth.max(), predictions.max())
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="red",
            linestyle="--",
            label="Ideal Prediction (y=x)",
        )
        plt.title(
            f"Predictions vs. Ground Truth for Stage {stage_idx_plot3}, Modality {modality_idx_plot3}, Feature {feature_idx_plot3}\n(T2DM Stage: Initial / Low Discordance)"
        )
        plt.xlabel(
            f"Ground Truth (Modality {modality_idx_plot3}, Feature {feature_idx_plot3})"
        )
        plt.ylabel(
            f"Predictions (Modality {modality_idx_plot3}, Feature {feature_idx_plot3})"
        )
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(
            os.path.join(
                working_dir,
                f"predictions_gt_stage_{stage_idx_plot3}_modality_{modality_idx_plot3}_feature_{feature_idx_plot3}.png",
            )
        )
        print(
            f"Plot 'predictions_gt_stage_{stage_idx_plot3}_modality_{modality_idx_plot3}_feature_{feature_idx_plot3}.png' created."
        )
    else:
        print(
            f"Skipping prediction/ground truth plot for Stage {stage_idx_plot3}, Modality {modality_idx_plot3}, Feature {feature_idx_plot3} due to missing data."
        )
    plt.close()
except Exception as e:
    print(f"Error creating Predictions vs. Ground Truth plot for Stage 0: {e}")
    plt.close()

# Plot 4: Predictions vs. Ground Truth for Last Stage, Modality 0, Feature 0
try:
    plt.figure(figsize=(7, 7))
    stage_idx_plot4 = NUM_STAGES - 1  # Last stage
    modality_idx_plot4 = 0
    feature_idx_plot4 = 0  # We choose the first feature dimension
    key_plot4 = f"stage_{stage_idx_plot4}_modality_{modality_idx_plot4}"

    if (
        key_plot4 in experiment_data
        and experiment_data[key_plot4]["predictions"]
        and experiment_data[key_plot4]["ground_truth"]
        and len(experiment_data[key_plot4]["predictions"][0]) > 0
    ):

        predictions = experiment_data[key_plot4]["predictions"][0][:, feature_idx_plot4]
        ground_truth = experiment_data[key_plot4]["ground_truth"][0][
            :, feature_idx_plot4
        ]

        plt.scatter(ground_truth, predictions, alpha=0.6, s=10)
        # Plot y=x line for ideal prediction
        min_val = min(ground_truth.min(), predictions.min())
        max_val = max(ground_truth.max(), predictions.max())
        plt.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="red",
            linestyle="--",
            label="Ideal Prediction (y=x)",
        )
        plt.title(
            f"Predictions vs. Ground Truth for Stage {stage_idx_plot4}, Modality {modality_idx_plot4}, Feature {feature_idx_plot4}\n(T2DM Stage: Last / High Discordance)"
        )
        plt.xlabel(
            f"Ground Truth (Modality {modality_idx_plot4}, Feature {feature_idx_plot4})"
        )
        plt.ylabel(
            f"Predictions (Modality {modality_idx_plot4}, Feature {feature_idx_plot4})"
        )
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig(
            os.path.join(
                working_dir,
                f"predictions_gt_stage_{stage_idx_plot4}_modality_{modality_idx_plot4}_feature_{feature_idx_plot4}.png",
            )
        )
        print(
            f"Plot 'predictions_gt_stage_{stage_idx_plot4}_modality_{modality_idx_plot4}_feature_{feature_idx_plot4}.png' created."
        )
    else:
        print(
            f"Skipping prediction/ground truth plot for Stage {stage_idx_plot4}, Modality {modality_idx_plot4}, Feature {feature_idx_plot4} due to missing data."
        )
    plt.close()
except Exception as e:
    print(f"Error creating Predictions vs. Ground Truth plot for Last Stage: {e}")
    plt.close()

# Print out evaluation metric(s) - MAECMPR for each stage
print("\n--- Evaluation Metric: Overall MAECMPR per Stage ---")
for s_idx in range(NUM_STAGES):
    stage_key = f"stage_{s_idx}"
    if stage_key in experiment_data and experiment_data[stage_key]["metrics"]["val"]:
        print(
            f"Stage {s_idx}: MAECMPR = {experiment_data[stage_key]['metrics']['val'][-1]:.4f}"
        )
    else:
        print(f"Stage {s_idx}: MAECMPR data not found.")
