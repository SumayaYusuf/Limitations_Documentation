import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    data_path = experiment_data["input_normalization_strategy_ablation"][
        "synthetic_t2dm_data"
    ]
    strategies = list(data_path["overall_avg_mae_all_models"].keys())
    num_modalities = 4
    num_features_per_modality = 5
    stage_names = [
        "Healthy",
        "Pre-diabetes",
        "Medication-controlled",
        "Insulin-dependent",
    ]

except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()

# --- Plot 1: Training Loss Curves for each Normalization Strategy ---
try:
    plt.figure(figsize=(10, 6))
    for strategy_name in strategies:
        train_losses_raw = data_path["losses"]["train"][strategy_name]

        epochs = sorted(list(set([d["epoch"] for d in train_losses_raw])))
        avg_losses_per_epoch = []
        for epoch in epochs:
            epoch_losses = [d["loss"] for d in train_losses_raw if d["epoch"] == epoch]
            if epoch_losses:
                avg_losses_per_epoch.append(np.mean(epoch_losses))
            else:
                avg_losses_per_epoch.append(np.nan)

        plt.plot(
            epochs,
            avg_losses_per_epoch,
            label=f'{strategy_name.replace("_", " ").title()} Normalization',
        )

    plt.title("Synthetic T2DM Data: Training Loss Curves by Normalization Strategy")
    plt.xlabel("Epoch")
    plt.ylabel("Average Training Loss (MAE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "synthetic_t2dm_training_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Training Loss Curves plot: {e}")
    plt.close()

# --- Plot 2: Overall Average Validation MAE (Across all Models) Comparison ---
try:
    mae_values = [data_path["overall_avg_mae_all_models"][s] for s in strategies]
    strategy_labels = [s.replace("_", " ").title() for s in strategies]

    plt.figure(figsize=(8, 6))
    plt.bar(strategy_labels, mae_values, color=["skyblue", "lightcoral", "lightgreen"])
    plt.title(
        "Synthetic T2DM Data: Overall Average Validation MAE by Normalization Strategy"
    )
    plt.xlabel("Normalization Strategy")
    plt.ylabel("Average MAE (across all cross-modal predictions)")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "synthetic_t2dm_overall_avg_val_mae.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Overall Average Validation MAE plot: {e}")
    plt.close()

# --- Plot 3: MAECMPR per Stage Comparison ---
try:
    maecmpr_data = {}
    for strategy_name in strategies:
        maecmpr_data[strategy_name] = data_path["maecmpr_per_stage"][strategy_name]

    x = np.arange(len(stage_names))
    width = 0.2
    multiplier = 0

    fig, ax = plt.subplots(figsize=(12, 7))

    for strategy_name in strategies:
        offset = width * multiplier
        ax.bar(
            x + offset,
            maecmpr_data[strategy_name],
            width,
            label=strategy_name.replace("_", " ").title(),
        )
        multiplier += 1

    ax.set_title(
        "Synthetic T2DM Data: MAECMPR per T2DM Stage by Normalization Strategy"
    )
    ax.set_xlabel("T2DM Stage")
    ax.set_ylabel("Mean Absolute Cross-Modal Prediction Residual (MAECMPR)")
    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(stage_names)
    ax.legend(title="Normalization Strategy")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "synthetic_t2dm_maecmpr_per_stage.png"))
    plt.close()
except Exception as e:
    print(f"Error creating MAECMPR per Stage plot: {e}")
    plt.close()

# --- Plot 4: Sample Validation Predictions vs. Ground Truth (e.g., Z-score, Modality 0) ---
try:
    sample_strategy = "z_score"
    if sample_strategy not in strategies:
        sample_strategy = strategies[0]

    sample_modality_idx = 0

    val_predictions = data_path["val_predictions"][sample_strategy][sample_modality_idx]
    val_ground_truth = data_path["val_ground_truth"][sample_strategy][
        sample_modality_idx
    ]

    num_samples = 5
    if len(val_predictions) > num_samples:
        sample_indices = np.random.choice(
            len(val_predictions), num_samples, replace=False
        )
    else:
        sample_indices = np.arange(len(val_predictions))

    fig, axes = plt.subplots(
        num_samples, 1, figsize=(10, 3 * num_samples), sharex=True, sharey=True
    )
    if num_samples == 1:
        axes = [axes]

    for i, p_idx in enumerate(sample_indices):
        ax = axes[i]
        predicted_features = val_predictions[p_idx]
        ground_truth_features = val_ground_truth[p_idx]

        feature_labels = [f"Feature {j+1}" for j in range(num_features_per_modality)]
        x_indices = np.arange(num_features_per_modality)

        ax.plot(
            x_indices, ground_truth_features, "o-", label="Ground Truth", color="blue"
        )
        ax.plot(x_indices, predicted_features, "x--", label="Predicted", color="red")

        ax.set_xticks(x_indices)
        ax.set_xticklabels(feature_labels)
        ax.set_ylabel("Feature Value")
        ax.set_title(f"Sample Participant {p_idx+1} (Modality {sample_modality_idx})")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.7)

    fig.suptitle(
        f'Synthetic T2DM Data: Validation Predictions vs. Ground Truth\n({sample_strategy.replace("_", " ").title()} Normalization, Modality {sample_modality_idx})',
        y=1.02,
    )
    plt.xlabel("Features within Modality")
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(
        os.path.join(
            working_dir,
            f"synthetic_t2dm_{sample_strategy}_mod{sample_modality_idx}_val_preds_gt_samples.png",
        )
    )
    plt.close()
except Exception as e:
    print(f"Error creating Sample Validation Predictions vs. Ground Truth plot: {e}")
    plt.close()
