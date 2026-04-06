import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data = {}
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    print("Experiment data loaded successfully.")
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Plot 1: MAECMPR Across Stages
try:
    if experiment_data:
        stages = sorted([s for s in experiment_data.keys() if s.startswith("stage_")])
        maecmpr_values = []
        stage_labels = []

        for stage_key in stages:
            if experiment_data[stage_key]["maecmpr"]:
                maecmpr_values.append(experiment_data[stage_key]["maecmpr"][0])
                stage_labels.append(f"Stage {stage_key.split('_')[1]}")

        if maecmpr_values:
            plt.figure(figsize=(8, 6))
            plt.bar(stage_labels, maecmpr_values, color="skyblue")
            plt.xlabel("T2DM Progression Stage")
            plt.ylabel(
                "MAECMPR (Mean Absolute Error of Cross-Modal Prediction Residuals)"
            )
            plt.title("MAECMPR Across T2DM Progression Stages (Synthetic Data)")
            plt.grid(axis="y", linestyle="--")
            plt.tight_layout()
            plot_path = os.path.join(
                working_dir, "synthetic_data_maecmpr_across_stages.png"
            )
            plt.savefig(plot_path)
            plt.close()
            print(f"Plot saved: {plot_path}")
        else:
            print("No MAECMPR values found to plot.")
    else:
        print("Experiment data is empty, cannot plot MAECMPR.")
except Exception as e:
    print(f"Error creating MAECMPR plot: {e}")
    plt.close()

# Plot 2-5: Training and Validation Loss Curves per Modality for each Stage
if experiment_data:
    stages = sorted([s for s in experiment_data.keys() if s.startswith("stage_")])
    modality_names = [f"Modality {m}" for m in range(3)]

    for stage_key in stages:
        try:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle(
                f'Training and Validation Losses for {stage_key.replace("_", " ").title()} Modality Predictions (Synthetic Data)',
                fontsize=16,
            )

            for m_idx, modality_name in enumerate(modality_names):
                modality_loss_key = f"modality_{m_idx}_pred_loss"
                ax = axes[m_idx]

                train_losses = (
                    experiment_data[stage_key]["modality_losses"]
                    .get(modality_loss_key, {})
                    .get("train", [])
                )
                val_losses = (
                    experiment_data[stage_key]["modality_losses"]
                    .get(modality_loss_key, {})
                    .get("val", [])
                )

                if train_losses and val_losses:
                    epochs = range(1, len(train_losses) + 1)
                    ax.plot(epochs, train_losses, label="Train Loss", color="blue")
                    ax.plot(
                        epochs,
                        val_losses,
                        label="Validation Loss",
                        color="red",
                        linestyle="--",
                    )
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss (MSE)")
                    ax.set_title(f"{modality_name} Prediction Loss")
                    ax.legend()
                    ax.grid(True, linestyle=":")
                else:
                    ax.set_title(f"{modality_name} Prediction Loss (No data)")
                    ax.text(
                        0.5,
                        0.5,
                        "No data available",
                        horizontalalignment="center",
                        verticalalignment="center",
                        transform=ax.transAxes,
                    )

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plot_path = os.path.join(
                working_dir, f"synthetic_data_{stage_key}_modality_losses.png"
            )
            plt.savefig(plot_path)
            plt.close(fig)
            print(f"Plot saved: {plot_path}")
        except Exception as e:
            print(f"Error creating loss curve plots for {stage_key}: {e}")
            plt.close()
else:
    print("Experiment data is empty, cannot plot loss curves.")
