import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory
working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    # If data loading fails, subsequent plotting attempts will also fail.
    # It's better to exit or return in a real application, for this exercise we print and continue.
    exit()

# Extract and sort weight decay values to ensure consistent plotting order
weight_decay_values_float = sorted(
    [float(k.split("_")[1]) for k in experiment_data["weight_decay_tuning"].keys()]
)
wd_keys_sorted = [f"wd_{wd}" for wd in weight_decay_values_float]

# --- Plot 1: Average Training MAE Loss vs. Epoch for each Weight Decay ---
try:
    plt.figure(figsize=(10, 6))
    for wd_key in wd_keys_sorted:
        train_losses_raw = experiment_data["weight_decay_tuning"][wd_key]["losses"][
            "train"
        ]

        # Aggregate losses by epoch, averaging across all modality models
        epochs_data = {}
        for item in train_losses_raw:
            epoch = item["epoch"]
            loss = item["loss"]
            epochs_data.setdefault(epoch, []).append(loss)

        epochs = sorted(epochs_data.keys())
        avg_losses = [np.mean(epochs_data[e]) for e in epochs]

        plt.plot(
            epochs, avg_losses, label=f'Weight Decay: {float(wd_key.split("_")[1]):.5f}'
        )

    plt.title("Average Training MAE Loss vs. Epoch for T2DM Cross-Modal Prediction")
    plt.xlabel("Epoch")
    plt.ylabel("Average Training MAE Loss")
    plt.legend(title="Weight Decay")
    plt.grid(True)
    plt.savefig(os.path.join(working_dir, "T2DM_Training_Loss_Curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Training Loss Curves plot: {e}")
    plt.close()

# --- Plot 2: Overall Average Validation MAE Across Weight Decay Values ---
try:
    val_maes = [
        experiment_data["weight_decay_tuning"][wd_key]["metrics"]["val"][0][
            "overall_avg_mae_all_models"
        ]
        for wd_key in wd_keys_sorted
    ]

    plt.figure(figsize=(10, 6))
    plt.plot(weight_decay_values_float, val_maes, marker="o", linestyle="-")
    plt.xscale(
        "log"
    )  # Use log scale for x-axis due to wide range of weight decay values
    plt.title("Overall Average Validation MAE Across Weight Decay Values (T2DM)")
    plt.xlabel("Weight Decay (log scale)")
    plt.ylabel("Overall Average Validation MAE")
    plt.grid(True)
    plt.savefig(os.path.join(working_dir, "T2DM_Validation_MAE_vs_Weight_Decay.png"))
    plt.close()
except Exception as e:
    print(f"Error creating Validation MAE plot: {e}")
    plt.close()

# --- Plot 3: MAECMPR per Stage vs. Weight Decay ---
try:
    stage_names = [
        "Healthy",
        "Pre-diabetes",
        "Medication-controlled",
        "Insulin-dependent",
    ]
    maecmpr_data_by_stage = {stage_idx: [] for stage_idx in range(len(stage_names))}

    for wd_key in wd_keys_sorted:
        maecmpr_vals = experiment_data["weight_decay_tuning"][wd_key][
            "maecmpr_per_stage"
        ]
        for stage_idx, val in enumerate(maecmpr_vals):
            maecmpr_data_by_stage[stage_idx].append(val)

    plt.figure(figsize=(10, 6))
    for stage_idx, stage_name in enumerate(stage_names):
        plt.plot(
            weight_decay_values_float,
            maecmpr_data_by_stage[stage_idx],
            marker="o",
            label=f"Stage {stage_idx}: {stage_name}",
        )

    plt.xscale("log")  # Use log scale for x-axis
    plt.title("MAECMPR per Stage vs. Weight Decay for T2DM Dataset")
    plt.xlabel("Weight Decay (log scale)")
    plt.ylabel("Mean Absolute Cross-Modal Prediction Residual (MAECMPR)")
    plt.legend(title="T2DM Stage")
    plt.grid(True)
    plt.savefig(os.path.join(working_dir, "T2DM_MAECMPR_per_Stage_vs_Weight_Decay.png"))
    plt.close()
except Exception as e:
    print(f"Error creating MAECMPR per Stage plot: {e}")
    plt.close()
