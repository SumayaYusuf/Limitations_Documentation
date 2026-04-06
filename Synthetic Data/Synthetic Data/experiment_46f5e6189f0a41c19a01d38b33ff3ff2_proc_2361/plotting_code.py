import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()  # Exit if data cannot be loaded, as no plots can be made

dataset_name = "synthetic_t2dm_data"
stage_names = ["Healthy", "Pre-diabetes", "Medication-controlled", "Insulin-dependent"]
epochs = 20  # Fixed number of epochs from experiment code


# Helper function to compute average loss per epoch across multiple models
def get_avg_epoch_losses(losses_data_per_param):
    if not losses_data_per_param:
        return []

    # Extract unique epochs and sort them
    epochs_present = sorted(list(set(d["epoch"] for d in losses_data_per_param)))
    avg_losses = []

    for epoch in epochs_present:
        # Collect losses for all models at the current epoch
        losses_this_epoch = [
            d["loss"] for d in losses_data_per_param if d["epoch"] == epoch
        ]
        if losses_this_epoch:
            avg_losses.append(np.mean(losses_this_epoch))
        else:
            avg_losses.append(np.nan)  # Should not occur if data is complete
    return avg_losses


# --- Activation Function Tuning Plots ---
ablation_type_act = "activation_function_tuning"
act_data = experiment_data[ablation_type_act][dataset_name]
act_param_names = list(act_data["losses"]["train"].keys())

# Plot 1: Training MAE Loss Curves for Different Activation Functions
try:
    plt.figure(figsize=(10, 6))
    for param_name in act_param_names:
        train_losses = get_avg_epoch_losses(act_data["losses"]["train"][param_name])
        plt.plot(range(epochs), train_losses, label=param_name)

    plt.title(
        f"Training MAE Loss Curves for Different Activation Functions on {dataset_name}"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Average Training MAE Loss")
    plt.legend(title="Activation Function")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            working_dir, f"{dataset_name}_ActivationTuning_TrainLossCurves.png"
        )
    )
    plt.close()
    print("Plot: Activation Tuning Training Loss Curves saved.")
except Exception as e:
    print(f"Error creating Activation Tuning Training Loss Curves plot: {e}")
    plt.close()

# Plot 2: Overall Average Validation MAE for Activation Functions (Bar Chart)
try:
    plt.figure(figsize=(10, 6))
    mae_values = [act_data["overall_avg_mae_all_models"][p] for p in act_param_names]
    plt.bar(act_param_names, mae_values, color="skyblue")
    plt.title(
        f"Overall Average Validation MAE Across Models by Activation Function on {dataset_name}"
    )
    plt.xlabel("Activation Function")
    plt.ylabel("Average Validation MAE")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{dataset_name}_ActivationTuning_OverallValMAE.png")
    )
    plt.close()
    print("Plot: Activation Tuning Overall Validation MAE saved.")
except Exception as e:
    print(f"Error creating Activation Tuning Overall Validation MAE plot: {e}")
    plt.close()

# Plot 3: MAECMPR per Stage for Activation Functions (Grouped Bar Chart)
try:
    plt.figure(figsize=(12, 7))
    x = np.arange(len(stage_names))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    for param_name in act_param_names:
        offset = width * multiplier
        maecmpr_values = act_data["maecmpr_per_stage"][param_name]
        plt.bar(x + offset, maecmpr_values, width, label=param_name)
        multiplier += 1

    plt.title(f"MAECMPR per Stage for Different Activation Functions on {dataset_name}")
    plt.ylabel("MAECMPR Value")
    # Center the group of bars on the x-tick
    plt.xticks(
        x + width * (len(act_param_names) - 1) / 2, stage_names, rotation=20, ha="right"
    )
    plt.xlabel("T2DM Stage")
    plt.legend(title="Activation Function", loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            working_dir, f"{dataset_name}_ActivationTuning_MAECMPR_per_Stage.png"
        )
    )
    plt.close()
    print("Plot: Activation Tuning MAECMPR per Stage saved.")
except Exception as e:
    print(f"Error creating Activation Tuning MAECMPR per Stage plot: {e}")
    plt.close()

# --- Dropout Ablation Plots ---
ablation_type_dropout = "dropout_ablation"
dropout_data = experiment_data[ablation_type_dropout][dataset_name]
dropout_param_names = list(dropout_data["losses"]["train"].keys())

# Plot 4: Training MAE Loss Curves for Different Dropout Rates
try:
    plt.figure(figsize=(10, 6))
    for param_name in dropout_param_names:
        train_losses = get_avg_epoch_losses(dropout_data["losses"]["train"][param_name])
        label_name = param_name.replace("Dropout_", "Dropout Rate ")
        plt.plot(range(epochs), train_losses, label=label_name)

    plt.title(
        f"Training MAE Loss Curves for Different Dropout Rates (with ReLU) on {dataset_name}"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Average Training MAE Loss")
    plt.legend(title="Dropout Rate")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{dataset_name}_DropoutAblation_TrainLossCurves.png")
    )
    plt.close()
    print("Plot: Dropout Ablation Training Loss Curves saved.")
except Exception as e:
    print(f"Error creating Dropout Ablation Training Loss Curves plot: {e}")
    plt.close()

# Plot 5: Overall Average Validation MAE for Dropout Rates (Bar Chart)
try:
    plt.figure(figsize=(10, 6))
    mae_values = [
        dropout_data["overall_avg_mae_all_models"][p] for p in dropout_param_names
    ]
    labels = [p.replace("Dropout_", "Dropout Rate ") for p in dropout_param_names]
    plt.bar(labels, mae_values, color="lightcoral")
    plt.title(
        f"Overall Average Validation MAE Across Models by Dropout Rate (with ReLU) on {dataset_name}"
    )
    plt.xlabel("Dropout Rate")
    plt.ylabel("Average Validation MAE")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{dataset_name}_DropoutAblation_OverallValMAE.png")
    )
    plt.close()
    print("Plot: Dropout Ablation Overall Validation MAE saved.")
except Exception as e:
    print(f"Error creating Dropout Ablation Overall Validation MAE plot: {e}")
    plt.close()

# Plot 6: MAECMPR per Stage for Dropout Rates (Grouped Bar Chart)
try:
    plt.figure(figsize=(12, 7))
    x = np.arange(len(stage_names))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0

    for param_name in dropout_param_names:
        offset = width * multiplier
        maecmpr_values = dropout_data["maecmpr_per_stage"][param_name]
        label_name = param_name.replace("Dropout_", "Dropout Rate ")
        plt.bar(x + offset, maecmpr_values, width, label=label_name)
        multiplier += 1

    plt.title(
        f"MAECMPR per Stage for Different Dropout Rates (with ReLU) on {dataset_name}"
    )
    plt.ylabel("MAECMPR Value")
    # Center the group of bars on the x-tick
    plt.xticks(
        x + width * (len(dropout_param_names) - 1) / 2,
        stage_names,
        rotation=20,
        ha="right",
    )
    plt.xlabel("T2DM Stage")
    plt.legend(title="Dropout Rate", loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            working_dir, f"{dataset_name}_DropoutAblation_MAECMPR_per_Stage.png"
        )
    )
    plt.close()
    print("Plot: Dropout Ablation MAECMPR per Stage saved.")
except Exception as e:
    print(f"Error creating Dropout Ablation MAECMPR per Stage plot: {e}")
    plt.close()

print("\nAll plotting attempts completed.")
