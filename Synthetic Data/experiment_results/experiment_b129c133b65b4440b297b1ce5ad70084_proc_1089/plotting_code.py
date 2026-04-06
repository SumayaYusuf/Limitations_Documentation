import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory as per guidelines
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)  # Ensure working directory exists

experiment_data = None
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    dataset_name = "synthetic_t2dm_data"
except Exception as e:
    print(f"Error loading experiment data: {e}")
    # If data cannot be loaded, further plotting will fail.
    # The script should exit or prevent subsequent plotting blocks from running.
    # For this exercise, we will assume data loading is critical.
    exit()

# Extract data for plotting
train_losses_raw = experiment_data[dataset_name]["losses"]["train"]
val_losses_raw = experiment_data[dataset_name]["losses"]["val"]
maecmpr_per_stage = experiment_data[dataset_name]["maecmpr_per_stage"]

# Group training losses by model_idx for plotting curves
grouped_train_losses = {}
for item in train_losses_raw:
    model_idx = item["model_idx"]
    epoch = item["epoch"]
    loss = item["loss"]
    if model_idx not in grouped_train_losses:
        grouped_train_losses[model_idx] = {"epochs": [], "losses": []}
    grouped_train_losses[model_idx]["epochs"].append(epoch)
    grouped_train_losses[model_idx]["losses"].append(loss)

# Map final validation losses by model_idx
final_val_losses_map = {item["model_idx"]: item["loss"] for item in val_losses_raw}

stage_names = ["Healthy", "Pre-diabetes", "Medication-controlled", "Insulin-dependent"]
num_modalities = 4  # Based on experiment code context

# Plot 1: Training Loss Curves and Final Validation MAE for each Cross-Modal Predictor
for model_idx in range(num_modalities):
    try:
        plt.figure(figsize=(10, 6))
        if model_idx in grouped_train_losses:
            epochs = grouped_train_losses[model_idx]["epochs"]
            losses = grouped_train_losses[model_idx]["losses"]
            plt.plot(
                epochs,
                losses,
                label=f"Model {model_idx+1} Training MAE Loss",
                color="blue",
            )

            if model_idx in final_val_losses_map:
                final_val_loss = final_val_losses_map[model_idx]
                plt.axhline(
                    y=final_val_loss,
                    color="red",
                    linestyle="--",
                    label=f"Model {model_idx+1} Final Validation MAE",
                )
                # Optional: add text for the final validation loss value
                plt.text(
                    epochs[-1],
                    final_val_loss,
                    f"{final_val_loss:.4f}",
                    color="red",
                    ha="right",
                    va="bottom",
                )

        plt.title(
            f"Synthetic T2DM Data: Training & Final Validation MAE for Predictor {model_idx+1}"
        )
        plt.xlabel("Epoch")
        plt.ylabel("MAE Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plot_filename = f"{dataset_name}_training_val_loss_predictor_{model_idx+1}.png"
        plt.savefig(os.path.join(working_dir, plot_filename))
        plt.close()
    except Exception as e:
        print(
            f"Error creating training/validation loss plot for predictor {model_idx+1}: {e}"
        )
        plt.close()

# Plot 2: MAECMPR per T2DM Stage
try:
    plt.figure(figsize=(10, 6))
    bars = plt.bar(stage_names, maecmpr_per_stage, color="skyblue")
    plt.xlabel("T2DM Stage")
    plt.ylabel("Mean Absolute Cross-Modal Prediction Residual (MAECMPR)")
    plt.title(f"Synthetic T2DM Data: MAECMPR per T2DM Stage")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Add MAECMPR values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.005,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
        )

    plot_filename = f"{dataset_name}_maecmpr_per_stage.png"
    plt.savefig(os.path.join(working_dir, plot_filename))
    plt.close()
except Exception as e:
    print(f"Error creating MAECMPR per stage plot: {e}")
    plt.close()

# Print out evaluation metrics (MAECMPR values)
print("\n--- Mean Absolute Cross-Modal Prediction Residual (MAECMPR) per Stage ---")
for s_idx, maecmpr_val in enumerate(maecmpr_per_stage):
    print(f"Stage {s_idx} ({stage_names[s_idx]}): MAECMPR = {maecmpr_val:.4f}")
