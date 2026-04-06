import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)  # Ensure working directory exists

# Load experiment data
experiment_data = {}
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

dataset_name = "synthetic_t2dm_data"
stage_names = ["Healthy", "Pre-diabetes", "Medication-controlled", "Insulin-dependent"]

# --- Print Evaluation Metrics ---
if (
    "activation_function_tuning" in experiment_data
    and dataset_name in experiment_data["activation_function_tuning"]
):
    data_section = experiment_data["activation_function_tuning"][dataset_name]

    overall_avg_mae = data_section.get("overall_avg_mae_all_models", {})
    if overall_avg_mae:
        print("--- Overall Average MAE Across All Cross-Modal Models ---")
        for act_name, mae_value in overall_avg_mae.items():
            print(f"Activation Function: {act_name}, Average MAE: {mae_value:.4f}")

    maecmpr_per_stage = data_section.get("maecmpr_per_stage", {})
    if maecmpr_per_stage:
        print(
            "\n--- Mean Absolute Cross-Modal Prediction Residual (MAECMPR) per Stage ---"
        )
        for act_name, maecmpr_values in maecmpr_per_stage.items():
            print(f"Activation Function: {act_name}")
            for s_idx, maecmpr_val in enumerate(maecmpr_values):
                print(
                    f"  Stage {s_idx} ({stage_names[s_idx]}): MAECMPR = {maecmpr_val:.4f}"
                )
    print("\n")  # Add a newline for separation before plots

# --- Plot 1: MAECMPR per T2DM Stage for Different Activation Functions ---
try:
    if (
        "activation_function_tuning" in experiment_data
        and dataset_name in experiment_data["activation_function_tuning"]
    ):
        data_section = experiment_data["activation_function_tuning"][dataset_name]
        maecmpr_per_stage = data_section.get("maecmpr_per_stage", {})

        num_stages = len(stage_names)

        if maecmpr_per_stage:
            activation_functions = list(maecmpr_per_stage.keys())
            num_activations = len(activation_functions)

            fig, ax = plt.subplots(figsize=(12, 7))
            bar_width = 0.8 / num_activations
            index = np.arange(num_stages)

            for i, act_name in enumerate(activation_functions):
                maecmpr_values = maecmpr_per_stage[act_name]
                ax.bar(index + i * bar_width, maecmpr_values, bar_width, label=act_name)

            ax.set_xlabel("T2DM Stage")
            ax.set_ylabel("MAECMPR")
            ax.set_title(
                "MAECMPR per T2DM Stage for Different Activation Functions\nDataset: Synthetic T2DM Data"
            )
            ax.set_xticks(
                index + bar_width * (num_activations - 1) / 2
            )  # Center xticks
            ax.set_xticklabels(stage_names)
            ax.legend(title="Activation Function")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{dataset_name}_MAECMPR_per_stage.png")
            )
            plt.close()
except Exception as e:
    print(f"Error creating MAECMPR plot: {e}")
    plt.close()

# --- Plot 2: Overall Average MAE Across All Cross-Modal Models per Activation Function ---
try:
    if (
        "activation_function_tuning" in experiment_data
        and dataset_name in experiment_data["activation_function_tuning"]
    ):
        data_section = experiment_data["activation_function_tuning"][dataset_name]
        overall_avg_mae = data_section.get("overall_avg_mae_all_models", {})

        if overall_avg_mae:
            activation_functions = list(overall_avg_mae.keys())
            mae_values = [
                overall_avg_mae[act_name] for act_name in activation_functions
            ]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(activation_functions, mae_values, color="skyblue")
            ax.set_xlabel("Activation Function")
            ax.set_ylabel("Average MAE")
            ax.set_title(
                "Overall Average MAE Across All Cross-Modal Models per Activation Function\nDataset: Synthetic T2DM Data"
            )
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{dataset_name}_overall_avg_MAE.png")
            )
            plt.close()
except Exception as e:
    print(f"Error creating Overall Average MAE plot: {e}")
    plt.close()

# --- Plot 3: Average Training Loss Curves per Activation Function ---
try:
    if (
        "activation_function_tuning" in experiment_data
        and dataset_name in experiment_data["activation_function_tuning"]
    ):
        data_section = experiment_data["activation_function_tuning"][dataset_name]
        train_losses_raw = data_section.get("losses", {}).get("train", {})

        if train_losses_raw:
            fig, ax = plt.subplots(figsize=(12, 7))

            for act_name, losses_list in train_losses_raw.items():
                if losses_list:
                    # Aggregate losses by epoch
                    losses_by_epoch = {}
                    for item in losses_list:
                        epoch = item["epoch"]
                        loss = item["loss"]
                        if epoch not in losses_by_epoch:
                            losses_by_epoch[epoch] = []
                        losses_by_epoch[epoch].append(loss)

                    # Calculate average loss per epoch
                    epochs = sorted(losses_by_epoch.keys())
                    avg_losses = [np.mean(losses_by_epoch[epoch]) for epoch in epochs]

                    ax.plot(epochs, avg_losses, label=act_name)

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Training Loss (MAE)")
            ax.set_title(
                "Average Training Loss Curves per Activation Function\nDataset: Synthetic T2DM Data"
            )
            ax.legend(title="Activation Function")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, f"{dataset_name}_avg_train_loss_curves.png")
            )
            plt.close()
except Exception as e:
    print(f"Error creating Average Training Loss Curves plot: {e}")
    plt.close()
