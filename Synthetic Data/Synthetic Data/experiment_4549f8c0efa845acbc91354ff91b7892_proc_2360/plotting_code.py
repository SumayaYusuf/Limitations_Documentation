import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory as per guidelines
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)  # Ensure working directory exists

# Load experiment data
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    exit()  # Exit if data cannot be loaded, as plotting won't be possible

generalizability_data = experiment_data.get("generalizability_ablation", {})

if not generalizability_data:
    print("No data found in 'generalizability_ablation'. No plots will be generated.")

# Define stage names for better readability in plots
stage_names = ["Healthy", "Pre-diabetes", "Medication-controlled", "Insulin-dependent"]
epochs_range = range(
    20
)  # Assuming 20 epochs from the training function in the problem description

# Iterate through each synthetic dataset to generate plots
for dataset_tag, dataset_results in generalizability_data.items():
    print(f"\nGenerating plots for dataset: {dataset_tag}")

    # Get a sorted list of activation functions for consistent plotting order
    activation_functions = list(dataset_results["losses"]["train"].keys())
    activation_functions.sort()

    # --- Plot 1: Training Loss Curves for each Activation Function ---
    try:
        plt.figure(figsize=(10, 6))
        for act_name in activation_functions:
            train_losses_raw = dataset_results["losses"]["train"][act_name]

            # Aggregate training loss per epoch by averaging across all cross-modal models
            losses_by_epoch = {epoch: [] for epoch in epochs_range}
            for item in train_losses_raw:
                losses_by_epoch[item["epoch"]].append(item["loss"])

            avg_train_losses = [
                np.mean(losses_by_epoch[e]) if losses_by_epoch[e] else np.nan
                for e in epochs_range
            ]
            plt.plot(epochs_range, avg_train_losses, label=act_name)

        plt.title(f"Average Training L1 Loss per Epoch\nDataset: {dataset_tag}")
        plt.xlabel("Epoch")
        plt.ylabel("Average Training L1 Loss")
        plt.legend(title="Activation Function")
        plt.grid(True)
        plot_filename = f"train_loss_curves_{dataset_tag}.png"
        plt.savefig(os.path.join(working_dir, plot_filename))
        plt.close()
        print(f"Saved '{plot_filename}'")
    except Exception as e:
        print(f"Error creating Training Loss Curves plot for {dataset_tag}: {e}")
        plt.close()

    # --- Plot 2: Overall Average Validation MAE per Activation Function ---
    try:
        plt.figure(figsize=(10, 6))
        val_mae_values = [
            dataset_results["overall_avg_mae_all_models"].get(act_name, 0)
            for act_name in activation_functions
        ]

        bars = plt.bar(
            activation_functions,
            val_mae_values,
            color=plt.cm.viridis(np.linspace(0, 1, len(activation_functions))),
        )

        plt.title(
            f"Overall Average Validation Mean Absolute Error (MAE)\nDataset: {dataset_tag}"
        )
        plt.xlabel("Activation Function")
        plt.ylabel("Average Validation MAE")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.xticks(rotation=45, ha="right")
        for bar in bars:
            yval = bar.get_height()
            # Add value labels slightly above the bars
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + 0.001,
                f"{yval:.4f}",
                ha="center",
                va="bottom",
            )
        plt.tight_layout()
        plot_filename = f"overall_val_mae_{dataset_tag}.png"
        plt.savefig(os.path.join(working_dir, plot_filename))
        plt.close()
        print(f"Saved '{plot_filename}'")
    except Exception as e:
        print(
            f"Error creating Overall Average Validation MAE plot for {dataset_tag}: {e}"
        )
        plt.close()

    # --- Plot 3: MAECMPR per Stage for each Activation Function ---
    try:
        plt.figure(figsize=(12, 7))
        maecmpr_data = dataset_results["maecmpr_per_stage"]

        num_activations = len(activation_functions)
        bar_width = 0.15  # Adjust bar width for grouping
        index = np.arange(len(stage_names))  # X-axis locations for stages

        for i, act_name in enumerate(activation_functions):
            # Get MAECMPR values for current activation, default to zeros if not found
            maecmpr_values = maecmpr_data.get(act_name, [0] * len(stage_names))
            # Calculate offset for grouped bars
            offset = bar_width * (i - (num_activations - 1) / 2)
            plt.bar(index + offset, maecmpr_values, bar_width, label=act_name)

        plt.title(
            f"Mean Absolute Cross-Modal Prediction Residual (MAECMPR) per Stage\nDataset: {dataset_tag}"
        )
        plt.xlabel("T2DM Stage")
        plt.ylabel("MAECMPR")
        plt.xticks(index, stage_names, rotation=20, ha="right")
        plt.legend(
            title="Activation Function", bbox_to_anchor=(1.05, 1), loc="upper left"
        )
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()  # Adjust layout to prevent labels overlapping
        plot_filename = f"maecmpr_per_stage_{dataset_tag}.png"
        plt.savefig(os.path.join(working_dir, plot_filename))
        plt.close()
        print(f"Saved '{plot_filename}'")
    except Exception as e:
        print(f"Error creating MAECMPR per Stage plot for {dataset_tag}: {e}")
        plt.close()
