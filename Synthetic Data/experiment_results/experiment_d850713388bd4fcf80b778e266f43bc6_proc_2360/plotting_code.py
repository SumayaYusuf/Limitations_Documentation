import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Ensure the working directory exists
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}  # Initialize empty to prevent further errors if loading fails

if experiment_data:  # Only proceed if data loaded successfully
    dataset_keys = experiment_data["generalizability_across_datasets"].keys()
    stage_names = [
        "Healthy",
        "Pre-diabetes",
        "Medication-controlled",
        "Insulin-dependent",
    ]

    for dataset_key in dataset_keys:
        # Extract results for the current dataset
        dataset_results = experiment_data["generalizability_across_datasets"][
            dataset_key
        ]["activation_function_tuning"]

        # Get list of activation functions that were tuned
        activation_functions = list(dataset_results["losses"]["train"].keys())

        # --- Plot 1: Training and Validation MAE Loss Curves for each Activation Function ---
        # This section generates a separate plot for each activation function for a given dataset
        for act_name in activation_functions:
            try:
                # Aggregate training losses per epoch (averaging across models)
                train_losses_raw = dataset_results["losses"]["train"][act_name]
                epochs_data = {}
                for item in train_losses_raw:
                    epoch = item["epoch"]
                    loss = item["loss"]
                    if epoch not in epochs_data:
                        epochs_data[epoch] = []
                    epochs_data[epoch].append(loss)

                epochs = sorted(epochs_data.keys())
                avg_train_losses = [np.mean(epochs_data[e]) for e in epochs]

                # Get the overall average validation MAE (which is a single scalar value)
                val_mae = dataset_results["overall_avg_mae_all_models"].get(act_name)

                plt.figure(figsize=(10, 6))
                plt.plot(epochs, avg_train_losses, label=f"{act_name} Training MAE")
                if val_mae is not None:
                    plt.axhline(
                        y=val_mae,
                        color="r",
                        linestyle="--",
                        label=f"{act_name} Final Validation MAE ({val_mae:.4f})",
                    )

                plt.title(
                    f'Training and Validation MAE Loss for {act_name}\nDataset: {dataset_key.replace("_", " ").title()}'
                )
                plt.xlabel("Epoch")
                plt.ylabel("Mean Absolute Error (MAE)")
                plt.legend()
                plt.grid(True)
                plot_filename = os.path.join(
                    working_dir, f"train_val_mae_loss_{dataset_key}_{act_name}.png"
                )
                plt.savefig(plot_filename)
                plt.close()
            except Exception as e:
                print(
                    f"Error creating Training/Validation MAE Loss plot for {act_name} on {dataset_key}: {e}"
                )
                plt.close()

        # --- Plot 2: Overall Average Validation MAE Across Activation Functions (Bar Chart) ---
        try:
            overall_avg_mae_values = [
                dataset_results["overall_avg_mae_all_models"].get(act)
                for act in activation_functions
            ]

            # Filter out None values if any activation function didn't have results (robustness)
            valid_activations = [
                act
                for i, act in enumerate(activation_functions)
                if overall_avg_mae_values[i] is not None
            ]
            valid_mae_values = [
                val for val in overall_avg_mae_values if val is not None
            ]

            if valid_mae_values:
                plt.figure(figsize=(12, 7))
                plt.bar(valid_activations, valid_mae_values, color="skyblue")
                plt.title(
                    f'Overall Average Validation MAE Across Activation Functions\nDataset: {dataset_key.replace("_", " ").title()}'
                )
                plt.xlabel("Activation Function")
                plt.ylabel("Overall Average Validation MAE")
                plt.xticks(rotation=45, ha="right")
                plt.grid(axis="y", linestyle="--")
                plt.tight_layout()
                plot_filename = os.path.join(
                    working_dir, f"overall_avg_mae_{dataset_key}.png"
                )
                plt.savefig(plot_filename)
                plt.close()
            else:
                print(
                    f"No valid overall_avg_mae_all_models data for {dataset_key} to plot."
                )
        except Exception as e:
            print(
                f"Error creating Overall Average Validation MAE bar chart for {dataset_key}: {e}"
            )
            plt.close()

        # --- Plot 3: MAECMPR per T2DM Stage Across Activation Functions (Grouped Bar Chart) ---
        try:
            maecmpr_data = dataset_results["maecmpr_per_stage"]

            num_stages = len(stage_names)
            bar_width = 0.15  # Width of each individual bar
            x_pos = np.arange(num_stages)  # The label locations for T2DM stages

            plt.figure(figsize=(14, 8))

            num_activations = len(activation_functions)
            # Calculate the starting offset for the first group of bars to center them
            offset_start = -(num_activations - 1) * bar_width / 2

            for i, act_name in enumerate(activation_functions):
                maecmpr_values = maecmpr_data.get(
                    act_name, [0] * num_stages
                )  # Default to zeros if not found
                # Calculate the position for the current bar group
                plt.bar(
                    x_pos + offset_start + i * bar_width,
                    maecmpr_values,
                    bar_width,
                    label=act_name,
                )

            plt.title(
                f'MAECMPR per T2DM Stage Across Activation Functions\nDataset: {dataset_key.replace("_", " ").title()}'
            )
            plt.xlabel("T2DM Stage")
            plt.ylabel("Mean Absolute Cross-Modal Prediction Residual (MAECMPR)")
            plt.xticks(x_pos, stage_names, rotation=45, ha="right")
            plt.legend(title="Activation Function")
            plt.grid(axis="y", linestyle="--")
            plt.tight_layout()
            plot_filename = os.path.join(
                working_dir, f"maecmpr_per_stage_{dataset_key}.png"
            )
            plt.savefig(plot_filename)
            plt.close()
        except Exception as e:
            print(f"Error creating MAECMPR per Stage bar chart for {dataset_key}: {e}")
            plt.close()
else:
    print("No experiment data loaded. Skipping plotting.")
