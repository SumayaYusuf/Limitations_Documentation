import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Ensure working directory exists
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ABLATION_KEY = "robustness_to_synthetic_data_characteristics"

    # Extract dataset names and activation function names
    dataset_names = list(experiment_data[ABLATION_KEY].keys())
    # Assuming activation functions are consistent across datasets, pick from the first one
    if dataset_names:
        first_dataset = dataset_names[0]
        activation_functions = list(
            experiment_data[ABLATION_KEY][first_dataset][
                "overall_avg_mae_all_models"
            ].keys()
        )
    else:
        print("No datasets found in experiment data.")
        activation_functions = []

except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}
    ABLATION_KEY = ""
    dataset_names = []
    activation_functions = []

# --- Plotting Section ---

if experiment_data and ABLATION_KEY in experiment_data and dataset_names:
    stage_names = [
        "Healthy",
        "Pre-diabetes",
        "Medication-controlled",
        "Insulin-dependent",
    ]
    epochs_total = 20  # Defined in the experiment code

    for dataset_name in dataset_names:
        dataset_info = experiment_data[ABLATION_KEY][dataset_name]

        # Plot 1: Overall Average Validation MAE across Activation Functions for Each Dataset
        try:
            plt.figure(figsize=(10, 6))
            mae_values = [
                dataset_info["overall_avg_mae_all_models"][act]
                for act in activation_functions
            ]

            plt.bar(
                activation_functions,
                mae_values,
                color=plt.cm.Paired(np.arange(len(activation_functions))),
            )
            plt.xlabel("Activation Function")
            plt.ylabel("Overall Average Validation MAE")
            plt.title(
                f"Overall Average Validation MAE by Activation Function\nDataset: {dataset_name.replace('_', ' ').title()}"
            )
            plt.xticks(rotation=45, ha="right")
            plt.ylim(bottom=0)
            plt.grid(axis="y", linestyle="--", alpha=0.6)
            plt.tight_layout()
            plot_filename = f"{dataset_name}_overall_avg_mae.png"
            plt.savefig(os.path.join(working_dir, plot_filename))
            plt.close()
            print(f"Saved plot: {plot_filename}")
        except Exception as e:
            print(f"Error creating Overall Average MAE plot for {dataset_name}: {e}")
            plt.close()

        # Plot 2: MAECMPR per Stage for Each Activation Function and Dataset
        try:
            plt.figure(figsize=(12, 7))
            num_stages = len(stage_names)
            bar_width = 0.15
            index = np.arange(num_stages)

            for i, act_name in enumerate(activation_functions):
                maecmpr_vals = dataset_info["maecmpr_per_stage"][act_name]
                plt.bar(
                    index + i * bar_width,
                    maecmpr_vals,
                    bar_width,
                    label=act_name,
                    color=plt.cm.viridis(i / len(activation_functions)),
                )

            plt.xlabel("Disease Stage")
            plt.ylabel("Mean Absolute Cross-Modal Prediction Residual (MAECMPR)")
            plt.title(
                f"MAECMPR per Stage by Activation Function\nDataset: {dataset_name.replace('_', ' ').title()}"
            )
            plt.xticks(
                index + bar_width * (len(activation_functions) - 1) / 2,
                stage_names,
                rotation=20,
                ha="right",
            )
            plt.legend(
                title="Activation Function", bbox_to_anchor=(1.05, 1), loc="upper left"
            )
            plt.ylim(bottom=0)
            plt.grid(axis="y", linestyle="--", alpha=0.6)
            plt.tight_layout()
            plot_filename = f"{dataset_name}_maecmpr_per_stage.png"
            plt.savefig(os.path.join(working_dir, plot_filename))
            plt.close()
            print(f"Saved plot: {plot_filename}")
        except Exception as e:
            print(f"Error creating MAECMPR per Stage plot for {dataset_name}: {e}")
            plt.close()

        # Plot 3: Average Training Loss Curves and Final Validation MAE for Each Dataset
        try:
            plt.figure(figsize=(12, 7))

            for i, act_name in enumerate(activation_functions):
                # Aggregate training losses across modalities per epoch
                train_losses_per_epoch = {}
                for entry in dataset_info["losses"]["train"][act_name]:
                    epoch = entry["epoch"]
                    loss = entry["loss"]
                    if epoch not in train_losses_per_epoch:
                        train_losses_per_epoch[epoch] = []
                    train_losses_per_epoch[epoch].append(loss)

                avg_train_losses = [
                    np.mean(train_losses_per_epoch[e])
                    for e in sorted(train_losses_per_epoch.keys())
                ]

                # Get the overall average validation MAE
                final_val_mae = dataset_info["overall_avg_mae_all_models"][act_name]

                # Plot training curve
                plt.plot(
                    range(epochs_total),
                    avg_train_losses,
                    label=f"{act_name} Train Loss",
                    color=plt.cm.tab10(i),
                )
                # Plot final validation MAE as a horizontal line
                plt.hlines(
                    final_val_mae,
                    0,
                    epochs_total - 1,
                    linestyle="--",
                    color=plt.cm.tab10(i),
                    label=f"{act_name} Val MAE",
                )

            plt.xlabel("Epoch")
            plt.ylabel("Loss (MAE)")
            plt.title(
                f"Average Training Loss and Final Validation MAE\nDataset: {dataset_name.replace('_', ' ').title()}"
            )
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.ylim(bottom=0)
            plt.tight_layout()
            plot_filename = f"{dataset_name}_train_val_curves.png"
            plt.savefig(os.path.join(working_dir, plot_filename))
            plt.close()
            print(f"Saved plot: {plot_filename}")
        except Exception as e:
            print(
                f"Error creating Training/Validation curves plot for {dataset_name}: {e}"
            )
            plt.close()
