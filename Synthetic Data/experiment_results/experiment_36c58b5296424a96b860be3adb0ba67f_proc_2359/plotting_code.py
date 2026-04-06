import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

# Ensure the working directory exists
os.makedirs(working_dir, exist_ok=True)

experiment_data = None
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    print("Experiment data loaded successfully.")
except FileNotFoundError:
    print(
        f"Error: experiment_data.npy not found in {working_dir}. Please run the experiment first."
    )
except Exception as e:
    print(f"Error loading experiment data: {e}")

if experiment_data:
    model_depth_ablation_data = experiment_data.get("model_depth_ablation", {})

    stage_names = [
        "Healthy",
        "Pre-diabetes",
        "Medication-controlled",
        "Insulin-dependent",
    ]
    model_depths_to_test = [1, 2, 3, 4]  # From the experiment code

    for dataset_name, dataset_results in model_depth_ablation_data.items():
        data_seed = dataset_name.split("_")[-1]

        # 1. Plot Overall Average Validation MAE vs. Model Depth
        try:
            val_maes = []
            depth_labels = []
            for num_layers in model_depths_to_test:
                model_depth_key = f"depth_{num_layers}_layers"
                if model_depth_key in dataset_results:
                    val_mae = dataset_results[model_depth_key][
                        "overall_avg_mae_all_models"
                    ]
                    val_maes.append(val_mae)
                    depth_labels.append(str(num_layers))

            if val_maes:
                plt.figure(figsize=(8, 6))
                plt.plot(depth_labels, val_maes, marker="o", linestyle="-")
                plt.title(
                    f"Validation MAE vs. Model Depth (Synthetic T2DM, Seed {data_seed})"
                )
                plt.xlabel("Number of Hidden Layers")
                plt.ylabel("Overall Average Validation MAE")
                plt.grid(True, linestyle="--", alpha=0.6)
                plot_filename = f"val_mae_vs_depth_seed_{data_seed}.png"
                plt.savefig(os.path.join(working_dir, plot_filename))
                plt.close()
                print(f"Plot saved: {plot_filename}")
            else:
                print(f"No validation MAE data found for {dataset_name}.")
        except Exception as e:
            print(f"Error creating Validation MAE plot for {dataset_name}: {e}")
            plt.close()

        # 2. Plot MAECMPR per Stage vs. Model Depth
        try:
            maecmpr_data = {stage: [] for stage in stage_names}
            for num_layers in model_depths_to_test:
                model_depth_key = f"depth_{num_layers}_layers"
                if model_depth_key in dataset_results:
                    maecmpr_vals = dataset_results[model_depth_key]["maecmpr_per_stage"]
                    for i, stage_val in enumerate(maecmpr_vals):
                        maecmpr_data[stage_names[i]].append(stage_val)

            if any(maecmpr_data.values()):
                bar_width = 0.15
                r = np.arange(len(model_depths_to_test))

                plt.figure(figsize=(12, 7))

                for i, stage_name in enumerate(stage_names):
                    if maecmpr_data[stage_name]:  # Ensure data exists for stage
                        plt.bar(
                            r + i * bar_width,
                            maecmpr_data[stage_name],
                            width=bar_width,
                            label=stage_name,
                        )

                plt.title(
                    f"MAECMPR per Stage vs. Model Depth (Synthetic T2DM, Seed {data_seed})"
                )
                plt.xlabel("Number of Hidden Layers")
                plt.ylabel("Mean Absolute Cross-Modal Prediction Residual (MAECMPR)")
                plt.xticks(
                    r + bar_width * (len(stage_names) - 1) / 2, model_depths_to_test
                )
                plt.legend(title="T2DM Stage")
                plt.grid(axis="y", linestyle="--", alpha=0.6)
                plot_filename = f"maecmpr_vs_depth_seed_{data_seed}.png"
                plt.savefig(os.path.join(working_dir, plot_filename))
                plt.close()
                print(f"Plot saved: {plot_filename}")
            else:
                print(f"No MAECMPR data found for {dataset_name}.")
        except Exception as e:
            print(f"Error creating MAECMPR plot for {dataset_name}: {e}")
            plt.close()

        # 3. Plot Training Loss Curves for each Modality and Model Depth
        for num_layers in model_depths_to_test:
            model_depth_key = f"depth_{num_layers}_layers"
            try:
                if model_depth_key in dataset_results:
                    train_losses_raw = dataset_results[model_depth_key]["losses"][
                        "train"
                    ]

                    # Aggregate losses per modality
                    train_losses_per_modality = {
                        0: [],
                        1: [],
                        2: [],
                        3: [],
                    }  # Assuming 4 modalities
                    for item in train_losses_raw:
                        train_losses_per_modality[item["model_idx"]].append(
                            item["loss"]
                        )

                    # Only plot if there's actual training data
                    if any(train_losses_per_modality[0]):
                        plt.figure(figsize=(10, 6))
                        for mod_idx, losses in train_losses_per_modality.items():
                            if losses:
                                plt.plot(
                                    losses, label=f"Predicting Modality {mod_idx + 1}"
                                )

                        plt.title(
                            f"Training Loss Curves for Depth {num_layers}, Seed {data_seed} (Cross-Modal Tasks)"
                        )
                        plt.xlabel("Epoch")
                        plt.ylabel("Training MAE Loss")
                        plt.legend(title="Task")
                        plt.grid(True, linestyle="--", alpha=0.6)
                        plot_filename = (
                            f"train_loss_depth_{num_layers}_seed_{data_seed}.png"
                        )
                        plt.savefig(os.path.join(working_dir, plot_filename))
                        plt.close()
                        print(f"Plot saved: {plot_filename}")
                    else:
                        print(
                            f"No training loss data found for {dataset_name}, {model_depth_key}."
                        )
                else:
                    print(
                        f"Model depth key '{model_depth_key}' not found for {dataset_name}."
                    )
            except Exception as e:
                print(
                    f"Error creating Training Loss plot for {dataset_name}, {model_depth_key}: {e}"
                )
                plt.close()
