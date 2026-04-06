import matplotlib.pyplot as plt
import numpy as np
import os

# Setup working directory as per guidelines
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2026-04-05_22-53-06_intermodal_discordance_diabetes_attempt_2/logs/0-run/experiment_results/experiment_bf36a28d5a164210bd3bb002b72ef414_proc_1088/experiment_data.npy",
    "experiments/2026-04-05_22-53-06_intermodal_discordance_diabetes_attempt_2/logs/0-run/experiment_results/experiment_cceebd04590b486bae7c18af03ac1f0f_proc_1089/experiment_data.npy",
    "experiments/2026-04-05_22-53-06_intermodal_discordance_diabetes_attempt_2/logs/0-run/experiment_results/experiment_68e1c93bff534d69bb876b4a7a6debff_proc_1087/experiment_data.npy",
]

all_experiment_data = []
dataset_name = (
    "synthetic_t2dm_data"  # Assuming a consistent dataset name across all runs
)

try:
    for experiment_data_path in experiment_data_path_list:
        # Construct the full path using AI_SCIENTIST_ROOT if available, otherwise assume local pathing
        full_path = os.path.join(
            os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
        )
        experiment_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(experiment_data)

    if not all_experiment_data:
        raise ValueError("No experiment data loaded after trying all paths.")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    # Exit if data loading is critical and fails for all runs
    exit()

# --- Data Aggregation ---
# Initialize structures to hold data from all runs for aggregation
aggregated_train_losses = {}  # {(model_idx, epoch): [loss_run1, loss_run2, ...]}
aggregated_val_losses = (
    {}
)  # {model_idx: [final_val_loss_run1, final_val_loss_run2, ...]}
all_maecmpr_per_stage_runs = (
    []
)  # [[maecmpr_run1_stage0, ...], [maecmpr_run2_stage0, ...]]

num_modalities = 4  # Based on the problem description (predictor 1-4, so model_idx 0-3)
stage_names = ["Healthy", "Pre-diabetes", "Medication-controlled", "Insulin-dependent"]

successful_runs_count = 0
for run_data in all_experiment_data:
    try:
        train_losses_raw = run_data[dataset_name]["losses"]["train"]
        val_losses_raw = run_data[dataset_name]["losses"]["val"]
        maecmpr_per_stage_run = run_data[dataset_name]["maecmpr_per_stage"]

        # Aggregate training losses
        for item in train_losses_raw:
            model_idx = item["model_idx"]
            epoch = item["epoch"]
            loss = item["loss"]
            key = (model_idx, epoch)
            if key not in aggregated_train_losses:
                aggregated_train_losses[key] = []
            aggregated_train_losses[key].append(loss)

        # Aggregate final validation losses
        for item in val_losses_raw:
            model_idx = item["model_idx"]
            loss = item["loss"]
            if model_idx not in aggregated_val_losses:
                aggregated_val_losses[model_idx] = []
            aggregated_val_losses[model_idx].append(loss)

        # Aggregate MAECMPR per stage
        all_maecmpr_per_stage_runs.append(maecmpr_per_stage_run)
        successful_runs_count += 1
    except KeyError as ke:
        print(
            f"Skipping a run due to missing key '{ke}' in data. Check data structure for dataset '{dataset_name}'."
        )
    except Exception as e:
        print(
            f"An unexpected error occurred while processing one run's data: {e}. Skipping this run."
        )

# If any data was successfully aggregated, proceed with calculations
if successful_runs_count > 0:
    # Calculate mean and standard error for aggregated training losses
    mean_train_losses = {}  # {model_idx: {epoch: mean_loss}}
    sem_train_losses = {}  # {model_idx: {epoch: sem_loss}}

    for (model_idx, epoch), losses in aggregated_train_losses.items():
        if model_idx not in mean_train_losses:
            mean_train_losses[model_idx] = {}
            sem_train_losses[model_idx] = {}
        mean_train_losses[model_idx][epoch] = np.mean(losses)
        # Use ddof=1 for sample standard deviation, then calculate SEM
        sem_train_losses[model_idx][epoch] = (
            np.std(losses, ddof=1) / np.sqrt(len(losses)) if len(losses) > 1 else 0.0
        )

    # Calculate mean and standard error for aggregated final validation losses
    mean_val_losses = {}  # {model_idx: mean_loss}
    sem_val_losses = {}  # {model_idx: sem_loss}
    for model_idx, losses in aggregated_val_losses.items():
        mean_val_losses[model_idx] = np.mean(losses)
        sem_val_losses[model_idx] = (
            np.std(losses, ddof=1) / np.sqrt(len(losses)) if len(losses) > 1 else 0.0
        )

    # Calculate mean and standard error for MAECMPR per stage
    if all_maecmpr_per_stage_runs:
        maecmpr_array = np.array(all_maecmpr_per_stage_runs)
        mean_maecmpr_per_stage = np.mean(maecmpr_array, axis=0)
        sem_maecmpr_per_stage = (
            np.std(maecmpr_array, axis=0, ddof=1) / np.sqrt(maecmpr_array.shape[0])
            if maecmpr_array.shape[0] > 1
            else np.zeros_like(mean_maecmpr_per_stage)
        )
    else:
        mean_maecmpr_per_stage = np.zeros(len(stage_names))
        sem_maecmpr_per_stage = np.zeros(len(stage_names))
else:
    # If no runs were successfully loaded, initialize empty or zeroed data for plotting to avoid errors
    mean_train_losses, sem_train_losses = {}, {}
    mean_val_losses, sem_val_losses = {}, {}
    mean_maecmpr_per_stage = np.zeros(len(stage_names))
    sem_maecmpr_per_stage = np.zeros(len(stage_names))
    print(
        "Warning: No experiment data was successfully aggregated. Plots might be empty."
    )

# --- Plotting ---

# Plot 1: Aggregated Training/Validation Loss Curves with Standard Error
for model_idx in range(num_modalities):
    try:
        plt.figure(figsize=(10, 6))

        epochs_for_model = sorted(mean_train_losses.get(model_idx, {}).keys())
        if epochs_for_model:
            mean_losses = [
                mean_train_losses[model_idx][epoch] for epoch in epochs_for_model
            ]
            sem_losses = [
                sem_train_losses[model_idx][epoch] for epoch in epochs_for_model
            ]

            plt.plot(
                epochs_for_model,
                mean_losses,
                label=f"Mean Training MAE Loss",
                color="blue",
            )
            plt.fill_between(
                epochs_for_model,
                np.array(mean_losses) - np.array(sem_losses),
                np.array(mean_losses) + np.array(sem_losses),
                color="blue",
                alpha=0.2,
                label="Training MAE SEM",
            )
        else:
            plt.text(
                0.5,
                0.5,
                "No training data available for this predictor.",
                horizontalalignment="center",
                verticalalignment="center",
                transform=plt.gca().transAxes,
            )

        if model_idx in mean_val_losses:
            mean_final_val_loss = mean_val_losses[model_idx]
            sem_final_val_loss = sem_val_losses[model_idx]

            # Plot mean final validation loss as a horizontal line
            plt.axhline(
                y=mean_final_val_loss,
                color="red",
                linestyle="--",
                label=f"Mean Final Validation MAE",
            )

            # Add text annotation for mean and SEM
            annotation_text = (
                f"{mean_final_val_loss:.4f} \u00b1 {sem_final_val_loss:.4f}"
            )

            # Position text relative to the plot
            if epochs_for_model:
                x_pos = max(epochs_for_model)  # End of the training curve
                plt.text(
                    x_pos,
                    mean_final_val_loss,
                    annotation_text,
                    color="red",
                    ha="right",
                    va="bottom",
                    fontsize=9,
                )
            elif (
                plt.xlim()[1] > 0
            ):  # If there's a plot range even without training data
                plt.text(
                    plt.xlim()[1] * 0.9,
                    mean_final_val_loss,
                    annotation_text,
                    color="red",
                    ha="right",
                    va="bottom",
                    fontsize=9,
                )
            else:  # Fallback if no data at all
                plt.text(
                    0.5,
                    mean_final_val_loss,
                    annotation_text,
                    color="red",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        plt.title(
            f"{dataset_name}: Aggregated Training & Final Validation MAE for Predictor {model_idx+1}"
        )
        plt.xlabel("Epoch")
        plt.ylabel("MAE Loss")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plot_filename = (
            f"{dataset_name}_aggregated_training_val_loss_predictor_{model_idx+1}.png"
        )
        plt.savefig(os.path.join(working_dir, plot_filename))
        plt.close()
    except Exception as e:
        print(
            f"Error creating aggregated training/validation loss plot for predictor {model_idx+1}: {e}"
        )
        plt.close()

# Plot 2: Aggregated MAECMPR per T2DM Stage with Standard Error Bars
try:
    plt.figure(figsize=(10, 6))

    # Check if there's meaningful MAECMPR data to plot
    if len(mean_maecmpr_per_stage) > 0 and not np.all(mean_maecmpr_per_stage == 0):
        bars = plt.bar(
            stage_names,
            mean_maecmpr_per_stage,
            yerr=sem_maecmpr_per_stage,
            capsize=5,
            color="skyblue",
            label="Mean MAECMPR \u00b1 SEM",
        )

        plt.xlabel("T2DM Stage")
        plt.ylabel("Mean Absolute Cross-Modal Prediction Residual (MAECMPR)")
        plt.title(f"{dataset_name}: Aggregated MAECMPR per T2DM Stage")
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        # Add mean values on top of bars
        for i, bar in enumerate(bars):
            yval = bar.get_height()
            error_val = sem_maecmpr_per_stage[i]
            # Position text slightly above the error bar cap for clarity
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                yval + error_val + 0.005,
                f"{yval:.4f}",
                ha="center",
                va="bottom",
                color="black",
                fontsize=9,
            )

        plt.legend()  # Show legend for error bars
    else:
        plt.text(
            0.5,
            0.5,
            "No MAECMPR data available to plot.",
            horizontalalignment="center",
            verticalalignment="center",
            transform=plt.gca().transAxes,
        )

    plot_filename = f"{dataset_name}_aggregated_maecmpr_per_stage.png"
    plt.savefig(os.path.join(working_dir, plot_filename))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated MAECMPR per stage plot: {e}")
    plt.close()

# Print out aggregated evaluation metrics
print(
    "\n--- Aggregated Mean Absolute Cross-Modal Prediction Residual (MAECMPR) per Stage ---"
)
if len(mean_maecmpr_per_stage) > 0 and not np.all(mean_maecmpr_per_stage == 0):
    for s_idx, stage_name in enumerate(stage_names):
        print(
            f"Stage {s_idx} ({stage_name}): Mean MAECMPR = {mean_maecmpr_per_stage[s_idx]:.4f} \u00b1 {sem_maecmpr_per_stage[s_idx]:.4f} (SEM)"
        )
else:
    print("No MAECMPR data available for printing.")
