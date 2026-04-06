import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Experiment data paths from the instruction
experiment_data_path_list = [
    "experiments/2026-04-05_22-53-06_intermodal_discordance_diabetes_attempt_2/logs/0-run/experiment_results/experiment_af78f6c7c09f4f88ac2b8f79d2db4122_proc_2074/experiment_data.npy",
    "experiments/2026-04-05_22-53-06_intermodal_discordance_diabetes_attempt_2/logs/0-run/experiment_results/experiment_d8b322a796314cbd9e7bee42dfa914fd_proc_2075/experiment_data.npy",
    "experiments/2026-04-05_22-53-06_intermodal_discordance_diabetes_attempt_2/logs/0-run/experiment_results/experiment_316afa868a774aa6b18575276ab45512_proc_2073/experiment_data.npy",
]

all_experiment_data = []
try:
    for experiment_data_path in experiment_data_path_list:
        # Construct the full path, assuming AI_SCIENTIST_ROOT is the base for these relative paths
        full_path = os.path.join(
            os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
        )
        if not os.path.exists(full_path):
            print(f"Warning: Data file not found at {full_path}. Skipping.")
            continue
        experiment_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")

dataset_name = "synthetic_t2dm_data"
stage_names = ["Healthy", "Pre-diabetes", "Medication-controlled", "Insulin-dependent"]

# Initialize aggregation structures
activation_functions = []
if all_experiment_data:
    first_data_section = (
        all_experiment_data[0]
        .get("activation_function_tuning", {})
        .get(dataset_name, {})
    )
    if first_data_section:
        # Get activation functions from the first experiment data for consistent keys
        overall_mae_data = first_data_section.get("overall_avg_mae_all_models", {})
        if overall_mae_data:
            activation_functions = sorted(list(overall_mae_data.keys()))

# Prepare dictionaries to store raw values for aggregation
aggregated_overall_mae = {act: [] for act in activation_functions}
aggregated_maecmpr_per_stage = {
    act: {s_idx: [] for s_idx in range(len(stage_names))}
    for act in activation_functions
}
aggregated_train_losses_by_epoch = {
    act: {} for act in activation_functions
}  # {act: {epoch: [loss_run1, loss_run2,...]}}

# Populate aggregation structures by iterating through all loaded experiment data
if not all_experiment_data:
    print("No experiment data loaded. Skipping metric aggregation and plotting.")
else:
    for exp_data in all_experiment_data:
        data_section = exp_data.get("activation_function_tuning", {}).get(
            dataset_name, {}
        )
        if not data_section:
            continue

        # Aggregate overall_avg_mae
        current_overall_avg_mae = data_section.get("overall_avg_mae_all_models", {})
        for act_name, mae_value in current_overall_avg_mae.items():
            if act_name in aggregated_overall_mae:
                aggregated_overall_mae[act_name].append(mae_value)

        # Aggregate maecmpr_per_stage
        current_maecmpr_per_stage = data_section.get("maecmpr_per_stage", {})
        for act_name, maecmpr_values in current_maecmpr_per_stage.items():
            if act_name in aggregated_maecmpr_per_stage:
                for s_idx, maecmpr_val in enumerate(maecmpr_values):
                    if s_idx < len(stage_names):
                        aggregated_maecmpr_per_stage[act_name][s_idx].append(
                            maecmpr_val
                        )

        # Aggregate training losses
        train_losses_raw = data_section.get("losses", {}).get("train", {})
        for act_name, losses_list in train_losses_raw.items():
            if act_name in aggregated_train_losses_by_epoch:
                for item in losses_list:
                    epoch = item["epoch"]
                    loss = item["loss"]
                    if epoch not in aggregated_train_losses_by_epoch[act_name]:
                        aggregated_train_losses_by_epoch[act_name][epoch] = []
                    aggregated_train_losses_by_epoch[act_name][epoch].append(loss)

    # Calculate means and standard errors for plotting and printing
    # Standard error of the mean = standard deviation / sqrt(N)
    # Use ddof=1 for sample standard deviation. If N=1, std_err is 0.

    mean_overall_mae = {}
    std_err_overall_mae = {}
    for act, vals in aggregated_overall_mae.items():
        if vals:
            mean_overall_mae[act] = np.mean(vals)
            std_err_overall_mae[act] = (
                np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            )
        else:
            mean_overall_mae[act] = np.nan
            std_err_overall_mae[act] = np.nan

    mean_maecmpr_per_stage = {}
    std_err_maecmpr_per_stage = {}
    for act_name, stages_data in aggregated_maecmpr_per_stage.items():
        mean_maecmpr_per_stage[act_name] = []
        std_err_maecmpr_per_stage[act_name] = []
        for s_idx in range(len(stage_names)):
            vals = stages_data.get(s_idx, [])
            if vals:
                mean_maecmpr_per_stage[act_name].append(np.mean(vals))
                std_err_maecmpr_per_stage[act_name].append(
                    np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                )
            else:
                mean_maecmpr_per_stage[act_name].append(np.nan)
                std_err_maecmpr_per_stage[act_name].append(np.nan)

    # --- Print Aggregated Evaluation Metrics ---
    print(
        "--- Overall Average MAE Across All Cross-Modal Models (Mean ± Std Error) ---"
    )
    if mean_overall_mae:
        for act_name in activation_functions:
            mean_val = mean_overall_mae.get(act_name, np.nan)
            std_err_val = std_err_overall_mae.get(act_name, np.nan)
            print(
                f"Activation Function: {act_name}, Average MAE: {mean_val:.4f} ± {std_err_val:.4f}"
            )
    else:
        print("No overall MAE data available for aggregation.")

    print(
        "\n--- Mean Absolute Cross-Modal Prediction Residual (MAECMPR) per Stage (Mean ± Std Error) ---"
    )
    if mean_maecmpr_per_stage:
        for act_name in activation_functions:
            print(f"Activation Function: {act_name}")
            current_means = mean_maecmpr_per_stage.get(act_name, [])
            current_stds = std_err_maecmpr_per_stage.get(act_name, [])
            for s_idx, stage_name in enumerate(stage_names):
                mean_val = (
                    current_means[s_idx] if s_idx < len(current_means) else np.nan
                )
                std_err_val = (
                    current_stds[s_idx] if s_idx < len(current_stds) else np.nan
                )
                print(
                    f"  Stage {s_idx} ({stage_name}): MAECMPR = {mean_val:.4f} ± {std_err_val:.4f}"
                )
    else:
        print("No MAECMPR per stage data available for aggregation.")
    print("\n")

    # --- Plot 1: MAECMPR per T2DM Stage for Different Activation Functions (Aggregated) ---
    try:
        if mean_maecmpr_per_stage:
            num_stages = len(stage_names)
            num_activations = len(activation_functions)

            fig, ax = plt.subplots(figsize=(12, 7))
            bar_width = 0.8 / num_activations
            index = np.arange(num_stages)

            for i, act_name in enumerate(activation_functions):
                maecmpr_means = mean_maecmpr_per_stage.get(
                    act_name, [np.nan] * num_stages
                )
                maecmpr_stds = std_err_maecmpr_per_stage.get(
                    act_name, [np.nan] * num_stages
                )
                ax.bar(
                    index + i * bar_width,
                    maecmpr_means,
                    bar_width,
                    yerr=maecmpr_stds,
                    capsize=5,
                    label=f"{act_name} (Mean ± Std Error)",
                )

            ax.set_xlabel("T2DM Stage")
            ax.set_ylabel("MAECMPR")
            ax.set_title(
                f"MAECMPR per T2DM Stage for Different Activation Functions (Mean ± Std Error)\nDataset: {dataset_name}"
            )
            ax.set_xticks(index + bar_width * (num_activations - 1) / 2)
            ax.set_xticklabels(stage_names)
            ax.legend(title="Activation Function")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, f"{dataset_name}_MAECMPR_per_stage_aggregated.png"
                )
            )
            plt.close()
        else:
            print("Skipping MAECMPR per stage plot: No aggregated data.")
    except Exception as e:
        print(f"Error creating MAECMPR plot: {e}")
        plt.close()

    # --- Plot 2: Overall Average MAE Across All Cross-Modal Models per Activation Function (Aggregated) ---
    try:
        if mean_overall_mae:
            activation_functions_plot = list(mean_overall_mae.keys())
            mae_means = [
                mean_overall_mae[act_name] for act_name in activation_functions_plot
            ]
            mae_stds = [
                std_err_overall_mae.get(act_name, 0)
                for act_name in activation_functions_plot
            ]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(
                activation_functions_plot,
                mae_means,
                yerr=mae_stds,
                capsize=5,
                color="skyblue",
                label="Mean MAE ± Std Error",
            )
            ax.set_xlabel("Activation Function")
            ax.set_ylabel("Average MAE")
            ax.set_title(
                f"Overall Average MAE Across All Cross-Modal Models per Activation Function (Mean ± Std Error)\nDataset: {dataset_name}"
            )
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            ax.legend(title="Metric")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, f"{dataset_name}_overall_avg_MAE_aggregated.png"
                )
            )
            plt.close()
        else:
            print("Skipping Overall Average MAE plot: No aggregated data.")
    except Exception as e:
        print(f"Error creating Overall Average MAE plot: {e}")
        plt.close()

    # --- Plot 3: Average Training Loss Curves per Activation Function (Aggregated) ---
    try:
        if aggregated_train_losses_by_epoch:
            fig, ax = plt.subplots(figsize=(12, 7))

            for act_name in activation_functions:
                losses_by_epoch_for_act = aggregated_train_losses_by_epoch.get(
                    act_name, {}
                )
                if losses_by_epoch_for_act:
                    epochs = sorted(losses_by_epoch_for_act.keys())

                    if not epochs:  # Skip if no loss data for this activation function
                        continue

                    mean_losses = [
                        np.mean(losses_by_epoch_for_act[epoch]) for epoch in epochs
                    ]
                    std_err_losses = [
                        (
                            np.std(losses_by_epoch_for_act[epoch], ddof=1)
                            / np.sqrt(len(losses_by_epoch_for_act[epoch]))
                            if len(losses_by_epoch_for_act[epoch]) > 1
                            else 0
                        )
                        for epoch in epochs
                    ]

                    ax.plot(epochs, mean_losses, label=f"{act_name} (Mean)")
                    lower_bound = np.array(mean_losses) - np.array(std_err_losses)
                    upper_bound = np.array(mean_losses) + np.array(std_err_losses)
                    ax.fill_between(
                        epochs,
                        lower_bound,
                        upper_bound,
                        alpha=0.2,
                        label=f"{act_name} (Std Error)",
                    )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Training Loss (MAE)")
            ax.set_title(
                f"Average Training Loss Curves per Activation Function (Mean ± Std Error)\nDataset: {dataset_name}"
            )
            ax.legend(title="Activation Function (Metric Type)")
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, f"{dataset_name}_avg_train_loss_curves_aggregated.png"
                )
            )
            plt.close()
        else:
            print("Skipping Average Training Loss Curves plot: No aggregated data.")
    except Exception as e:
        print(f"Error creating Average Training Loss Curves plot: {e}")
        plt.close()
