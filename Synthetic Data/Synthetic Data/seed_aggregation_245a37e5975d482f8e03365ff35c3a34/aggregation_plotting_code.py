import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)  # Ensure working directory exists

# Experiment Data Paths provided in the instructions
experiment_data_path_list = [
    "None/experiment_data.npy",
    "experiments/2026-04-05_22-53-06_intermodal_discordance_diabetes_attempt_2/logs/0-run/experiment_results/experiment_e69bcd6720eb44af95532668eaa735f7_proc_1850/experiment_data.npy",
    "experiments/2026-04-05_22-53-06_intermodal_discordance_diabetes_attempt_2/logs/0-run/experiment_results/experiment_5bee923be87c451c95bdacc88a470be7_proc_1849/experiment_data.npy",
]

all_experiment_data = []
for experiment_data_path in experiment_data_path_list:
    try:
        full_path = ""
        # Handle the "None/experiment_data.npy" case as a potential default in working_dir
        if (
            "experiment_data.npy" in experiment_data_path
            and experiment_data_path.startswith("None/")
        ):
            full_path = os.path.join(working_dir, "experiment_data.npy")
            if not os.path.exists(full_path):
                print(
                    f"Warning: Default experiment_data.npy not found at {full_path}. Skipping '{experiment_data_path}'."
                )
                continue
        else:
            # Use os.getenv("AI_SCIENTIST_ROOT") as prefix for other paths
            # os.path.join handles None gracefully, treating it as an empty string.
            root_path = os.getenv(
                "AI_SCIENTIST_ROOT", ""
            )  # Default to empty string if not set
            full_path = os.path.join(root_path, experiment_data_path)

            if not os.path.exists(full_path):
                print(
                    f"Warning: Experiment data path '{full_path}' does not exist. Skipping '{experiment_data_path}'."
                )
                continue

        experiment_data_item = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(experiment_data_item)
    except Exception as e:
        print(
            f"Error loading experiment data from '{experiment_data_path}' (resolved path: '{full_path}'): {e}"
        )

if not all_experiment_data:
    print("No experiment data loaded. Cannot generate plots or print metrics.")

dataset_name = "synthetic_t2dm_data"
stage_names = ["Healthy", "Pre-diabetes", "Medication-controlled", "Insulin-dependent"]

# --- Aggregate Metrics Across All Runs ---
# Initialize aggregation dictionaries
aggregated_overall_mae = {}  # {act_name: [mae_run1, mae_run2, ...]}
aggregated_maecmpr_per_stage = (
    {}
)  # {act_name: {stage_idx: [maecmpr_run1, maecmpr_run2, ...]}}
aggregated_train_losses_raw = {}  # {act_name: {epoch: [loss_run1, loss_run2, ...]}}

# Determine all unique activation functions from the first available experiment data
activation_functions = []
for data_item in all_experiment_data:
    if (
        "activation_function_tuning" in data_item
        and dataset_name in data_item["activation_function_tuning"]
    ):
        act_fns_in_item = (
            data_item["activation_function_tuning"][dataset_name]
            .get("overall_avg_mae_all_models", {})
            .keys()
        )
        if act_fns_in_item:
            activation_functions = list(act_fns_in_item)
            break
if not activation_functions:
    print("No activation functions found in data. Cannot proceed with aggregation.")
    # Exit aggregation process if no activation functions are found.
else:
    for act_name in activation_functions:
        aggregated_overall_mae[act_name] = []
        aggregated_maecmpr_per_stage[act_name] = {
            s_idx: [] for s_idx in range(len(stage_names))
        }
        aggregated_train_losses_raw[act_name] = {}

    for exp_data in all_experiment_data:
        if (
            "activation_function_tuning" in exp_data
            and dataset_name in exp_data["activation_function_tuning"]
        ):
            data_section = exp_data["activation_function_tuning"][dataset_name]

            # Overall Average MAE
            overall_avg_mae = data_section.get("overall_avg_mae_all_models", {})
            for act_name, mae_value in overall_avg_mae.items():
                if act_name in aggregated_overall_mae:
                    aggregated_overall_mae[act_name].append(mae_value)

            # MAECMPR per Stage
            maecmpr_per_stage = data_section.get("maecmpr_per_stage", {})
            for act_name, maecmpr_values in maecmpr_per_stage.items():
                if act_name in aggregated_maecmpr_per_stage:
                    for s_idx, maecmpr_val in enumerate(maecmpr_values):
                        if s_idx < len(stage_names):  # Ensure stage index is valid
                            aggregated_maecmpr_per_stage[act_name][s_idx].append(
                                maecmpr_val
                            )

            # Training Losses
            train_losses = data_section.get("losses", {}).get("train", {})
            for act_name, losses_list in train_losses.items():
                if act_name in aggregated_train_losses_raw:
                    for item in losses_list:
                        epoch = item["epoch"]
                        loss = item["loss"]
                        if epoch not in aggregated_train_losses_raw[act_name]:
                            aggregated_train_losses_raw[act_name][epoch] = []
                        aggregated_train_losses_raw[act_name][epoch].append(loss)

# Calculate means and standard errors for plotting and printing
overall_mae_means = {}
overall_mae_sems = {}  # Standard Error of the Mean

maecmpr_per_stage_means = {}  # {act_name: [mean_s0, mean_s1, ...]}
maecmpr_per_stage_sems = {}  # {act_name: [sem_s0, sem_s1, ...]}

train_losses_agg_data = {}  # {act_name: {epoch: {'mean': float, 'sem': float}}}

num_runs = len(all_experiment_data)  # Number of runs for SEM calculation

if num_runs > 0:
    for act_name in activation_functions:
        # Overall Average MAE
        if aggregated_overall_mae[act_name]:
            overall_mae_means[act_name] = np.mean(aggregated_overall_mae[act_name])
            overall_mae_sems[act_name] = (
                np.std(aggregated_overall_mae[act_name]) / np.sqrt(num_runs)
                if num_runs > 1
                else 0
            )

        # MAECMPR per Stage
        maecmpr_per_stage_means[act_name] = []
        maecmpr_per_stage_sems[act_name] = []
        for s_idx in range(len(stage_names)):
            if aggregated_maecmpr_per_stage[act_name][s_idx]:
                maecmpr_per_stage_means[act_name].append(
                    np.mean(aggregated_maecmpr_per_stage[act_name][s_idx])
                )
                maecmpr_per_stage_sems[act_name].append(
                    np.std(aggregated_maecmpr_per_stage[act_name][s_idx])
                    / np.sqrt(num_runs)
                    if num_runs > 1
                    else 0
                )
            else:
                maecmpr_per_stage_means[act_name].append(
                    np.nan
                )  # Use NaN if no data for this stage/act
                maecmpr_per_stage_sems[act_name].append(np.nan)

        # Training Losses
        train_losses_agg_data[act_name] = {}
        if act_name in aggregated_train_losses_raw:
            epochs = sorted(aggregated_train_losses_raw[act_name].keys())
            for epoch in epochs:
                losses_at_epoch = aggregated_train_losses_raw[act_name][epoch]
                if losses_at_epoch:
                    train_losses_agg_data[act_name][epoch] = {
                        "mean": np.mean(losses_at_epoch),
                        "sem": (
                            np.std(losses_at_epoch) / np.sqrt(num_runs)
                            if num_runs > 1
                            else 0
                        ),
                    }

# --- Print Evaluation Metrics ---
if overall_mae_means:
    print("--- Overall Average MAE Across All Cross-Modal Models (Mean ± SEM) ---")
    for act_name, mae_mean in overall_mae_means.items():
        sem = overall_mae_sems.get(act_name, 0)
        print(
            f"Activation Function: {act_name}, Average MAE: {mae_mean:.4f} ± {sem:.4f}"
        )

if maecmpr_per_stage_means:
    print(
        "\n--- Mean Absolute Cross-Modal Prediction Residual (MAECMPR) per Stage (Mean ± SEM) ---"
    )
    for act_name, maecmpr_means in maecmpr_per_stage_means.items():
        maecmpr_sems = maecmpr_per_stage_sems.get(act_name, [0] * len(stage_names))
        print(f"Activation Function: {act_name}")
        for s_idx, maecmpr_val in enumerate(maecmpr_means):
            sem_val = maecmpr_sems[s_idx]
            if not np.isnan(maecmpr_val):  # Only print if there's valid data
                print(
                    f"  Stage {s_idx} ({stage_names[s_idx]}): MAECMPR = {maecmpr_val:.4f} ± {sem_val:.4f}"
                )
print("\n")  # Add a newline for separation before plots

# --- Plot 1: MAECMPR per T2DM Stage for Different Activation Functions (Mean with Error Bars) ---
try:
    if maecmpr_per_stage_means:
        num_stages = len(stage_names)

        # Filter out activation functions that have no data
        valid_activation_functions = [
            act
            for act in activation_functions
            if any(not np.isnan(m) for m in maecmpr_per_stage_means.get(act, []))
        ]
        num_activations = len(valid_activation_functions)

        if num_activations > 0:
            fig, ax = plt.subplots(figsize=(12, 7))
            bar_width = 0.8 / num_activations
            index = np.arange(num_stages)

            for i, act_name in enumerate(valid_activation_functions):
                means = maecmpr_per_stage_means[act_name]
                sems = maecmpr_per_stage_sems[act_name]

                # Filter out NaN values for plotting
                plot_means = [m for m in means if not np.isnan(m)]
                plot_sems = [s for s, m in zip(sems, means) if not np.isnan(m)]
                plot_index = [idx for idx, m in enumerate(means) if not np.isnan(m)]

                if (
                    plot_means
                ):  # Only plot if there's valid data for this activation function
                    ax.bar(
                        index[plot_index] + i * bar_width,
                        plot_means,
                        bar_width,
                        yerr=plot_sems,
                        capsize=4,
                        label=f"{act_name} (Mean ± SEM)",
                    )

            ax.set_xlabel("T2DM Stage")
            ax.set_ylabel("MAECMPR (Mean ± SEM)")
            ax.set_title(
                f"MAECMPR per T2DM Stage for Different Activation Functions\nDataset: {dataset_name}"
            )
            ax.set_xticks(
                index + bar_width * (num_activations - 1) / 2
            )  # Center xticks
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
            print(
                f"Skipping MAECMPR plot: No valid data for any activation function found."
            )
except Exception as e:
    print(f"Error creating aggregated MAECMPR plot: {e}")
    plt.close()

# --- Plot 2: Overall Average MAE Across All Cross-Modal Models per Activation Function (Mean with Error Bars) ---
try:
    if overall_mae_means:
        # Filter out activation functions that have no data
        plot_activation_functions = [
            act for act in activation_functions if act in overall_mae_means
        ]
        plot_mae_values = [
            overall_mae_means[act_name] for act_name in plot_activation_functions
        ]
        plot_mae_sems = [
            overall_mae_sems[act_name] for act_name in plot_activation_functions
        ]

        if plot_activation_functions:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(
                plot_activation_functions,
                plot_mae_values,
                yerr=plot_mae_sems,
                capsize=5,
                color="skyblue",
                label="Mean ± SEM",
            )
            ax.set_xlabel("Activation Function")
            ax.set_ylabel("Average MAE (Mean ± SEM)")
            ax.set_title(
                f"Overall Average MAE Across All Cross-Modal Models per Activation Function\nDataset: {dataset_name}"
            )
            ax.legend()
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, f"{dataset_name}_overall_avg_MAE_aggregated.png"
                )
            )
            plt.close()
        else:
            print(
                f"Skipping Overall Average MAE plot: No valid data for any activation function found."
            )
except Exception as e:
    print(f"Error creating aggregated Overall Average MAE plot: {e}")
    plt.close()

# --- Plot 3: Average Training Loss Curves per Activation Function (Mean with Shaded SEM) ---
try:
    if train_losses_agg_data:
        fig, ax = plt.subplots(figsize=(12, 7))

        # Keep track of colors for fill_between
        colors = plt.cm.get_cmap("tab10", len(activation_functions))

        plotted_any = False
        for i, act_name in enumerate(activation_functions):
            epoch_data = train_losses_agg_data.get(act_name)
            if epoch_data:
                epochs = sorted(epoch_data.keys())
                if epochs:
                    means = [epoch_data[epoch]["mean"] for epoch in epochs]
                    sems = [epoch_data[epoch]["sem"] for epoch in epochs]

                    # Plot mean line
                    ax.plot(epochs, means, label=f"{act_name} (Mean)", color=colors(i))
                    # Plot shaded error region, use the same color as the line
                    ax.fill_between(
                        epochs,
                        np.array(means) - np.array(sems),
                        np.array(means) + np.array(sems),
                        alpha=0.2,
                        color=colors(i),
                        label=f"{act_name} (SEM Range)" if num_runs > 1 else None,
                    )
                    plotted_any = True

        if plotted_any:
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Training Loss (MAE)")
            ax.set_title(
                f"Average Training Loss Curves per Activation Function\nDataset: {dataset_name}"
            )
            # Create a more structured legend to avoid duplicate labels for line and fill
            handles, labels = ax.get_legend_handles_labels()
            # Combine mean and SEM legend entries
            legend_dict = {}
            for handle, label in zip(handles, labels):
                if "(Mean)" in label:
                    act_name = label.replace(" (Mean)", "")
                    legend_dict[act_name] = (
                        [handle]
                        if act_name not in legend_dict
                        else legend_dict[act_name]
                    )
                elif (
                    "(SEM Range)" in label and num_runs > 1
                ):  # Only add SEM to legend if it's meaningful (num_runs > 1)
                    act_name = label.replace(" (SEM Range)", "")
                    if act_name in legend_dict:
                        legend_dict[act_name].append(handle)

            final_handles = []
            final_labels = []
            for act_name in activation_functions:
                if act_name in legend_dict:
                    final_handles.extend(legend_dict[act_name])
                    final_labels.append(f"{act_name} (Mean)")
                    if num_runs > 1 and len(legend_dict[act_name]) > 1:
                        final_labels.append(f"{act_name} (SEM Range)")

            # Reorder handles and labels to interleave mean and SEM for each activation function
            ordered_handles = []
            ordered_labels = []
            for act_name in activation_functions:
                if act_name in legend_dict:
                    ordered_handles.append(legend_dict[act_name][0])  # Mean line
                    ordered_labels.append(f"{act_name} (Mean)")
                    if num_runs > 1 and len(legend_dict[act_name]) > 1:
                        ordered_handles.append(legend_dict[act_name][1])  # SEM fill
                        ordered_labels.append(f"{act_name} (SEM Range)")

            ax.legend(ordered_handles, ordered_labels, title="Activation Function")

            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, f"{dataset_name}_avg_train_loss_curves_aggregated.png"
                )
            )
            plt.close()
        else:
            print(
                f"Skipping Average Training Loss Curves plot: No valid data for any activation function found."
            )
except Exception as e:
    print(f"Error creating aggregated Average Training Loss Curves plot: {e}")
    plt.close()
