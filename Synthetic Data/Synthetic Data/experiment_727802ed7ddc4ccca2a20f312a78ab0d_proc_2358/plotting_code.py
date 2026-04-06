import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")


# Helper function to extract average epoch losses from the structured data
def get_average_epoch_losses(loss_data_list, epochs):
    epoch_losses = {epoch: [] for epoch in range(epochs)}
    for entry in loss_data_list:
        if (
            entry["epoch"] != -1
        ):  # Exclude final validation losses, only per-epoch training losses
            epoch_losses[entry["epoch"]].append(entry["loss"])
    # Calculate mean for each epoch, handling cases where an epoch might have no data (though unlikely here)
    avg_epoch_losses = [
        np.mean(epoch_losses[epoch]) if epoch_losses[epoch] else 0
        for epoch in range(epochs)
    ]
    return avg_epoch_losses


# Stage names for MAECMPR plots for better readability
stage_names = ["Healthy", "Pre-diabetes", "Medication-controlled", "Insulin-dependent"]

# Load experiment data
experiment_data = None
try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")

if experiment_data:
    # --- Activation Function Tuning Plots ---
    act_tuning_dataset_key = "synthetic_t2dm_data"
    act_tuning_data = experiment_data["activation_function_tuning"][
        act_tuning_dataset_key
    ]
    activation_functions = list(act_tuning_data["losses"]["train"].keys())

    # 1. Activation Function Tuning: Average Training MAE Loss per Epoch
    try:
        plt.figure(figsize=(10, 6))
        for act_name in activation_functions:
            train_losses = act_tuning_data["losses"]["train"][act_name]
            avg_train_losses = get_average_epoch_losses(
                train_losses, epochs=20
            )  # Assuming 20 epochs
            plt.plot(range(1, 21), avg_train_losses, label=act_name)
        plt.title(
            "Activation Function Tuning: Average Training MAE Loss per Epoch\n(Dataset: Synthetic T2DM Data)"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Average Training MAE Loss")
        plt.legend(title="Activation Function")
        plt.grid(True)
        plt.savefig(
            os.path.join(
                working_dir,
                f"activation_tuning_train_loss_curves_{act_tuning_dataset_key}.png",
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating activation tuning train loss curves plot: {e}")
        plt.close()

    # 2. Activation Function Tuning: Overall Average MAE on Validation Set
    try:
        avg_maes_act = {
            act_name: act_tuning_data["overall_avg_mae_all_models"][act_name]
            for act_name in activation_functions
        }
        labels_act = list(avg_maes_act.keys())
        values_act = list(avg_maes_act.values())

        plt.figure(figsize=(10, 6))
        plt.bar(labels_act, values_act, color="skyblue")
        plt.title(
            "Activation Function Tuning: Overall Average MAE on Validation Set\n(Dataset: Synthetic T2DM Data)"
        )
        plt.xlabel("Activation Function")
        plt.ylabel("Overall Average MAE")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(
            os.path.join(
                working_dir,
                f"activation_tuning_overall_mae_{act_tuning_dataset_key}.png",
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating activation tuning overall MAE plot: {e}")
        plt.close()

    # 3. Activation Function Tuning: MAECMPR per Stage
    try:
        maecmpr_data_act = act_tuning_data["maecmpr_per_stage"]
        num_stages = len(stage_names)
        bar_width = 0.15
        index = np.arange(num_stages)

        plt.figure(figsize=(12, 7))
        for i, act_name in enumerate(activation_functions):
            plt.bar(
                index + i * bar_width,
                maecmpr_data_act[act_name],
                bar_width,
                label=act_name,
            )

        plt.title(
            "Activation Function Tuning: MAECMPR per T2DM Stage\n(Dataset: Synthetic T2DM Data)"
        )
        plt.xlabel("T2DM Stage")
        plt.ylabel("MAECMPR")
        plt.xticks(index + bar_width * (len(activation_functions) - 1) / 2, stage_names)
        plt.legend(title="Activation Function")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(
            os.path.join(
                working_dir,
                f"activation_tuning_maecmpr_per_stage_{act_tuning_dataset_key}.png",
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating activation tuning MAECMPR plot: {e}")
        plt.close()

    # 4. Activation Function Tuning: Validation Predictions vs. Ground Truth (Modality 0)
    try:
        # Identify the best activation function based on overall MAE
        best_act = min(avg_maes_act, key=avg_maes_act.get)

        # Select predictions and ground truth for the best activation function and modality 0
        # `val_predictions` and `val_ground_truth` are lists of numpy arrays, one for each modality.
        predictions_mod0 = act_tuning_data["val_predictions"][best_act][0].flatten()
        ground_truth_mod0 = act_tuning_data["val_ground_truth"][best_act][0].flatten()

        plt.figure(figsize=(8, 8))
        plt.scatter(ground_truth_mod0, predictions_mod0, alpha=0.3, s=10)
        # Plot y=x line for ideal prediction
        min_val = min(ground_truth_mod0.min(), predictions_mod0.min())
        max_val = max(ground_truth_mod0.max(), predictions_mod0.max())
        plt.plot(
            [min_val, max_val], [min_val, max_val], "--r", label="Ideal Prediction"
        )
        plt.title(
            f"Activation Function Tuning: Predictions vs. Ground Truth for Modality 0\n(Config: {best_act} - Dataset: Synthetic T2DM Data)"
        )
        plt.xlabel("Ground Truth Values")
        plt.ylabel("Predicted Values")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")  # Ensure axes are scaled equally
        plt.savefig(
            os.path.join(
                working_dir,
                f"activation_tuning_predictions_vs_gt_mod0_{best_act.lower()}_{act_tuning_dataset_key}.png",
            )
        )
        plt.close()
    except Exception as e:
        print(
            f"Error creating activation tuning predictions vs. ground truth plot: {e}"
        )
        plt.close()

    # --- Optimizer Choice Ablation Plots ---
    opt_ablation_dataset_key = "synthetic_t2dm_data"
    opt_ablation_data = experiment_data["optimizer_choice_ablation"][
        opt_ablation_dataset_key
    ]
    optimizers = list(opt_ablation_data["losses"]["train"].keys())

    # 5. Optimizer Choice Ablation: Average Training MAE Loss per Epoch
    try:
        plt.figure(figsize=(10, 6))
        for opt_name in optimizers:
            train_losses_opt = opt_ablation_data["losses"]["train"][opt_name]
            avg_train_losses_opt = get_average_epoch_losses(
                train_losses_opt, epochs=20
            )  # Assuming 20 epochs
            plt.plot(range(1, 21), avg_train_losses_opt, label=opt_name)
        plt.title(
            "Optimizer Choice Ablation: Average Training MAE Loss per Epoch\n(Dataset: Synthetic T2DM Data)"
        )
        plt.xlabel("Epoch")
        plt.ylabel("Average Training MAE Loss")
        plt.legend(title="Optimizer")
        plt.grid(True)
        plt.savefig(
            os.path.join(
                working_dir,
                f"optimizer_ablation_train_loss_curves_{opt_ablation_dataset_key}.png",
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating optimizer ablation train loss curves plot: {e}")
        plt.close()

    # 6. Optimizer Choice Ablation: Overall Average MAE on Validation Set
    try:
        avg_maes_opt = {
            opt_name: opt_ablation_data["overall_avg_mae_all_models"][opt_name]
            for opt_name in optimizers
        }
        labels_opt = list(avg_maes_opt.keys())
        values_opt = list(avg_maes_opt.values())

        plt.figure(figsize=(10, 6))
        plt.bar(labels_opt, values_opt, color="lightgreen")
        plt.title(
            "Optimizer Choice Ablation: Overall Average MAE on Validation Set\n(Dataset: Synthetic T2DM Data)"
        )
        plt.xlabel("Optimizer")
        plt.ylabel("Overall Average MAE")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(
            os.path.join(
                working_dir,
                f"optimizer_ablation_overall_mae_{opt_ablation_dataset_key}.png",
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating optimizer ablation overall MAE plot: {e}")
        plt.close()

    # 7. Optimizer Choice Ablation: MAECMPR per Stage
    try:
        maecmpr_data_opt = opt_ablation_data["maecmpr_per_stage"]
        num_stages = len(stage_names)
        bar_width = 0.2
        index = np.arange(num_stages)

        plt.figure(figsize=(12, 7))
        for i, opt_name in enumerate(optimizers):
            plt.bar(
                index + i * bar_width,
                maecmpr_data_opt[opt_name],
                bar_width,
                label=opt_name,
            )

        plt.title(
            "Optimizer Choice Ablation: MAECMPR per T2DM Stage\n(Dataset: Synthetic T2DM Data)"
        )
        plt.xlabel("T2DM Stage")
        plt.ylabel("MAECMPR")
        plt.xticks(index + bar_width * (len(optimizers) - 1) / 2, stage_names)
        plt.legend(title="Optimizer")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.savefig(
            os.path.join(
                working_dir,
                f"optimizer_ablation_maecmpr_per_stage_{opt_ablation_dataset_key}.png",
            )
        )
        plt.close()
    except Exception as e:
        print(f"Error creating optimizer ablation MAECMPR plot: {e}")
        plt.close()

    # 8. Optimizer Choice Ablation: Validation Predictions vs. Ground Truth (Modality 0)
    try:
        # Identify the best optimizer based on overall MAE
        best_opt = min(avg_maes_opt, key=avg_maes_opt.get)

        # Select predictions and ground truth for the best optimizer and modality 0
        predictions_mod0_opt = opt_ablation_data["val_predictions"][best_opt][
            0
        ].flatten()
        ground_truth_mod0_opt = opt_ablation_data["val_ground_truth"][best_opt][
            0
        ].flatten()

        plt.figure(figsize=(8, 8))
        plt.scatter(
            ground_truth_mod0_opt,
            predictions_mod0_opt,
            alpha=0.3,
            s=10,
            color="forestgreen",
        )
        # Plot y=x line for ideal prediction
        min_val_opt = min(ground_truth_mod0_opt.min(), predictions_mod0_opt.min())
        max_val_opt = max(ground_truth_mod0_opt.max(), predictions_mod0_opt.max())
        plt.plot(
            [min_val_opt, max_val_opt],
            [min_val_opt, max_val_opt],
            "--r",
            label="Ideal Prediction",
        )
        plt.title(
            f"Optimizer Choice Ablation: Predictions vs. Ground Truth for Modality 0\n(Config: {best_opt} - Dataset: Synthetic T2DM Data)"
        )
        plt.xlabel("Ground Truth Values")
        plt.ylabel("Predicted Values")
        plt.legend()
        plt.grid(True)
        plt.axis("equal")  # Ensure axes are scaled equally
        plt.savefig(
            os.path.join(
                working_dir,
                f"optimizer_ablation_predictions_vs_gt_mod0_{best_opt.lower()}_{opt_ablation_dataset_key}.png",
            )
        )
        plt.close()
    except Exception as e:
        print(
            f"Error creating optimizer ablation predictions vs. ground truth plot: {e}"
        )
        plt.close()

# The evaluation metrics (e.g., overall average MAE) are printed in the experiment code.
# This plotting script focuses on visualizing them, not re-calculating or printing new metrics.
