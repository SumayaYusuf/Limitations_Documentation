import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os

# --- CRITICAL GPU REQUIREMENTS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Setup working directory ---
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# --- Experiment data storage ---
# Restructured to support generalizability across datasets and hyperparameter tuning for activation functions
experiment_data = {
    "generalizability_across_datasets": {
        # This will store results for each synthetic dataset (e.g., "dataset_seed_42", "dataset_seed_100")
        # Each dataset entry will contain the activation_function_tuning results for that specific dataset
    }
}


# --- Synthetic Data Generation ---
def generate_synthetic_t2dm_data(
    num_participants_per_stage=200,
    num_modalities=4,
    num_features_per_modality=5,
    random_seed=42,  # Added random_seed parameter
):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # Modalities: Cardiac (0), Metabolic (1), Retinal (2), Clinical_Env (3)
    # T2DM Stages: Healthy (0), Pre-diabetes (1), Medication-controlled (2), Insulin-dependent (3)

    all_data = []
    all_stage_labels = []

    for stage in range(4):
        for _ in range(num_participants_per_stage):
            participant_data_modalities = []

            # Base features, highly correlated in healthy stage
            # Features follow a general trend across modalities initially
            base_trend = np.random.randn(1) * 2 + 5

            for mod_idx in range(num_modalities):
                modality_features = (
                    base_trend + np.random.randn(num_features_per_modality) * 0.5
                )  # Small base noise

                # Introduce stage-dependent modifications and discordance
                # We want modalities to deviate from this "base_trend" differently across stages
                if stage == 0:  # Healthy - high consistency
                    modality_features += (
                        np.random.randn(num_features_per_modality) * 0.3
                    )  # Very small additional noise
                elif stage == 1:  # Pre-diabetes - metabolic starts to deviate
                    if mod_idx == 1:  # Metabolic
                        modality_features += (
                            np.random.randn(num_features_per_modality) * 1.5 + 2
                        )  # Higher values, more noise, shifted mean
                    else:
                        modality_features += (
                            np.random.randn(num_features_per_modality) * 0.7
                        )
                elif (
                    stage == 2
                ):  # Medication-controlled - cardiac and metabolic more pronounced deviation
                    if mod_idx == 0:  # Cardiac
                        modality_features += (
                            np.random.randn(num_features_per_modality) * 2.0 + 3
                        )
                    elif mod_idx == 1:  # Metabolic
                        modality_features += (
                            np.random.randn(num_features_per_modality) * 2.5 + 4
                        )
                    else:
                        modality_features += (
                            np.random.randn(num_features_per_modality) * 1.0
                        )
                elif (
                    stage == 3
                ):  # Insulin-dependent - all modalities show significant discordance
                    if mod_idx == 0:  # Cardiac
                        modality_features += (
                            np.random.randn(num_features_per_modality) * 3.0 + 5
                        )
                    elif mod_idx == 1:  # Metabolic
                        modality_features += (
                            np.random.randn(num_features_per_modality) * 3.5 + 6
                        )
                    elif mod_idx == 2:  # Retinal
                        modality_features += (
                            np.random.randn(num_features_per_modality) * 2.5 + 4
                        )
                    else:  # Clinical_Env
                        modality_features += (
                            np.random.randn(num_features_per_modality) * 2.0 + 3
                        )

                participant_data_modalities.append(modality_features)

            all_data.append(np.concatenate(participant_data_modalities))
            all_stage_labels.append(stage)

    all_data = np.array(all_data, dtype=np.float32)
    all_stage_labels = np.array(all_stage_labels, dtype=np.int64)

    # Normalize features (min-max normalization across all data)
    min_vals = all_data.min(axis=0)
    max_vals = all_data.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Prevent division by zero for constant features
    normalized_data = (all_data - min_vals) / range_vals

    return normalized_data, all_stage_labels, num_modalities, num_features_per_modality


# --- Model Definition (Simple MLP) ---
class CrossModalPredictor(nn.Module):
    def __init__(
        self, input_dim, output_dim, activation_fn
    ):  # Accept activation_fn instance
        super(CrossModalPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.activation = activation_fn  # Assign the provided activation function
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)  # Use the assigned activation function
        x = self.fc2(x)
        x = self.activation(x)  # Use the assigned activation function
        x = self.fc3(x)
        return x


# --- Training Function ---
def train_model(
    model,
    dataloader,
    criterion,
    optimizer,
    epochs,
    model_idx,
    act_name,
    dataset_results_container,
):  # Added dataset_results_container
    model.train()
    # Ensure the list for this activation function exists in the *current dataset's* results container
    if act_name not in dataset_results_container["losses"]["train"]:
        dataset_results_container["losses"]["train"][act_name] = []

    for epoch in range(epochs):
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        # Store loss for the current activation function in the passed container
        dataset_results_container["losses"]["train"][act_name].append(
            {"model_idx": model_idx, "epoch": epoch, "loss": epoch_loss}
        )


# --- Evaluation Function ---
def evaluate_model(
    model, dataloader, criterion, model_idx, act_name, dataset_results_container
):  # Added dataset_results_container
    model.eval()
    total_loss = 0.0
    predictions_list = []
    ground_truth_list = []

    # Ensure the list for this activation function exists in the *current dataset's* results container
    if act_name not in dataset_results_container["losses"]["val"]:
        dataset_results_container["losses"]["val"][act_name] = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            predictions_list.append(outputs.cpu().numpy())
            ground_truth_list.append(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    # Store loss for the current activation function in the passed container
    dataset_results_container["losses"]["val"][act_name].append(
        {"model_idx": model_idx, "epoch": -1, "loss": avg_loss}
    )

    return avg_loss, np.concatenate(predictions_list), np.concatenate(ground_truth_list)


# --- Main Experiment Logic ---
def run_experiment():
    dataset_seeds = [
        42,
        100,
        200,
    ]  # Random seeds for generating multiple independent datasets

    for dataset_seed in dataset_seeds:
        dataset_key = f"dataset_seed_{dataset_seed}"
        print(
            f"\n--- Running experiments on dataset generated with seed: {dataset_seed} ---"
        )

        # Initialize the structure for this specific dataset within experiment_data
        experiment_data["generalizability_across_datasets"][dataset_key] = {
            "activation_function_tuning": {  # This sub-dict holds the tuning results for this dataset
                "maecmpr_per_stage": {},
                "metrics": {"train": {}, "val": {}},
                "losses": {"train": {}, "val": {}},
                "stage_labels": [],
                "val_predictions": {},
                "val_ground_truth": {},
                "overall_avg_mae_all_models": {},
            }
        }
        # Create a convenient reference to the current dataset's results container
        current_dataset_results = experiment_data["generalizability_across_datasets"][
            dataset_key
        ]["activation_function_tuning"]

        # --- Generate data and split once for all tuning runs for the current dataset ---
        data, stage_labels, num_modalities, num_features_per_modality = (
            generate_synthetic_t2dm_data(random_seed=dataset_seed)
        )
        total_features = data.shape[1]

        dataset = TensorDataset(torch.tensor(data), torch.tensor(stage_labels))
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        print(
            f"Dataset (Seed {dataset_seed}) - Total participants: {len(data)}, Train: {len(train_dataset)}, Val: {len(val_dataset)}"
        )
        print(
            f"Dataset (Seed {dataset_seed}) - Number of modalities: {num_modalities}, Features per modality: {num_features_per_modality}"
        )

        # Store validation set's original stage labels once for MAECMPR calculation for the current dataset
        val_original_indices = val_dataset.indices
        val_stage_labels_for_maecmpr = stage_labels[val_original_indices]
        current_dataset_results["stage_labels"] = val_stage_labels_for_maecmpr.tolist()

        # --- Define activation functions for tuning ---
        activation_functions_to_tune = {
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU(0.01),
            "ELU": nn.ELU(),
            "GELU": nn.GELU(),
            "Tanh": nn.Tanh(),
        }

        # --- Hyperparameter Tuning Loop for Activation Functions (within current dataset) ---
        for act_name, activation_fn_instance in activation_functions_to_tune.items():
            print(
                f"\n--- Running experiment with Activation Function: {act_name} for Dataset Seed {dataset_seed} ---"
            )

            val_predictions_per_modality_current_run = []
            val_ground_truth_per_modality_current_run = []
            all_val_model_losses_current_run = []

            # Initialize lists for current activation within current_dataset_results's specific keys
            current_dataset_results["metrics"]["train"][act_name] = []
            current_dataset_results["metrics"]["val"][act_name] = []
            current_dataset_results["val_predictions"][act_name] = []
            current_dataset_results["val_ground_truth"][act_name] = []
            # 'losses' are handled within train_model/evaluate_model functions, which initialize if needed

            for i in range(num_modalities):
                # Prepare input and target for current modality prediction task
                target_feature_indices = np.arange(
                    i * num_features_per_modality, (i + 1) * num_features_per_modality
                )
                input_feature_indices = np.array(
                    [
                        idx
                        for idx in range(total_features)
                        if idx not in target_feature_indices
                    ]
                )

                # Create custom datasets for this specific cross-modal prediction task
                class CrossModalSubDataset(torch.utils.data.Dataset):
                    def __init__(self, full_data_tensor, input_indices, target_indices):
                        self.inputs = full_data_tensor[:, input_indices]
                        self.targets = full_data_tensor[:, target_indices]

                    def __len__(self):
                        return len(self.inputs)

                    def __getitem__(self, idx):
                        return {
                            "inputs": self.inputs[idx],
                            "targets": self.targets[idx],
                        }

                train_full_data_tensor = train_dataset.dataset.tensors[0][
                    train_dataset.indices
                ]
                val_full_data_tensor = val_dataset.dataset.tensors[0][
                    val_dataset.indices
                ]

                train_cross_modal_dataset = CrossModalSubDataset(
                    train_full_data_tensor,
                    input_feature_indices,
                    target_feature_indices,
                )
                val_cross_modal_dataset = CrossModalSubDataset(
                    val_full_data_tensor, input_feature_indices, target_feature_indices
                )

                train_dataloader = DataLoader(
                    train_cross_modal_dataset, batch_size=32, shuffle=True
                )
                val_dataloader = DataLoader(
                    val_cross_modal_dataset, batch_size=32, shuffle=False
                )

                input_dim = len(input_feature_indices)
                output_dim = num_features_per_modality

                # Instantiate model with the current activation function instance
                model = CrossModalPredictor(
                    input_dim, output_dim, activation_fn_instance
                ).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                criterion = (
                    nn.L1Loss()
                )  # Using L1Loss (MAE) for consistency with MAECMPR

                train_model(
                    model,
                    train_dataloader,
                    criterion,
                    optimizer,
                    epochs=20,
                    model_idx=i,
                    act_name=act_name,
                    dataset_results_container=current_dataset_results,  # Pass the results container
                )
                val_loss, predictions, ground_truth = evaluate_model(
                    model,
                    val_dataloader,
                    criterion,
                    model_idx=i,
                    act_name=act_name,
                    dataset_results_container=current_dataset_results,  # Pass the results container
                )

                all_val_model_losses_current_run.append(val_loss)
                val_predictions_per_modality_current_run.append(predictions)
                val_ground_truth_per_modality_current_run.append(ground_truth)

                print(
                    f"  Modality {i+1} (Predicting from others) with {act_name}, Dataset {dataset_seed}): Validation MAE Loss = {val_loss:.4f}"
                )

            # Store aggregated validation loss for this activation function within the current dataset's results
            avg_val_loss_all_models = np.mean(all_val_model_losses_current_run)
            current_dataset_results["metrics"]["val"][act_name].append(
                {"overall_avg_mae_all_models": avg_val_loss_all_models}
            )
            current_dataset_results["overall_avg_mae_all_models"][
                act_name
            ] = avg_val_loss_all_models

            print(
                f"\n  Average Validation MAE across all {num_modalities} cross-modal models with {act_name} for Dataset {dataset_seed}: {avg_val_loss_all_models:.4f}"
            )

            # Store predictions and ground truth for this activation function run within the current dataset's results
            current_dataset_results["val_predictions"][
                act_name
            ] = val_predictions_per_modality_current_run
            current_dataset_results["val_ground_truth"][
                act_name
            ] = val_ground_truth_per_modality_current_run

            # --- Calculate Mean Absolute Cross-Modal Prediction Residual (MAECMPR) ---
            maecmpr_per_stage_list = [[] for _ in range(4)]  # For 4 stages

            # Use the val_stage_labels_for_maecmpr stored for the current dataset
            for p_idx in range(
                len(val_dataset)
            ):  # Iterate through each participant in validation set
                stage = val_stage_labels_for_maecmpr[
                    p_idx
                ]  # This is correct as it's for the current dataset

                participant_residuals = []
                for mod_idx in range(num_modalities):
                    pred_features = val_predictions_per_modality_current_run[mod_idx][
                        p_idx
                    ]
                    gt_features = val_ground_truth_per_modality_current_run[mod_idx][
                        p_idx
                    ]

                    modality_residual = np.mean(np.abs(pred_features - gt_features))
                    participant_residuals.append(modality_residual)

                avg_participant_residual = np.mean(participant_residuals)
                maecmpr_per_stage_list[stage].append(avg_participant_residual)

            final_maecmpr_values = [
                np.mean(res_list) if res_list else 0
                for res_list in maecmpr_per_stage_list
            ]

            print(
                f"\n  --- Mean Absolute Cross-Modal Prediction Residual (MAECMPR) per Stage for {act_name}, Dataset {dataset_seed} ---"
            )
            stage_names = [
                "Healthy",
                "Pre-diabetes",
                "Medication-controlled",
                "Insulin-dependent",
            ]
            for s_idx, maecmpr_val in enumerate(final_maecmpr_values):
                print(
                    f"  Stage {s_idx} ({stage_names[s_idx]}): MAECMPR = {maecmpr_val:.4f}"
                )

            current_dataset_results["maecmpr_per_stage"][
                act_name
            ] = final_maecmpr_values

    # Save all experiment data after ALL datasets and tuning loops are complete
    np.save(
        os.path.join(working_dir, "experiment_data.npy"),
        experiment_data,
        allow_pickle=True,
    )
    print(
        f"\nExperiment data saved to {os.path.join(working_dir, 'experiment_data.npy')}"
    )


# Run the hyperparameter tuning experiment
run_experiment()
