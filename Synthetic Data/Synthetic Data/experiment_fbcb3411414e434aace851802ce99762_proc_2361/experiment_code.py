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
# Restructured for the input normalization strategy ablation
experiment_data = {
    "input_normalization_strategy_ablation": {
        "synthetic_t2dm_data": {
            "maecmpr_per_stage": {},  # Dictionary: strategy_name -> list of MAECMPR values per stage
            "metrics": {
                "train": {},
                "val": {},
            },  # Dictionary: strategy_name -> list of general aggregated metrics
            "losses": {
                "train": {},
                "val": {},
            },  # Dictionary: strategy_name -> list of per-model, per-epoch losses
            "stage_labels": [],  # Original stage labels for validation set participants (common across runs)
            "val_predictions": {},  # Dictionary: strategy_name -> list of numpy arrays (predictions per modality for validation set)
            "val_ground_truth": {},  # Dictionary: strategy_name -> list of numpy arrays (ground truth per modality for validation set)
            "overall_avg_mae_all_models": {},  # Dictionary: strategy_name -> overall average MAE across all models
        }
    }
}


# --- Synthetic Data Generation ---
def generate_synthetic_t2dm_data(
    num_participants_per_stage=200,
    num_modalities=4,
    num_features_per_modality=5,
    random_seed=42,
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

    # Return raw data; normalization will be handled separately
    return all_data, all_stage_labels, num_modalities, num_features_per_modality


# --- Data Normalization Function ---
def normalize_data(data, strategy="min_max", stats=None):
    """
    Applies normalization to the data based on the specified strategy.
    If stats are not provided, they are calculated from the current data.
    If stats are provided, they are used to normalize the data (e.g., for validation/test sets).
    """
    if strategy == "none":
        return data, {"min": None, "max": None, "mean": None, "std": None}

    if stats is None:  # Calculate stats from the current data (e.g., training data)
        if strategy == "min_max":
            min_vals = data.min(axis=0)
            max_vals = data.max(axis=0)
            range_vals = max_vals - min_vals
            # Prevent division by zero for features with zero range (constant values)
            range_vals[range_vals == 0] = 1
            normalized_data = (data - min_vals) / range_vals
            return normalized_data, {"min": min_vals, "max": max_vals}
        elif strategy == "z_score":
            mean_vals = data.mean(axis=0)
            std_vals = data.std(axis=0)
            # Prevent division by zero for features with zero std (constant values)
            std_vals[std_vals == 0] = 1
            normalized_data = (data - mean_vals) / std_vals
            return normalized_data, {"mean": mean_vals, "std": std_vals}
        else:
            raise ValueError(f"Unknown normalization strategy: {strategy}")
    else:  # Apply normalization using provided stats (e.g., to validation data)
        if strategy == "min_max":
            min_vals = stats["min"]
            max_vals = stats["max"]
            range_vals = max_vals - min_vals
            range_vals[range_vals == 0] = 1
            normalized_data = (data - min_vals) / range_vals
            return normalized_data, stats
        elif strategy == "z_score":
            mean_vals = stats["mean"]
            std_vals = stats["std"]
            std_vals[std_vals == 0] = 1
            normalized_data = (data - mean_vals) / std_vals
            return normalized_data, stats
        else:
            raise ValueError(f"Unknown normalization strategy: {strategy}")


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
    model, dataloader, criterion, optimizer, epochs, model_idx, strategy_name
):  # Changed act_name to strategy_name
    model.train()
    # Ensure the list for this strategy name exists in experiment_data
    if (
        strategy_name
        not in experiment_data["input_normalization_strategy_ablation"][
            "synthetic_t2dm_data"
        ]["losses"]["train"]
    ):
        experiment_data["input_normalization_strategy_ablation"]["synthetic_t2dm_data"][
            "losses"
        ]["train"][strategy_name] = []

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
        # Store loss for the current strategy name
        experiment_data["input_normalization_strategy_ablation"]["synthetic_t2dm_data"][
            "losses"
        ]["train"][strategy_name].append(
            {"model_idx": model_idx, "epoch": epoch, "loss": epoch_loss}
        )


# --- Evaluation Function ---
def evaluate_model(
    model, dataloader, criterion, model_idx, strategy_name
):  # Changed act_name to strategy_name
    model.eval()
    total_loss = 0.0
    predictions_list = []
    ground_truth_list = []

    # Ensure the list for this strategy name exists in experiment_data
    if (
        strategy_name
        not in experiment_data["input_normalization_strategy_ablation"][
            "synthetic_t2dm_data"
        ]["losses"]["val"]
    ):
        experiment_data["input_normalization_strategy_ablation"]["synthetic_t2dm_data"][
            "losses"
        ]["val"][strategy_name] = []

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
    # Store loss for the current strategy name
    experiment_data["input_normalization_strategy_ablation"]["synthetic_t2dm_data"][
        "losses"
    ]["val"][strategy_name].append(
        {"model_idx": model_idx, "epoch": -1, "loss": avg_loss}
    )  # -1 for final evaluation

    return avg_loss, np.concatenate(predictions_list), np.concatenate(ground_truth_list)


# --- Main Experiment Logic ---
def run_experiment():
    # --- Generate RAW data once for all ablation runs ---
    raw_data, stage_labels, num_modalities, num_features_per_modality = (
        generate_synthetic_t2dm_data()
    )
    total_features = raw_data.shape[1]

    # Use a dummy full dataset to get reproducible train/val splits of indices
    full_raw_dataset_indices = list(range(len(raw_data)))
    train_size = int(0.8 * len(full_raw_dataset_indices))
    val_size = len(full_raw_dataset_indices) - train_size
    train_indices_tensor, val_indices_tensor = random_split(
        full_raw_dataset_indices, [train_size, val_size]
    )
    train_indices = train_indices_tensor.indices  # Get actual indices
    val_indices = val_indices_tensor.indices  # Get actual indices

    print(
        f"Total participants: {len(raw_data)}, Train: {len(train_indices)}, Val: {len(val_indices)}"
    )
    print(
        f"Number of modalities: {num_modalities}, Features per modality: {num_features_per_modality}"
    )

    # Store validation set's original stage labels once for MAECMPR calculation
    val_stage_labels_for_maecmpr = stage_labels[val_indices]
    experiment_data["input_normalization_strategy_ablation"]["synthetic_t2dm_data"][
        "stage_labels"
    ] = val_stage_labels_for_maecmpr.tolist()

    # --- Define normalization strategies for ablation ---
    normalization_strategies_to_test = {
        "min_max": "min_max",  # Current baseline
        "z_score": "z_score",
        "none": "none",
    }

    # Use a fixed activation function for this ablation study
    fixed_activation_fn_instance = nn.ReLU()  # Using ReLU as a default

    # --- Ablation Study Loop: Iterate through each normalization strategy ---
    for strategy_name, strategy_type in normalization_strategies_to_test.items():
        print(
            f"\n--- Running experiment with Normalization Strategy: {strategy_name} ---"
        )

        # Get raw data splits based on previously determined indices
        train_raw_data_split = raw_data[train_indices]
        val_raw_data_split = raw_data[val_indices]
        train_stage_labels_split = stage_labels[train_indices]
        val_stage_labels_split = stage_labels[val_indices]

        # Apply normalization based on the current strategy
        # Normalization statistics are calculated ONLY from the training data
        normalized_train_data, normalization_stats = normalize_data(
            train_raw_data_split, strategy_type
        )
        # Apply the same normalization to the validation data using the training stats
        normalized_val_data, _ = normalize_data(
            val_raw_data_split, strategy_type, stats=normalization_stats
        )

        # Create TensorDatasets with the normalized data and their corresponding stage labels
        train_dataset_normalized = TensorDataset(
            torch.tensor(normalized_train_data), torch.tensor(train_stage_labels_split)
        )
        val_dataset_normalized = TensorDataset(
            torch.tensor(normalized_val_data), torch.tensor(val_stage_labels_split)
        )

        val_predictions_per_modality_current_run = []
        val_ground_truth_per_modality_current_run = []
        all_val_model_losses_current_run = []

        # Initialize lists for current strategy within experiment_data's specific keys
        experiment_data["input_normalization_strategy_ablation"]["synthetic_t2dm_data"][
            "metrics"
        ]["train"][strategy_name] = []
        experiment_data["input_normalization_strategy_ablation"]["synthetic_t2dm_data"][
            "metrics"
        ]["val"][strategy_name] = []
        experiment_data["input_normalization_strategy_ablation"]["synthetic_t2dm_data"][
            "val_predictions"
        ][strategy_name] = []
        experiment_data["input_normalization_strategy_ablation"]["synthetic_t2dm_data"][
            "val_ground_truth"
        ][strategy_name] = []

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
            # These datasets use the already normalized full data tensors
            class CrossModalSubDataset(torch.utils.data.Dataset):
                def __init__(self, full_data_tensor, input_indices, target_indices):
                    self.inputs = full_data_tensor[:, input_indices]
                    self.targets = full_data_tensor[:, target_indices]

                def __len__(self):
                    return len(self.inputs)

                def __getitem__(self, idx):
                    return {"inputs": self.inputs[idx], "targets": self.targets[idx]}

            train_cross_modal_dataset = CrossModalSubDataset(
                train_dataset_normalized.tensors[0],
                input_feature_indices,
                target_feature_indices,
            )
            val_cross_modal_dataset = CrossModalSubDataset(
                val_dataset_normalized.tensors[0],
                input_feature_indices,
                target_feature_indices,
            )

            train_dataloader = DataLoader(
                train_cross_modal_dataset, batch_size=32, shuffle=True
            )
            val_dataloader = DataLoader(
                val_cross_modal_dataset, batch_size=32, shuffle=False
            )

            input_dim = len(input_feature_indices)
            output_dim = num_features_per_modality

            # Instantiate model with the fixed activation function instance
            model = CrossModalPredictor(
                input_dim, output_dim, fixed_activation_fn_instance
            ).to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.L1Loss()  # Using L1Loss (MAE) for consistency with MAECMPR

            train_model(
                model,
                train_dataloader,
                criterion,
                optimizer,
                epochs=20,
                model_idx=i,
                strategy_name=strategy_name,
            )
            val_loss, predictions, ground_truth = evaluate_model(
                model,
                val_dataloader,
                criterion,
                model_idx=i,
                strategy_name=strategy_name,
            )

            all_val_model_losses_current_run.append(val_loss)
            val_predictions_per_modality_current_run.append(predictions)
            val_ground_truth_per_modality_current_run.append(ground_truth)

            print(
                f"Modality {i+1} (Predicting from others) with {strategy_name}): Validation MAE Loss = {val_loss:.4f}"
            )

        # Store aggregated validation loss for this normalization strategy
        avg_val_loss_all_models = np.mean(all_val_model_losses_current_run)
        experiment_data["input_normalization_strategy_ablation"]["synthetic_t2dm_data"][
            "metrics"
        ]["val"][strategy_name].append(
            {"overall_avg_mae_all_models": avg_val_loss_all_models}
        )
        experiment_data["input_normalization_strategy_ablation"]["synthetic_t2dm_data"][
            "overall_avg_mae_all_models"
        ][strategy_name] = avg_val_loss_all_models

        print(
            f"\nAverage Validation MAE across all {num_modalities} cross-modal models with {strategy_name}: {avg_val_loss_all_models:.4f}"
        )

        # Store predictions and ground truth for this strategy run
        experiment_data["input_normalization_strategy_ablation"]["synthetic_t2dm_data"][
            "val_predictions"
        ][strategy_name] = val_predictions_per_modality_current_run
        experiment_data["input_normalization_strategy_ablation"]["synthetic_t2dm_data"][
            "val_ground_truth"
        ][strategy_name] = val_ground_truth_per_modality_current_run

        # --- Calculate Mean Absolute Cross-Modal Prediction Residual (MAECMPR) ---
        maecmpr_per_stage_list = [[] for _ in range(4)]  # For 4 stages

        for p_idx in range(
            len(val_dataset_normalized)
        ):  # Iterate through each participant in validation set
            stage = val_stage_labels_for_maecmpr[p_idx]

            participant_residuals = []
            for mod_idx in range(num_modalities):
                pred_features = val_predictions_per_modality_current_run[mod_idx][p_idx]
                gt_features = val_ground_truth_per_modality_current_run[mod_idx][p_idx]

                modality_residual = np.mean(np.abs(pred_features - gt_features))
                participant_residuals.append(modality_residual)

            avg_participant_residual = np.mean(participant_residuals)
            maecmpr_per_stage_list[stage].append(avg_participant_residual)

        final_maecmpr_values = [
            np.mean(res_list) if res_list else 0 for res_list in maecmpr_per_stage_list
        ]

        print(
            f"\n--- Mean Absolute Cross-Modal Prediction Residual (MAECMPR) per Stage for {strategy_name} ---"
        )
        stage_names = [
            "Healthy",
            "Pre-diabetes",
            "Medication-controlled",
            "Insulin-dependent",
        ]
        for s_idx, maecmpr_val in enumerate(final_maecmpr_values):
            print(f"Stage {s_idx} ({stage_names[s_idx]}): MAECMPR = {maecmpr_val:.4f}")

        experiment_data["input_normalization_strategy_ablation"]["synthetic_t2dm_data"][
            "maecmpr_per_stage"
        ][strategy_name] = final_maecmpr_values

    # Save all experiment data after tuning loop is complete
    np.save(
        os.path.join(working_dir, "experiment_data.npy"),
        experiment_data,
        allow_pickle=True,
    )
    print(
        f"\nExperiment data saved to {os.path.join(working_dir, 'experiment_data.npy')}"
    )


# Run the ablation experiment
run_experiment()
