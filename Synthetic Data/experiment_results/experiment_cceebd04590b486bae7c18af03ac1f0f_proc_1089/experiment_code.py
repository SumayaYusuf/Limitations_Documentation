# Set random seed
import random
import numpy as np
import torch

seed = 2
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

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
experiment_data = {
    "synthetic_t2dm_data": {
        "maecmpr_per_stage": [],  # Mean Absolute Cross-Modal Prediction Residual per stage
        "metrics": {"train": [], "val": []},  # General aggregated metrics
        "losses": {"train": [], "val": []},  # Per-model, per-epoch losses
        "stage_labels": [],  # Original stage labels for validation set participants
    },
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

    # Normalize features (min-max normalization across all data)
    min_vals = all_data.min(axis=0)
    max_vals = all_data.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Prevent division by zero for constant features
    normalized_data = (all_data - min_vals) / range_vals

    return normalized_data, all_stage_labels, num_modalities, num_features_per_modality


# --- Model Definition (Simple MLP) ---
class CrossModalPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CrossModalPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


# --- Training Function ---
def train_model(model, dataloader, criterion, optimizer, epochs, model_idx):
    model.train()
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
        experiment_data["synthetic_t2dm_data"]["losses"]["train"].append(
            {"model_idx": model_idx, "epoch": epoch, "loss": epoch_loss}
        )


# --- Evaluation Function ---
def evaluate_model(model, dataloader, criterion, model_idx):
    model.eval()
    total_loss = 0.0
    predictions_list = []
    ground_truth_list = []

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
    experiment_data["synthetic_t2dm_data"]["losses"]["val"].append(
        {"model_idx": model_idx, "epoch": -1, "loss": avg_loss}
    )  # -1 for final evaluation

    return avg_loss, np.concatenate(predictions_list), np.concatenate(ground_truth_list)


# --- Main Experiment Logic ---
def run_experiment():
    data, stage_labels, num_modalities, num_features_per_modality = (
        generate_synthetic_t2dm_data()
    )
    total_features = data.shape[1]

    # Split data into training and validation sets
    dataset = TensorDataset(torch.tensor(data), torch.tensor(stage_labels))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(
        f"Total participants: {len(data)}, Train: {len(train_dataset)}, Val: {len(val_dataset)}"
    )
    print(
        f"Number of modalities: {num_modalities}, Features per modality: {num_features_per_modality}"
    )

    # Store validation set's original stage labels for MAECMPR calculation
    val_original_indices = val_dataset.indices
    val_stage_labels_for_maecmpr = stage_labels[val_original_indices]
    experiment_data["synthetic_t2dm_data"][
        "stage_labels"
    ] = val_stage_labels_for_maecmpr.tolist()

    val_predictions_per_modality = []
    val_ground_truth_per_modality = []
    all_val_model_losses = []  # To track average validation loss across all models

    for i in range(num_modalities):
        # Prepare input and target for current modality prediction task
        target_feature_indices = np.arange(
            i * num_features_per_modality, (i + 1) * num_features_per_modality
        )
        input_feature_indices = np.array(
            [idx for idx in range(total_features) if idx not in target_feature_indices]
        )

        # Create custom datasets for this specific cross-modal prediction task
        class CrossModalSubDataset(torch.utils.data.Dataset):
            def __init__(self, full_data_tensor, input_indices, target_indices):
                self.inputs = full_data_tensor[:, input_indices]
                self.targets = full_data_tensor[:, target_indices]

            def __len__(self):
                return len(self.inputs)

            def __getitem__(self, idx):
                return {"inputs": self.inputs[idx], "targets": self.targets[idx]}

        train_full_data_tensor = train_dataset.dataset.tensors[0][train_dataset.indices]
        val_full_data_tensor = val_dataset.dataset.tensors[0][val_dataset.indices]

        train_cross_modal_dataset = CrossModalSubDataset(
            train_full_data_tensor, input_feature_indices, target_feature_indices
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

        model = CrossModalPredictor(input_dim, output_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.L1Loss()  # Using L1Loss (MAE) for consistency with MAECMPR

        train_model(
            model, train_dataloader, criterion, optimizer, epochs=20, model_idx=i
        )
        val_loss, predictions, ground_truth = evaluate_model(
            model, val_dataloader, criterion, model_idx=i
        )

        all_val_model_losses.append(val_loss)
        val_predictions_per_modality.append(predictions)
        val_ground_truth_per_modality.append(ground_truth)

        print(
            f"Modality {i+1} (Predicting from others): Validation MAE Loss = {val_loss:.4f}"
        )

    # Store aggregated validation loss
    avg_val_loss_all_models = np.mean(all_val_model_losses)
    experiment_data["synthetic_t2dm_data"]["metrics"]["val"].append(
        {"overall_avg_mae_all_models": avg_val_loss_all_models}
    )
    print(
        f"\nAverage Validation MAE across all {num_modalities} cross-modal models: {avg_val_loss_all_models:.4f}"
    )

    # --- Calculate Mean Absolute Cross-Modal Prediction Residual (MAECMPR) ---
    maecmpr_per_stage_list = [[] for _ in range(4)]  # For 4 stages

    for p_idx in range(
        len(val_dataset)
    ):  # Iterate through each participant in validation set
        stage = val_stage_labels_for_maecmpr[p_idx]

        participant_residuals = []
        for mod_idx in range(num_modalities):
            pred_features = val_predictions_per_modality[mod_idx][p_idx]
            gt_features = val_ground_truth_per_modality[mod_idx][p_idx]

            modality_residual = np.mean(np.abs(pred_features - gt_features))
            participant_residuals.append(modality_residual)

        avg_participant_residual = np.mean(participant_residuals)
        maecmpr_per_stage_list[stage].append(avg_participant_residual)

    final_maecmpr_values = [
        np.mean(res_list) if res_list else 0 for res_list in maecmpr_per_stage_list
    ]

    print("\n--- Mean Absolute Cross-Modal Prediction Residual (MAECMPR) per Stage ---")
    stage_names = [
        "Healthy",
        "Pre-diabetes",
        "Medication-controlled",
        "Insulin-dependent",
    ]
    for s_idx, maecmpr_val in enumerate(final_maecmpr_values):
        print(f"Stage {s_idx} ({stage_names[s_idx]}): MAECMPR = {maecmpr_val:.4f}")

    experiment_data["synthetic_t2dm_data"]["maecmpr_per_stage"] = final_maecmpr_values

    # Save all experiment data
    np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
    print(
        f"\nExperiment data saved to {os.path.join(working_dir, 'experiment_data.npy')}"
    )


run_experiment()
