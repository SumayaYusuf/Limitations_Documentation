import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress specific sklearn warnings related to feature names if they arise
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# --- CRITICAL GPU REQUIREMENTS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Experiment data structure for saving
experiment_data = {
    f"stage_{s}": {
        "maecmpr": [],  # MAECMPR for the entire stage
        "modality_losses": {
            f"modality_{m}_pred_loss": {"train": [], "val": []}
            for m in range(3)  # For 3 modalities
        },
    }
    for s in range(4)  # For 4 stages
}


# --- 1. Synthetic Data Generation ---
def generate_synthetic_data(
    n_samples_per_stage=200, n_modalities=3, features_per_modality=3, n_stages=4
):
    """
    Generates synthetic multi-modal data for different T2DM progression stages.
    Relationships between modalities become noisier and drift more as stages progress.
    """
    all_stage_data = {}
    modality_names = [f"Modality_{i}" for i in range(n_modalities)]

    for stage_idx in range(n_stages):
        current_stage_modalities = {
            name: np.zeros((n_samples_per_stage, features_per_modality))
            for name in modality_names
        }

        # Parameters vary with stage to simulate progression and discordance
        # Higher stage_idx means more severe progression
        base_offset = stage_idx * 2
        noise_multiplier = (
            1.0 + stage_idx * 0.7
        )  # Discordance (noise in relationships) increases with stage
        drift_rate = stage_idx * 0.1  # Baseline values drift as stage progresses

        for i in range(n_samples_per_stage):
            # Modality 0 (e.g., Cardiac) - somewhat base level, slight drift
            current_stage_modalities["Modality_0"][i, 0] = np.random.normal(
                5 + base_offset, 1
            )
            current_stage_modalities["Modality_0"][i, 1] = current_stage_modalities[
                "Modality_0"
            ][i, 0] * 0.8 + np.random.normal(0, 0.2 + noise_multiplier * 0.1)
            current_stage_modalities["Modality_0"][i, 2] = (
                current_stage_modalities["Modality_0"][i, 0] * 0.5
                + current_stage_modalities["Modality_0"][i, 1] * 0.3
                + np.random.normal(0, 0.3 + noise_multiplier * 0.15)
            )

            # Modality 1 (e.g., Metabolic) - depends on Modality 0, with increasing noise and more significant drift
            current_stage_modalities["Modality_1"][i, 0] = current_stage_modalities[
                "Modality_0"
            ][i, 0] * 0.7 + np.random.normal(
                2 + base_offset * 0.5 + drift_rate * 1.5, 0.5 + noise_multiplier * 0.2
            )
            current_stage_modalities["Modality_1"][i, 1] = (
                current_stage_modalities["Modality_0"][i, 1] * 0.6
                + current_stage_modalities["Modality_1"][i, 0] * 0.3
                + np.random.normal(
                    1 + base_offset * 0.8 + drift_rate * 2.5,
                    0.4 + noise_multiplier * 0.25,
                )
            )
            current_stage_modalities["Modality_1"][i, 2] = (
                current_stage_modalities["Modality_0"][i, 2] * 0.4
                + current_stage_modalities["Modality_1"][i, 1] * 0.5
                + np.random.normal(
                    0.5 + base_offset * 1.0 + drift_rate * 3.5,
                    0.6 + noise_multiplier * 0.3,
                )
            )

            # Modality 2 (e.g., Retinal) - depends on Modality 1, potentially most discordant with highest noise and drift
            current_stage_modalities["Modality_2"][i, 0] = current_stage_modalities[
                "Modality_1"
            ][i, 0] * 0.9 + np.random.normal(
                0 + base_offset * 0.2 + drift_rate * 2, 0.3 + noise_multiplier * 0.15
            )
            current_stage_modalities["Modality_2"][i, 1] = (
                current_stage_modalities["Modality_1"][i, 1] * 0.7
                + current_stage_modalities["Modality_2"][i, 0] * 0.2
                + np.random.normal(
                    0 + base_offset * 0.5 + drift_rate * 3, 0.5 + noise_multiplier * 0.2
                )
            )
            current_stage_modalities["Modality_2"][i, 2] = (
                current_stage_modalities["Modality_1"][i, 2] * 0.5
                + current_stage_modalities["Modality_2"][i, 1] * 0.4
                + np.random.normal(
                    0 + base_offset * 0.7 + drift_rate * 4,
                    0.7 + noise_multiplier * 0.35,
                )
            )

        # Store numpy arrays directly
        all_stage_data[f"stage_{stage_idx}"] = {
            name: current_stage_modalities[name] for name in modality_names
        }
    return all_stage_data, modality_names, features_per_modality


# --- 2. Model Definition (Simple Feedforward Neural Network) ---
class CrossModalPredictor(nn.Module):
    """
    A simple feedforward neural network for cross-modal prediction.
    Predicts features of one modality from features of other modalities.
    """

    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(CrossModalPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


# --- 3. Training and Evaluation Functions ---
def train_model(
    model, dataloader, optimizer, criterion, epoch, modality_key, stage_key
):
    """
    Trains the cross-modal predictor for one epoch.
    Records training loss in experiment_data.
    """
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    experiment_data[stage_key]["modality_losses"][modality_key]["train"].append(
        avg_loss
    )
    return avg_loss


def evaluate_model(model, dataloader, criterion, modality_key, stage_key):
    """
    Evaluates the cross-modal predictor on a given dataloader (e.g., validation set).
    Records validation loss and returns predictions and ground truth.
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    avg_loss = total_loss / len(dataloader)
    experiment_data[stage_key]["modality_losses"][modality_key]["val"].append(avg_loss)
    return avg_loss, np.vstack(all_predictions), np.vstack(all_targets)


# --- Main Experiment Logic ---
N_SAMPLES_PER_STAGE = 200
N_MODALITIES = 3
FEATURES_PER_MODALITY = 3
N_STAGES = 4
BATCH_SIZE = 32
N_EPOCHS = 30
LEARNING_RATE = 0.001

print("Generating synthetic data...")
all_stage_data, modality_names, features_per_modality = generate_synthetic_data(
    n_samples_per_stage=N_SAMPLES_PER_STAGE,
    n_modalities=N_MODALITIES,
    features_per_modality=FEATURES_PER_MODALITY,
    n_stages=N_STAGES,
)
print("Synthetic data generation complete.")

overall_stage_maecmpr = {}

for stage_idx in range(N_STAGES):
    print(f"\n--- Processing Stage {stage_idx} ---")
    stage_key = f"stage_{stage_idx}"

    current_stage_all_modalities = {
        name: all_stage_data[stage_key][name] for name in modality_names
    }

    # Split the entire stage data into train and test sets to ensure consistent splits across modality predictions
    train_indices, test_indices = train_test_split(
        np.arange(N_SAMPLES_PER_STAGE), test_size=0.2, random_state=42
    )

    stage_abs_residuals = []

    for target_modality_idx, target_modality_name in enumerate(modality_names):
        print(f"  Predicting {target_modality_name} from others...")
        modality_key_for_loss = f"modality_{target_modality_idx}_pred_loss"

        # Prepare inputs (other modalities) and targets (target_modality)
        input_modalities = [
            name for i, name in enumerate(modality_names) if i != target_modality_idx
        ]

        X_full = np.hstack(
            [current_stage_all_modalities[name] for name in input_modalities]
        )
        y_full = current_stage_all_modalities[target_modality_name]

        # Apply standard scaling based on training data for both input (X) and target (y)
        scaler_X = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_full[train_indices])
        X_test_scaled = scaler_X.transform(X_full[test_indices])

        scaler_y = StandardScaler()
        y_train_scaled = scaler_y.fit_transform(y_full[train_indices])
        y_test_scaled = scaler_y.transform(y_full[test_indices])

        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(
            X_test_tensor, y_test_tensor
        )  # Used for validation loss tracking and final prediction

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Model, Optimizer, Loss
        input_dim = X_train_tensor.shape[1]
        output_dim = y_train_tensor.shape[1]
        model = CrossModalPredictor(input_dim, output_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        for epoch in range(N_EPOCHS):
            train_loss = train_model(
                model,
                train_loader,
                optimizer,
                criterion,
                epoch,
                modality_key_for_loss,
                stage_key,
            )
            # Evaluate on test set to track validation loss, results are not used for MAECMPR directly yet
            val_loss, _, _ = evaluate_model(
                model, test_loader, criterion, modality_key_for_loss, stage_key
            )
            if (epoch + 1) % 10 == 0 or epoch == N_EPOCHS - 1:
                print(
                    f"    Epoch {epoch+1}/{N_EPOCHS}: Train Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}"
                )

        # Final evaluation on the test set to collect predictions for MAECMPR
        _, predictions_scaled, targets_scaled = evaluate_model(
            model, test_loader, criterion, modality_key_for_loss, stage_key
        )

        # Inverse transform to get predictions and targets in original scale for residual calculation
        predictions_original = scaler_y.inverse_transform(predictions_scaled)
        targets_original = scaler_y.inverse_transform(targets_scaled)

        abs_residuals = np.abs(predictions_original - targets_original)
        stage_abs_residuals.append(abs_residuals.flatten())

    # Calculate MAECMPR for the current stage by averaging all collected residuals
    if stage_abs_residuals:
        maecmpr_for_stage = np.mean(np.hstack(stage_abs_residuals))
        overall_stage_maecmpr[stage_key] = maecmpr_for_stage
        experiment_data[stage_key]["maecmpr"].append(maecmpr_for_stage)
        print(f"  MAECMPR for {stage_key}: {maecmpr_for_stage:.4f}")
    else:
        print(f"  No residuals collected for {stage_key}.")

# Print final MAECMPR for all stages
print("\n--- Final MAECMPR Results per Stage ---")
for stage_key, maecmpr_value in overall_stage_maecmpr.items():
    print(f"{stage_key}: {maecmpr_value:.4f}")

# Save experiment data
np.save(os.path.join(working_dir, "experiment_data.npy"), experiment_data)
print(f"\nExperiment data saved to {os.path.join(working_dir, 'experiment_data.npy')}")
