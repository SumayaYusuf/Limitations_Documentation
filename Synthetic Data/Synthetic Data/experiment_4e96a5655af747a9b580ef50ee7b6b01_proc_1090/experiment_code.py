import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Setup working directory and device
working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Global dictionary to store experiment data
# Using defaultdict for convenience, but structure adheres to requirements
experiment_data = defaultdict(
    lambda: {
        "metrics": {"train": [], "val": []},
        "losses": {"train": [], "val": []},
        "predictions": [],
        "ground_truth": [],
    }
)


# 1. Synthetic Data Generation
def generate_synthetic_data(
    num_stages,
    num_samples_per_stage,
    num_modalities,
    feature_dim,
    base_noise_std=0.1,
    stage_noise_increment=0.1,
    random_seed=42,
):
    """
    Generates synthetic multi-modal data for different T2DM progression stages.
    Discordance increases with stages by adding more noise to inter-modal relationships.
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    all_stage_data = []

    # Fixed transformation matrices and biases for modalities to maintain underlying structure
    # This ensures there's a base relationship to break down when noise increases
    transformation_matrices = [
        np.random.randn(feature_dim, feature_dim) * 0.5 for _ in range(num_modalities)
    ]
    bias_vectors = [np.random.randn(feature_dim) * 0.1 for _ in range(num_modalities)]

    for s in range(num_stages):
        # Noise level increases with the stage number (s)
        current_noise_std = base_noise_std + s * stage_noise_increment
        stage_data = []  # List of dictionaries, each dict for a participant
        print(f"Generating data for Stage {s}: Noise Std = {current_noise_std:.2f}")

        for _ in range(num_samples_per_stage):
            participant_modalities = {}
            # Common latent factor for a participant, driving correlations between modalities
            latent_z = np.random.randn(feature_dim)

            for m in range(num_modalities):
                # Base features derived from latent_z and a fixed transformation matrix/bias
                base_features = latent_z @ transformation_matrices[m] + bias_vectors[m]

                # Add noise. Noise level increases with stage, simulating discordance.
                noise = np.random.randn(feature_dim) * current_noise_std

                participant_modalities[f"modality_{m}"] = torch.tensor(
                    base_features + noise, dtype=torch.float32
                )
            stage_data.append(participant_modalities)
        all_stage_data.append(stage_data)
    return all_stage_data


# 2. PyTorch Dataset
class CrossModalDataset(Dataset):
    def __init__(
        self,
        data_list,
        target_modality_idx,
        num_modalities,
        feature_dim,
        input_scaler: StandardScaler,
        target_scaler: StandardScaler,
    ):
        self.target_modality_idx = target_modality_idx
        self.num_modalities = num_modalities
        self.feature_dim = feature_dim
        self.input_scaler = input_scaler
        self.target_scaler = target_scaler

        X_list_raw = []
        y_list_raw = []

        for participant_data in data_list:
            input_modalities_raw = []
            for m_idx in range(num_modalities):
                if m_idx == target_modality_idx:
                    y_list_raw.append(participant_data[f"modality_{m_idx}"].numpy())
                else:
                    input_modalities_raw.append(
                        participant_data[f"modality_{m_idx}"].numpy()
                    )

            # Concatenate all other modalities for input X (raw numpy array)
            X_list_raw.append(np.concatenate(input_modalities_raw, axis=0))

        # Scale X and y using the provided scalers
        self.X = torch.tensor(
            self.input_scaler.transform(np.vstack(X_list_raw)), dtype=torch.float32
        )
        self.y = torch.tensor(
            self.target_scaler.transform(np.vstack(y_list_raw)), dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 3. Model Definition
class CrossModalPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CrossModalPredictor, self).__init__()
        # Using a simple linear model as a baseline for cross-modal prediction
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# 4. Training and Evaluation Function
def train_and_evaluate_stage(
    stage_data_raw,
    stage_idx,
    num_modalities,
    feature_dim,
    learning_rate,
    num_epochs,
    batch_size,
):

    stage_maecmpr_components = (
        []
    )  # To store MAE for each cross-modal prediction within this stage

    # 1. Fit individual scalers for EACH modality across all samples in this stage.
    # This ensures consistent scaling whether a modality is used as input or target.
    individual_modality_scalers = {}
    for m_idx in range(num_modalities):
        modality_name = f"modality_{m_idx}"
        modality_features_raw = np.vstack(
            [p_data[modality_name].numpy() for p_data in stage_data_raw]
        )
        scaler = StandardScaler()
        scaler.fit(modality_features_raw)
        individual_modality_scalers[modality_name] = scaler

    # 2. Loop through each modality, treating it as the target to be predicted by others
    for target_modality_idx in range(num_modalities):
        print(f"  Stage {stage_idx}, Predicting Modality {target_modality_idx}...")

        # Prepare raw concatenated input (X) and raw target (y) for this specific prediction task
        current_X_raw_concat_list = []
        current_y_raw_list = []
        for participant_data in stage_data_raw:
            input_modalities_raw_for_concat = []
            for m_idx in range(num_modalities):
                if m_idx == target_modality_idx:
                    current_y_raw_list.append(
                        participant_data[f"modality_{m_idx}"].numpy()
                    )
                else:
                    input_modalities_raw_for_concat.append(
                        participant_data[f"modality_{m_idx}"].numpy()
                    )
            current_X_raw_concat_list.append(
                np.concatenate(input_modalities_raw_for_concat, axis=0)
            )

        # Fit a StandardScaler for the CONCATENATED input X for this specific prediction task
        input_scaler_for_task = StandardScaler()
        input_scaler_for_task.fit(np.vstack(current_X_raw_concat_list))

        # Get the target scaler from the pre-fitted individual modality scalers
        target_scaler_for_task = individual_modality_scalers[
            f"modality_{target_modality_idx}"
        ]

        # Create dataset and dataloader with the appropriate scalers
        dataset = CrossModalDataset(
            stage_data_raw,
            target_modality_idx,
            num_modalities,
            feature_dim,
            input_scaler=input_scaler_for_task,
            target_scaler=target_scaler_for_task,
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        input_dim = (num_modalities - 1) * feature_dim
        output_dim = feature_dim

        model = CrossModalPredictor(input_dim, output_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.L1Loss()  # MAE Loss is directly used for training

        # Training loop for this specific cross-modal predictor
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for X_batch, y_batch in dataloader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * X_batch.size(0)

            avg_train_loss = total_loss / len(dataset)
            # Store training loss for each target_modality_idx model
            experiment_data[f"stage_{stage_idx}_modality_{target_modality_idx}"][
                "losses"
            ]["train"].append(avg_train_loss)

        # Evaluation after training
        model.eval()
        with torch.no_grad():
            X_all, y_all = dataset.X.to(device), dataset.y.to(device)
            predictions = model(X_all)

            # CRITICAL: Inverse transform predictions and ground truth to get MAE in original scale
            predictions_unscaled = torch.tensor(
                target_scaler_for_task.inverse_transform(predictions.cpu().numpy()),
                dtype=torch.float32,
            ).to(device)
            y_all_unscaled = torch.tensor(
                target_scaler_for_task.inverse_transform(y_all.cpu().numpy()),
                dtype=torch.float32,
            ).to(device)

            mae_residual = torch.mean(
                torch.abs(predictions_unscaled - y_all_unscaled)
            ).item()
            stage_maecmpr_components.append(
                mae_residual
            )  # Collect MAE for this specific prediction task

            # Store predictions and ground truth for this target_modality_idx
            experiment_data[f"stage_{stage_idx}_modality_{target_modality_idx}"][
                "predictions"
            ].append(predictions_unscaled.cpu().numpy())
            experiment_data[f"stage_{stage_idx}_modality_{target_modality_idx}"][
                "ground_truth"
            ].append(y_all_unscaled.cpu().numpy())
            experiment_data[f"stage_{stage_idx}_modality_{target_modality_idx}"][
                "metrics"
            ]["val"].append(
                mae_residual
            )  # Use MAE as val metric for individual model

    # Calculate the overall MAECMPR for the stage by averaging collected MAEs
    maecmpr_for_stage = np.mean(stage_maecmpr_components)

    # Store the overall MAECMPR for the stage in the main experiment_data['stage_s'] entry
    experiment_data[f"stage_{stage_idx}"]["metrics"]["val"].append(maecmpr_for_stage)

    return maecmpr_for_stage


# Main execution logic starts here (global scope)

# Hyperparameters
NUM_STAGES = 4  # e.g., Healthy, Pre-diabetes, Medication-controlled, Insulin-dependent
NUM_SAMPLES_PER_STAGE = 200  # Number of participants per stage
NUM_MODALITIES = 5  # e.g., Cardiac, Metabolic, Retinal, Environmental, Clinical
FEATURE_DIM = 15  # Number of features per modality
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
BATCH_SIZE = 32

print("Generating synthetic data...")
all_stage_data = generate_synthetic_data(
    NUM_STAGES, NUM_SAMPLES_PER_STAGE, NUM_MODALITIES, FEATURE_DIM
)
print("Synthetic data generation complete.")

print("\nStarting cross-modal prediction and MAECMPR calculation...")
maecmpr_results_per_stage = []

for s_idx, stage_data in enumerate(all_stage_data):
    print(f"\nProcessing Stage {s_idx}...")
    # Train and evaluate all cross-modal predictors for the current stage
    stage_maecmpr = train_and_evaluate_stage(
        stage_data,
        s_idx,
        NUM_MODALITIES,
        FEATURE_DIM,
        LEARNING_RATE,
        NUM_EPOCHS,
        BATCH_SIZE,
    )
    maecmpr_results_per_stage.append(stage_maecmpr)
    # Print the MAECMPR for the stage as 'validation_loss' as per instructions
    print(f"Epoch {s_idx}: validation_loss = {stage_maecmpr:.4f}")

print("\nFinal MAECMPR results per stage:")
for s_idx, maecmpr in enumerate(maecmpr_results_per_stage):
    print(f"  Stage {s_idx}: MAECMPR = {maecmpr:.4f}")

# Save all experiment data to the working directory
np.save(os.path.join(working_dir, "experiment_data.npy"), dict(experiment_data))
print(f"\nExperiment data saved to {os.path.join(working_dir, 'experiment_data.npy')}")

# Print overall stage-level MAECMPR values stored in experiment_data for verification
print("\nOverall MAECMPR per stage from experiment_data (should match above):")
for s_idx in range(NUM_STAGES):
    if (
        f"stage_{s_idx}" in experiment_data
        and experiment_data[f"stage_{s_idx}"]["metrics"]["val"]
    ):
        # The MAECMPR for the stage is the last validation metric appended to its entry
        print(
            f"  Stage {s_idx}: MAECMPR = {experiment_data[f'stage_{s_idx}']['metrics']['val'][-1]:.4f}"
        )
