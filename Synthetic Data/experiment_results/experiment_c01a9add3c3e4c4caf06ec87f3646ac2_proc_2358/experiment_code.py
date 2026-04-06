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
# Restructured for the "Robustness to Synthetic Data Characteristics" ablation
ABLATION_KEY = "robustness_to_synthetic_data_characteristics"
experiment_data = {
    ABLATION_KEY: {}
    # Each entry within ABLATION_KEY will be a dictionary for a specific synthetic dataset,
    # containing all metrics, losses, predictions, etc., per activation function.
}


# --- Synthetic Data Generation (Modified for Ablation Study) ---
def generate_synthetic_t2dm_data(
    num_participants_per_stage=200,
    num_modalities=4,
    num_features_per_modality=5,
    random_seed=42,
    base_inter_modality_noise=0.5,  # Base noise added across all modalities
    stage_noise_overrides=None,  # dict: stage -> {mod_idx: noise_mult} to override default
    stage_shift_overrides=None,  # dict: stage -> {mod_idx: shift_val} to override default
    global_noise_multiplier=1.0,  # Overall scale for all random noise
    inter_modality_correlation_strength=1.0,  # How much the base_trend influences modality features
):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    all_data = []
    all_stage_labels = []

    # Define default original stage-specific noise and shifts for reference.
    # These are additional values on top of the base_trend and base_inter_modality_noise.
    # Default structure: {stage_idx: {mod_idx: {'noise': factor, 'shift': value}}}
    default_stage_params = {
        0: {
            0: {"noise": 0.3, "shift": 0},
            1: {"noise": 0.3, "shift": 0},
            2: {"noise": 0.3, "shift": 0},
            3: {"noise": 0.3, "shift": 0},
        },
        1: {
            0: {"noise": 0.7, "shift": 0},
            1: {"noise": 1.5, "shift": 2},
            2: {"noise": 0.7, "shift": 0},
            3: {"noise": 0.7, "shift": 0},
        },
        2: {
            0: {"noise": 2.0, "shift": 3},
            1: {"noise": 2.5, "shift": 4},
            2: {"noise": 1.0, "shift": 0},
            3: {"noise": 1.0, "shift": 0},
        },
        3: {
            0: {"noise": 3.0, "shift": 5},
            1: {"noise": 3.5, "shift": 6},
            2: {"noise": 2.5, "shift": 4},
            3: {"noise": 2.0, "shift": 3},
        },
    }

    for stage in range(4):
        for _ in range(num_participants_per_stage):
            participant_data_modalities = []

            # Base features for correlation across modalities
            base_trend = np.random.randn(1) * 2 + 5
            base_trend_contribution = base_trend * inter_modality_correlation_strength

            for mod_idx in range(num_modalities):
                # Start with scaled base_trend and general inter-modality noise
                modality_features = (
                    base_trend_contribution
                    + np.random.randn(num_features_per_modality)
                    * base_inter_modality_noise
                    * global_noise_multiplier
                )

                # Apply stage and modality specific modifications, using overrides if provided
                current_noise_factor = default_stage_params[stage][mod_idx]["noise"]
                current_shift_val = default_stage_params[stage][mod_idx]["shift"]

                if (
                    stage_noise_overrides
                    and stage in stage_noise_overrides
                    and mod_idx in stage_noise_overrides[stage]
                ):
                    current_noise_factor = stage_noise_overrides[stage][mod_idx]
                if (
                    stage_shift_overrides
                    and stage in stage_shift_overrides
                    and mod_idx in stage_shift_overrides[stage]
                ):
                    current_shift_val = stage_shift_overrides[stage][mod_idx]

                # Add specific noise and shift for the current stage/modality
                modality_features += (
                    np.random.randn(num_features_per_modality)
                    * current_noise_factor
                    * global_noise_multiplier
                    + current_shift_val
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
    def __init__(self, input_dim, output_dim, activation_fn):
        super(CrossModalPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.activation = activation_fn
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x


# --- Training Function (Modified to use global experiment_data) ---
def train_model(
    model, dataloader, criterion, optimizer, epochs, model_idx, act_name, dataset_name
):
    model.train()
    # Ensure the list for this activation function and dataset exists in experiment_data
    if act_name not in experiment_data[ABLATION_KEY][dataset_name]["losses"]["train"]:
        experiment_data[ABLATION_KEY][dataset_name]["losses"]["train"][act_name] = []

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
        # Store loss for the current activation function and dataset
        experiment_data[ABLATION_KEY][dataset_name]["losses"]["train"][act_name].append(
            {"model_idx": model_idx, "epoch": epoch, "loss": epoch_loss}
        )


# --- Evaluation Function (Modified to use global experiment_data) ---
def evaluate_model(model, dataloader, criterion, model_idx, act_name, dataset_name):
    model.eval()
    total_loss = 0.0
    predictions_list = []
    ground_truth_list = []

    # Ensure the list for this activation function and dataset exists in experiment_data
    if act_name not in experiment_data[ABLATION_KEY][dataset_name]["losses"]["val"]:
        experiment_data[ABLATION_KEY][dataset_name]["losses"]["val"][act_name] = []

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
    # Store loss for the current activation function and dataset
    experiment_data[ABLATION_KEY][dataset_name]["losses"]["val"][act_name].append(
        {"model_idx": model_idx, "epoch": -1, "loss": avg_loss}
    )

    return avg_loss, np.concatenate(predictions_list), np.concatenate(ground_truth_list)


# --- Main Experiment Logic ---
def run_experiment():
    # --- Define activation functions for tuning ---
    activation_functions_to_tune = {
        "ReLU": nn.ReLU(),
        "LeakyReLU": nn.LeakyReLU(0.01),
        "ELU": nn.ELU(),
        "GELU": nn.GELU(),
        "Tanh": nn.Tanh(),
    }

    # --- Define different synthetic dataset configurations for the ablation ---
    DATASET_CONFIGS = {
        "baseline_data": {
            "base_inter_modality_noise": 0.5,
            "global_noise_multiplier": 1.0,
            "inter_modality_correlation_strength": 1.0,
            "stage_noise_overrides": None,
            "stage_shift_overrides": None,
            "random_seed": 42,
        },
        "high_noise_low_correlation_data": {
            "base_inter_modality_noise": 0.5,
            "global_noise_multiplier": 2.0,  # Double overall noise
            "inter_modality_correlation_strength": 0.5,  # Weaker inter-modality correlation
            "stage_noise_overrides": None,
            "stage_shift_overrides": None,
            "random_seed": 43,  # Use different seed for distinct data
        },
        "stronger_stage_shifts_data": {
            "base_inter_modality_noise": 0.5,
            "global_noise_multiplier": 1.0,
            "inter_modality_correlation_strength": 1.0,
            "stage_noise_overrides": None,
            "stage_shift_overrides": {  # Custom strong shifts (doubled default values for affected modalities/stages)
                1: {1: 4},  # Metabolic shift in Pre-diabetes from 2 to 4
                2: {
                    0: 6,
                    1: 8,
                },  # Cardiac/Metabolic shifts in Med-controlled from 3,4 to 6,8
                3: {
                    0: 10,
                    1: 12,
                    2: 8,
                    3: 6,
                },  # All shifts in Insulin-dependent from 5,6,4,3 to 10,12,8,6
            },
            "random_seed": 44,  # Use different seed
        },
    }

    for dataset_name, data_gen_params in DATASET_CONFIGS.items():
        print(f"\n======== Starting Experiment for Dataset: {dataset_name} ========")
        # Initialize structure for this dataset within experiment_data
        experiment_data[ABLATION_KEY][dataset_name] = {
            "maecmpr_per_stage": {},
            "metrics": {"train": {}, "val": {}},
            "losses": {"train": {}, "val": {}},
            "stage_labels": [],
            "val_predictions": {},
            "val_ground_truth": {},
            "overall_avg_mae_all_models": {},
        }

        # --- Generate data and split once for all tuning runs for *this specific dataset* ---
        data, stage_labels, num_modalities, num_features_per_modality = (
            generate_synthetic_t2dm_data(**data_gen_params)
        )
        total_features = data.shape[1]

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

        val_original_indices = val_dataset.indices
        val_stage_labels_for_maecmpr = stage_labels[val_original_indices]
        experiment_data[ABLATION_KEY][dataset_name][
            "stage_labels"
        ] = val_stage_labels_for_maecmpr.tolist()

        # --- Hyperparameter Tuning Loop (Activations) for the current dataset ---
        for act_name, activation_fn_instance in activation_functions_to_tune.items():
            print(
                f"\n--- Running experiment with Activation Function: {act_name} on {dataset_name} ---"
            )

            val_predictions_per_modality_current_run = []
            val_ground_truth_per_modality_current_run = []
            all_val_model_losses_current_run = []

            # Ensure lists are initialized for current activation within the current dataset's specific keys
            experiment_data[ABLATION_KEY][dataset_name]["metrics"]["train"][
                act_name
            ] = []
            experiment_data[ABLATION_KEY][dataset_name]["metrics"]["val"][act_name] = []
            experiment_data[ABLATION_KEY][dataset_name]["val_predictions"][
                act_name
            ] = []
            experiment_data[ABLATION_KEY][dataset_name]["val_ground_truth"][
                act_name
            ] = []
            # Losses are handled within train/evaluate functions for specific act_name and dataset_name

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
                    dataset_name=dataset_name,
                )
                val_loss, predictions, ground_truth = evaluate_model(
                    model,
                    val_dataloader,
                    criterion,
                    model_idx=i,
                    act_name=act_name,
                    dataset_name=dataset_name,
                )

                all_val_model_losses_current_run.append(val_loss)
                val_predictions_per_modality_current_run.append(predictions)
                val_ground_truth_per_modality_current_run.append(ground_truth)

                print(
                    f"Modality {i+1} (Predicting from others) with {act_name} on {dataset_name}): Validation MAE Loss = {val_loss:.4f}"
                )

            # Store aggregated validation loss for this activation function and dataset
            avg_val_loss_all_models = np.mean(all_val_model_losses_current_run)
            experiment_data[ABLATION_KEY][dataset_name]["metrics"]["val"][
                act_name
            ].append({"overall_avg_mae_all_models": avg_val_loss_all_models})
            experiment_data[ABLATION_KEY][dataset_name]["overall_avg_mae_all_models"][
                act_name
            ] = avg_val_loss_all_models

            print(
                f"\nAverage Validation MAE across all {num_modalities} cross-modal models with {act_name} on {dataset_name}: {avg_val_loss_all_models:.4f}"
            )

            # Store predictions and ground truth for this activation function run and dataset
            experiment_data[ABLATION_KEY][dataset_name]["val_predictions"][
                act_name
            ] = val_predictions_per_modality_current_run
            experiment_data[ABLATION_KEY][dataset_name]["val_ground_truth"][
                act_name
            ] = val_ground_truth_per_modality_current_run

            # --- Calculate Mean Absolute Cross-Modal Prediction Residual (MAECMPR) ---
            maecmpr_per_stage_list = [[] for _ in range(4)]  # For 4 stages

            for p_idx in range(
                len(val_dataset)
            ):  # Iterate through each participant in validation set
                stage = val_stage_labels_for_maecmpr[p_idx]

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
                f"\n--- Mean Absolute Cross-Modal Prediction Residual (MAECMPR) per Stage for {act_name} on {dataset_name} ---"
            )
            stage_names = [
                "Healthy",
                "Pre-diabetes",
                "Medication-controlled",
                "Insulin-dependent",
            ]
            for s_idx, maecmpr_val in enumerate(final_maecmpr_values):
                print(
                    f"Stage {s_idx} ({stage_names[s_idx]}): MAECMPR = {maecmpr_val:.4f}"
                )

            experiment_data[ABLATION_KEY][dataset_name]["maecmpr_per_stage"][
                act_name
            ] = final_maecmpr_values

    # Save all experiment data after the entire ablation study is complete
    np.save(
        os.path.join(working_dir, "experiment_data.npy"),
        experiment_data,
        allow_pickle=True,
    )
    print(
        f"\nExperiment data saved to {os.path.join(working_dir, 'experiment_data.npy')}"
    )


# Run the ablation study
run_experiment()
