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
# This will be dynamically populated in run_experiment based on the ablation structure
experiment_data = {}


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


# --- Model Definition (Simple MLP with variable depth) ---
class CrossModalPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, activation_fn, num_hidden_layers=2):
        super(CrossModalPredictor, self).__init__()
        self.activation = activation_fn
        self.layers = nn.ModuleList()

        # Define layer sizes based on depth. Example sizes, can be customized.
        # This structure maintains decreasing hidden layer sizes.
        if num_hidden_layers == 1:
            hidden_dims = [128]
        elif num_hidden_layers == 2:  # Baseline
            hidden_dims = [128, 64]
        elif num_hidden_layers == 3:
            hidden_dims = [128, 64, 32]
        elif num_hidden_layers == 4:
            hidden_dims = [128, 64, 32, 16]
        else:
            raise ValueError(
                f"Unsupported number of hidden layers: {num_hidden_layers}. Choose from 1, 2, 3, 4."
            )

        current_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, h_dim))
            current_dim = h_dim

        self.output_layer = nn.Linear(current_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        x = self.output_layer(x)
        return x


# --- Training Function (modified to return losses) ---
def train_model_and_get_losses(model, dataloader, criterion, optimizer, epochs):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        train_losses.append(epoch_loss)
    return train_losses


# --- Evaluation Function (modified to return losses, predictions, and ground truth) ---
def evaluate_model_and_get_losses(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    predictions_list = []
    ground_truth_list = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            predictions_list.append(outputs.cpu().numpy())
            ground_truth_list.append(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, np.concatenate(predictions_list), np.concatenate(ground_truth_list)


# --- Main Experiment Logic ---
def run_experiment():
    global experiment_data  # Refer to the global experiment_data dictionary

    model_depths_to_test = [1, 2, 3, 4]  # Number of hidden layers
    random_seeds_for_data = [42, 43, 44]  # For three distinct synthetic datasets
    default_activation_fn = (
        nn.ReLU()
    )  # Use ReLU for all depth comparisons in this ablation

    # Initialize the top-level structure for this ablation
    experiment_data["model_depth_ablation"] = {}

    for data_seed in random_seeds_for_data:
        dataset_name = f"synthetic_t2dm_seed_{data_seed}"
        experiment_data["model_depth_ablation"][dataset_name] = {}
        print(f"\n--- Generating and processing data for seed: {data_seed} ---")

        # --- Generate data for current seed and split once ---
        data, stage_labels, num_modalities, num_features_per_modality = (
            generate_synthetic_t2dm_data(random_seed=data_seed)
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
        # Store validation set's original stage labels once for MAECMPR calculation
        experiment_data["model_depth_ablation"][dataset_name][
            "stage_labels"
        ] = val_stage_labels_for_maecmpr.tolist()

        # Define custom dataset for cross-modal tasks (placed here to use generated data split)
        class CrossModalSubDataset(torch.utils.data.Dataset):
            def __init__(self, full_data_tensor, input_indices, target_indices):
                self.inputs = full_data_tensor[:, input_indices]
                self.targets = full_data_tensor[:, target_indices]

            def __len__(self):
                return len(self.inputs)

            def __getitem__(self, idx):
                return {"inputs": self.inputs[idx], "targets": self.targets[idx]}

        for num_layers in model_depths_to_test:
            model_depth_key = f"depth_{num_layers}_layers"
            # Initialize structure for this model depth within the current dataset
            experiment_data["model_depth_ablation"][dataset_name][model_depth_key] = {
                "maecmpr_per_stage": [],
                "metrics": {"train": [], "val": []},
                "losses": {"train": [], "val": []},
                "val_predictions": [],
                "val_ground_truth": [],
                "overall_avg_mae_all_models": 0.0,
            }
            print(
                f"\n--- Running experiment with Model Depth: {num_layers} hidden layers for {dataset_name} ---"
            )

            val_predictions_per_modality_current_run = []
            val_ground_truth_per_modality_current_run = []
            all_val_model_losses_current_run = []

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

                # Instantiate model with the current number of hidden layers
                model = CrossModalPredictor(
                    input_dim,
                    output_dim,
                    default_activation_fn,
                    num_hidden_layers=num_layers,
                ).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                criterion = (
                    nn.L1Loss()
                )  # Using L1Loss (MAE) for consistency with MAECMPR

                # Train model and get losses
                train_losses_epochs = train_model_and_get_losses(
                    model, train_dataloader, criterion, optimizer, epochs=20
                )
                # Evaluate model and get losses, predictions, ground truth
                val_loss, predictions, ground_truth = evaluate_model_and_get_losses(
                    model, val_dataloader, criterion
                )

                # Store per-epoch train losses for this model and depth
                experiment_data["model_depth_ablation"][dataset_name][model_depth_key][
                    "losses"
                ]["train"].extend(
                    [
                        {"model_idx": i, "epoch": epoch, "loss": loss_val}
                        for epoch, loss_val in enumerate(train_losses_epochs)
                    ]
                )
                # Store validation loss for this model and depth
                experiment_data["model_depth_ablation"][dataset_name][model_depth_key][
                    "losses"
                ]["val"].append({"model_idx": i, "epoch": -1, "loss": val_loss})

                all_val_model_losses_current_run.append(val_loss)
                val_predictions_per_modality_current_run.append(predictions)
                val_ground_truth_per_modality_current_run.append(ground_truth)

                print(
                    f"Modality {i+1} (Predicting from others) with {num_layers} layers for {dataset_name}): Validation MAE Loss = {val_loss:.4f}"
                )

            # Store aggregated validation loss for this model depth
            avg_val_loss_all_models = np.mean(all_val_model_losses_current_run)
            experiment_data["model_depth_ablation"][dataset_name][model_depth_key][
                "metrics"
            ]["val"].append({"overall_avg_mae_all_models": avg_val_loss_all_models})
            experiment_data["model_depth_ablation"][dataset_name][model_depth_key][
                "overall_avg_mae_all_models"
            ] = avg_val_loss_all_models

            print(
                f"\nAverage Validation MAE across all {num_modalities} cross-modal models with {num_layers} layers for {dataset_name}: {avg_val_loss_all_models:.4f}"
            )

            # Store predictions and ground truth for this model depth run
            experiment_data["model_depth_ablation"][dataset_name][model_depth_key][
                "val_predictions"
            ] = val_predictions_per_modality_current_run
            experiment_data["model_depth_ablation"][dataset_name][model_depth_key][
                "val_ground_truth"
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
                f"\n--- Mean Absolute Cross-Modal Prediction Residual (MAECMPR) per Stage for {num_layers} layers for {dataset_name} ---"
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

            experiment_data["model_depth_ablation"][dataset_name][model_depth_key][
                "maecmpr_per_stage"
            ] = final_maecmpr_values

    # Save all experiment data after all ablation runs are complete
    np.save(
        os.path.join(working_dir, "experiment_data.npy"),
        experiment_data,
        allow_pickle=True,
    )
    print(
        f"\nExperiment data saved to {os.path.join(working_dir, 'experiment_data.npy')}"
    )


# Run the model depth ablation experiment
run_experiment()
