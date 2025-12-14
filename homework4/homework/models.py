from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Flatten track_left and track_right: each has shape (b, n_track, 2)
        # Combined input size: n_track * 2 * 2 = 40 for n_track=10
        input_size = n_track * 2 * 2
        
        # MLP layers
        hidden_size = 128
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, n_waypoints * 2),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        b = track_left.shape[0]
        
        # Flatten the track boundaries
        track_left_flat = track_left.view(b, -1)  # (b, n_track * 2)
        track_right_flat = track_right.view(b, -1)  # (b, n_track * 2)
        
        # Concatenate left and right tracks
        x = torch.cat([track_left_flat, track_right_flat], dim=1)  # (b, n_track * 2 * 2)
        
        # Pass through MLP
        output = self.mlp(x)  # (b, n_waypoints * 2)
        
        # Reshape to (b, n_waypoints, 2)
        waypoints = output.view(b, self.n_waypoints, 2)
        
        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Query embeddings for waypoints
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        
        # Project track points to d_model dimension
        self.track_proj = nn.Linear(2, d_model)
        
        # Transformer decoder layer for cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        # Output projection to waypoint coordinates
        self.output_proj = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        b = track_left.shape[0]
        
        # Concatenate left and right tracks: (b, n_track * 2, 2)
        track_points = torch.cat([track_left, track_right], dim=1)  # (b, 2 * n_track, 2)
        
        # Project track points to d_model dimension
        track_features = self.track_proj(track_points)  # (b, 2 * n_track, d_model)
        
        # Get query embeddings for waypoints
        query_indices = torch.arange(self.n_waypoints, device=track_left.device)
        queries = self.query_embed(query_indices)  # (n_waypoints, d_model)
        queries = queries.unsqueeze(0).expand(b, -1, -1)  # (b, n_waypoints, d_model)
        
        # Cross-attention: queries attend to track_features
        # transformer_decoder expects (tgt, memory) where:
        # - tgt: query sequence (b, n_waypoints, d_model)
        # - memory: key/value sequence (b, 2 * n_track, d_model)
        decoder_output = self.transformer_decoder(queries, track_features)  # (b, n_waypoints, d_model)
        
        # Project to waypoint coordinates
        waypoints = self.output_proj(decoder_output)  # (b, n_waypoints, 2)
        
        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)
        
        # CNN backbone
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)  # 128 * 1 * 2 = 256
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_waypoints * 2)

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        x = image
        x = (x - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]

        # CNN backbone
        # Input: (b, 3, 96, 128)
        
        # First conv block
        x = nn.functional.relu(self.conv1(x))  # (b, 32, 48, 64)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)  # (b, 32, 24, 32)
        
        # Second conv block
        x = nn.functional.relu(self.conv2(x))  # (b, 64, 12, 16)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)  # (b, 64, 6, 8)
        
        # Third conv block
        x = nn.functional.relu(self.conv3(x))  # (b, 128, 3, 4)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)  # (b, 128, 1, 2)
        
        # Flatten
        x = x.view(x.shape[0], -1)  # (b, 128 * 1 * 2) = (b, 256)
        
        # Fully connected layers
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, p=0.2, training=self.training)
        x = nn.functional.relu(self.fc2(x))
        
        # Output layer: predict n_waypoints * 2 values
        x = self.fc3(x)  # (b, n_waypoints * 2)
        
        # Reshape to (b, n_waypoints, 2)
        waypoints = x.view(x.shape[0], self.n_waypoints, 2)
        
        return waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
