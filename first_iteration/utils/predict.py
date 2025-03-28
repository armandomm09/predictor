# predict_match.py
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torchsummary import summary
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

from tba2 import get_match_series, get_future_match_series

device = "mps" if torch.mps.is_available() else "cpu"
print("Device:", device)


class MyNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.model(x)


input_dim = 42
model = MyNet(input_dim).to(device)
model.load_state_dict(torch.load("first_iteration/model.pt"))
model.eval()

scaler = StandardScaler()


def fit_scaler():
    data_df = pd.read_csv("first_iteration/clean_data.csv")
    data_df.dropna(inplace=True)
    original_df = data_df.copy()

    x = data_df.drop(["blue_score", "red_score"], axis=1).values
    y = data_df[["blue_score", "red_score"]].values

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.15,
    )

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)


def inferencia(fila: pd.Series, model, scaler, threshold=5.0):
    """
    Realiza la inferencia para un match jugado.
    Imprime la predicción, los valores reales y si acertó el ganador.
    Se asume que la Series tiene los inputs en el orden:
    red3_epa, red2_epa, red1_epa, blue3_epa, blue2_epa, blue1_epa,
    red3_total_points, red2_total_points, red1_total_points, blue3_total_points, blue2_total_points, blue1_total_points,
    red3_auto_points, red2_auto_points, red1_auto_points, blue3_auto_points, blue2_auto_points, blue1_auto_points,
    red3_teleop_points, red2_teleop_points, red1_teleop_points, blue3_teleop_points, blue2_teleop_points, blue1_teleop_points,
    red3_endgame_points, red2_endgame_points, red1_endgame_points, blue3_endgame_points, blue2_endgame_points, blue1_endgame_points,
    red3_rank, red2_rank, red1_rank, blue3_rank, blue2_rank, blue1_rank,
    red3_winrate, red2_winrate, red1_winrate, blue3_winrate, blue2_winrate, blue1_winrate,
    blue_score, red_score
    """

    features = fila.drop(labels=["blue_score", "red_score"]).values.reshape(1, -1)
    features_scaled = scaler.transform(features)
    tensor_features = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        prediction = model(tensor_features)
    predicted_scores = prediction.cpu().numpy().flatten()
    true_scores = fila[["blue_score", "red_score"]].values.astype(np.float32)
    error_abs = np.abs(true_scores - predicted_scores)
    actual_winner = "blue" if true_scores[0] > true_scores[1] else "red"
    pred_winner = "blue" if predicted_scores[0] > predicted_scores[1] else "red"
    print(
        f"Predicción -> blue_score: {predicted_scores[0]:.2f}, red_score: {predicted_scores[1]:.2f}"
    )
    print(
        f"Valor real -> blue_score: {true_scores[0]:.2f}, red_score: {true_scores[1]:.2f}"
    )
    print("Win prediction was", actual_winner == pred_winner)
    return actual_winner == pred_winner


def predict_future_match(match_key: str, model, scaler):
    """
    Realiza la predicción para un match que aún no se ha jugado.
    Se obtiene la Series con get_future_match_series() (donde los scores son 0)
    y se predice el resultado.
    """
    match_series = get_future_match_series(match_key)
    if match_series is None:
        print("No se pudo obtener la información del match futuro.")
        return None
    features = match_series.drop(labels=["blue_score", "red_score"]).values.reshape(
        1, -1
    )
    features_scaled = scaler.transform(features)
    tensor_features = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        prediction = model(tensor_features)
    predicted_scores = prediction.cpu().numpy().flatten()
    pred_winner = "blue" if predicted_scores[0] > predicted_scores[1] else "red"
    print(
        f"Predicción futura -> blue_score: {predicted_scores[0]:.2f}, red_score: {predicted_scores[1]:.2f}"
    )
    print("Predicted winner:", pred_winner)
    return predicted_scores, pred_winner


if __name__ == "__main__":
    # Ejemplo para un match jugado:
    # match_key_played = "2025mxto_qm28"  # Reemplaza por una key existente
    # try:
    #     played_series = get_match_series(match_key_played)
    #     print("Match jugado (Series):")
    #     print(played_series)
    #     inferencia(played_series, model, scaler)
    # except Exception as e:
    #     print(e)

    # Ejemplo para un match futuro:
    fit_scaler()
    match_key_future = "2025mxto_f1m2"
    predict_future_match(match_key_future, model, scaler)
