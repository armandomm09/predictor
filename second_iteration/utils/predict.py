# predict_match.py
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.optim import Adam
from torchsummary import summary
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

from tba import get_match_series, get_future_match_series

device = 'mps' if torch.mps.is_available() else 'cpu'
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
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.model(x)

# Cargar el modelo entrenado (asegúrate de ajustar el input_dim según tu entrenamiento)
input_dim = 78  # Debe coincidir con el número de features
model = MyNet(input_dim).to(device)
model.load_state_dict(torch.load("second_iteration/model.pt"))
model.eval()

# Suponemos que el scaler se ajustó con los datos de entrenamiento y se guarda
# Aquí se carga o se recrea; para el ejemplo, se crea un scaler dummy:
scaler = StandardScaler()

def fit_scaler():
    data_df = pd.read_csv("second_iteration/clean_data.csv")
    data_df.dropna(inplace=True)
    original_df = data_df.copy()

    x = data_df.drop(["blue_score", "red_score"], axis=1).values
    y = data_df[["blue_score", "red_score"]].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15,)

    
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)



def inferencia(fila: pd.Series, model, scaler, threshold=5.0):
    """
    Realiza inferencia para un match ya jugado.
    Imprime la predicción, los valores reales y si acertó el ganador.
    """
    features = fila.drop(labels=["blue_score", "red_score"]).values.reshape(1, -1)
    features_scaled = scaler.transform(features)
    tensor_features = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        prediction = model(tensor_features)
    predicted_scores = prediction.cpu().numpy().flatten()
    true_scores = fila[["blue_score", "red_score"]].values.astype(np.float32)
    error_abs = np.abs(true_scores - predicted_scores)
    accuracy = 100.0 if (error_abs < threshold).all() else 0.0
    actual_winner = "blue" if true_scores[0] > true_scores[1] else "red"
    pred_winner = "blue" if predicted_scores[0] > predicted_scores[1] else "red"
    print(f"Predicción -> blue_score: {predicted_scores[0]:.2f}, red_score: {predicted_scores[1]:.2f}")
    print(f"Valor real -> blue_score: {true_scores[0]:.2f}, red_score: {true_scores[1]:.2f}")
    print("Win prediction was", actual_winner == pred_winner)
    return actual_winner == pred_winner

def predict_future_match(match_key: str, model, scaler):
    """
    Realiza la predicción para un match que aún no se ha jugado.
    Se obtiene la Series de features con get_future_match_series() y se utiliza el modelo.
    Como no hay score oficial, solo se muestran las predicciones.
    """
    match_series = get_future_match_series(match_key)
    if match_series is None:
        print("No se pudo obtener la información del match.")
        return None
    # Se ignoran los scores reales (son 0) y se predicen
    features = match_series.drop(labels=["blue_score", "red_score"]).values.reshape(1, -1)
    features_scaled = scaler.transform(features)
    tensor_features = torch.tensor(features_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        prediction = model(tensor_features)
    predicted_scores = prediction.cpu().numpy().flatten()
    pred_winner = "blue" if predicted_scores[0] > predicted_scores[1] else "red"
    print(f"Predicción futura -> blue_score: {predicted_scores[0]:.2f}, red_score: {predicted_scores[1]:.2f}")
    print("Predicted winner:", pred_winner)
    return predicted_scores, pred_winner


if __name__ == "__main__":
    fit_scaler()
    # # Ejemplo para un match ya jugado:
    # match_key_played = "2025mxto_qm37"  # Reemplaza por una key existente en tu BD
    # try:
    #     played_series = get_match_series(match_key_played)
    #     print("Match jugado (Series):")
    #     print(played_series)
    #     inferencia(played_series, model, scaler)
    # except Exception as e:
    #     print(e)
    
    # Ejemplo para un match futuro (aún no jugado):
    match_key_future = "2025mxto_f1m2"  
    predict_future_match(match_key_future, model, scaler)
