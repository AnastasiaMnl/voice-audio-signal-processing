import LeastSquaresClassifier
import TrainingMLP
import extract_features
import post_process

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    #Extract features 
    X, y = extract_features.create_dataset("data/train/speech", "data/train/noise")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    #Least Squares
    scaler_ls = StandardScaler()
    X_train_ls = scaler_ls.fit_transform(X_train)
    X_val_ls = scaler_ls.transform(X_val)

    ls_model = LeastSquaresClassifier.LeastSquaresClassifier()
    ls_model.fit(X_train_ls, y_train)
    y_pred_ls = ls_model.predict(X_val_ls)
    acc_ls = accuracy_score(y_val, y_pred_ls)
    print(f"Least Squares Accuracy: {acc_ls:.4f}")

    #MLP 
    model_mlp, scaler_mlp = TrainingMLP.train_mlp(X, y, epochs=15, batch_size=128, lr=1e-3)

    #Prediction with MLP
    test_file = "data/test/S01_U04.CH4.wav"
    frame_hop = 0.01
    features = extract_features.extract_features(test_file, hop_len=frame_hop)
    features_scaled = scaler_mlp.transform(features)
    features_tensor = torch.tensor(features_scaled, dtype=torch.float32)

    model_mlp.eval()
    with torch.no_grad():
        outputs = model_mlp(features_tensor)
        preds = (outputs >= 0.5).int().view(-1).numpy()

    #Post-processing and saving to csv
    df_csv = post_process.post_process_predictions(preds, frame_hop=frame_hop, audiofile_name="S01_U04.CH4")
    df_csv.to_csv("output.csv", index=False)
    print("Results saved in output.csv")
