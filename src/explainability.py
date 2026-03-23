import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import os

def explain_model():
    # 1. Path setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    latent_path = os.path.join(project_root, 'data', 'processed', 'latent_features.csv')
    
    # 2. Prepare data
    df = pd.read_csv(latent_path)
    X = df.drop(columns=['hadm_id', 'los'], errors='ignore').select_dtypes(include=[np.number])
    y = df['los']
    
    # 3. Train model
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
    model.fit(X, y)
    
    # 4. Generate native importance plot
    plt.figure(figsize=(10, 8))
    # 'weight' is the number of times a feature is used to split the data
    xgb.plot_importance(model, max_num_features=20, importance_type='weight')
    plt.title("Top 20 Features Influencing LOS (Native XGBoost)")
    
    print("Displaying feature importance plot...")
    plt.show()

if __name__ == "__main__":
    explain_model()