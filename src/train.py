import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

def train_optimized_model():
    # 1. Load the latent features and LOS target
    latent_df = pd.read_csv('data/processed/latent_features.csv')
    
    # X = Latent features, y = LOS
    X = latent_df.drop(columns=['hadm_id', 'los'])
    y = latent_df['los']
    
    # 2. Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Define the Adaptive Optimization (Hyperparameter Grid)
    # This searches for the best parameters to optimize your specific dataset
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    }
    
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror')
    grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='neg_root_mean_squared_error')
    
    print("Starting Adaptive Optimization (GridSearch)...")
    grid_search.fit(X_train, y_train)
    
    # 4. Evaluate the best model
    best_model = grid_search.best_estimator_
    predictions = best_model.predict(X_test)
    
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2 = r2_score(y_test, predictions)
    
    print(f"Optimization Complete.")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Final Model RMSE: {rmse:.2f} days")
    print(f"Final Model R² Score: {r2:.2f}")
import matplotlib.pyplot as plt

def show_training_performance(history):
    plt.style.use('dark_background') # Maintains theme consistency
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    
    ax1.plot(history['loss'], label='Train Loss', color='#3498db')
    ax1.plot(history['val_loss'], label='Val Loss', color='#e67e22')
    ax1.set_title('BiLSTM Training vs Validation Loss')
    ax1.legend()
    
    ax2.plot(history['accuracy'], label='Train Acc', color='#3498db')
    ax2.plot(history['val_accuracy'], label='Val Acc', color='#e67e22')
    ax2.set_title('BiLSTM Training vs Validation Accuracy')
    ax2.legend()
    
    plt.show()

if __name__ == "__main__":
    train_optimized_model()