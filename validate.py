import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn # Needed for the class definition

# Hyperparameters (Knobs to turn)
HIDDEN_SIZE = 64          # Number of neurons in hidden layers
LEARNING_RATE = 0.001     # How fast the model learns
EPOCHS = 1000             # How many times to loop through the data

# ==========================================
# 0. DEFINE THE MODEL ARCHITECTURE
# ==========================================
# PASTE YOUR MODEL CLASS HERE! 
# It must look EXACTLY like it does in your training file.
# Example:
class IsotopeNet(nn.Module):
    def __init__(self, input_size):
        super(IsotopeNet, self).__init__()
        # Layer 1
        self.fc1 = nn.Linear(input_size, HIDDEN_SIZE)
        self.relu1 = nn.ReLU() # Activation function
        
        # Layer 2
        self.fc2 = nn.Linear(HIDDEN_SIZE, int(HIDDEN_SIZE/2))
        self.relu2 = nn.ReLU()
        
        # Output Layer (No activation, because we want a raw number value)
        self.fc3 = nn.Linear(int(HIDDEN_SIZE/2), 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# NOTE: If you don't paste the class above, the script will fail.

# ==========================================
# 1. SETUP & LOADING ARTIFACTS
# ==========================================
print("Loading preprocessors...")

SCALER_X_PATH = 'scaler_X.pkl'
SCALER_Y_PATH = 'scaler_y.pkl'
MODEL_WEIGHTS_PATH = 'best_model_weights.pth' 

# Load the scalers
preprocessor_X = joblib.load(SCALER_X_PATH)

try:
    scaler_y = joblib.load(SCALER_Y_PATH)
    target_is_scaled = True
    print("Target scaler loaded.")
except:
    print("No target scaler found. Assuming output is raw units.")
    scaler_y = None
    target_is_scaled = False

# ==========================================
# 2. DEFINE THE "VIRTUAL EXPERIMENT"
# ==========================================
conway_experiment = {
    # numeric
    'temp (C)': 25, 
    'Current Density (mA/cm^2)' : 1000, 
    'Initial Concentration' : 0.1, 
    'Target_Isotope_Mass' : 2,
    'Cathode_Loading (mg/cm^2)' : 2,
    'Cathode_electronegativity' : 2.28,
    'Cathode_WF' : 5.65, 
    'Cathode_Valence' : 10,
    'Anode_loading (mg/cm^2)' : 2,
    'Anode_electronegatvity' : 2.28,
    'Anode_WF' : 5.65,
    'Anode_valence' : 10,

    # categorical
    'Cathode_label' : 'Pt',
    'Anode_Label' : 'Pt', 
    'support' : 'N/A', 
    'Membrane' : 'Nafion'    
}

input_df = pd.DataFrame([conway_experiment])

# ==========================================
# 3. PREPROCESSING
# ==========================================
print("Processing inputs...")

try:
    # Transform data to numpy array
    X_processed_numpy = preprocessor_X.transform(input_df)
    
    # CONVERT TO PYTORCH TENSOR
    X_tensor = torch.tensor(X_processed_numpy, dtype=torch.float32)
    
except Exception as e:
    print("\n!!! ERROR IN PREPROCESSING !!!")
    print(f"Error details: {e}")
    print("Hint: Check if your training script used pd.get_dummies() manually.")
    exit()

# ==========================================
# 4. LOAD MODEL & INFERENCE
# ==========================================
print("Loading model weights...")

# Get input shape from the processed data (columns)
input_dim = X_tensor.shape[1]

# INITIALIZE THE MODEL (Using the class you pasted at the top)
# Make sure the class name matches what you pasted (e.g., SimpleNN, Net, Model)
model = IsotopeNet(input_dim) # <--- CHANGE 'SimpleNN' TO YOUR CLASS NAME

# Load the trained weights
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))

# Set to evaluation mode (Freezes Dropout/Batch Norm layers)
model.eval()

print("Running prediction...")
with torch.no_grad(): # Tells PyTorch not to calculate gradients (saves memory)
    y_pred_tensor = model(X_tensor)
    
    # Convert back to numpy
    y_pred_numpy = y_pred_tensor.numpy()

# Inverse Transform
if target_is_scaled and scaler_y is not None:
    separation_factor = scaler_y.inverse_transform(y_pred_numpy)
else:
    separation_factor = y_pred_numpy

final_result = float(separation_factor[0][0])

# ==========================================
# 5. THE COMPARISON REPORT
# ==========================================
print("\n" + "="*40)
print("     VALIDATION RESULTS: CONWAY (1960)")
print("="*40)
print(f"{'Source':<25} | {'Separation Factor (S)':<10}")
print("-" * 40)
print(f"{'Butler-Volmer Eq':<25} | {'3.00':<10}")
print(f"{'Experimental (Truth)':<25} | {'3.90':<10}")
print("-" * 40)
print(f"{'Neural Net Prediction':<25} | {final_result:.4f}")
print("-" * 40)

# Automatic Analysis
print("\nANALYSIS:")
if 2.8 <= final_result <= 3.2:
    print(">> The model behaves like the standard Butler-Volmer equation (Linear Kinetics).")
elif 3.5 <= final_result <= 4.3:
    print(">> SUCCESS! The model captures the non-linear physics/coverage effects.")
    print(">> It is closer to reality than the standard equation.")
else:
    print(">> Result is ambiguous. Check inputs.")
print("="*40 + "\n")