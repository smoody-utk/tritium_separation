import torch 



import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ==========================================
# 1. USER CONFIGURATION (EDIT THIS SECTION)
# ==========================================
DATA_FILE = "my_master_data.xlsx"  # Your Excel file name
MODEL_SAVE_PATH = "tritium_model.pth"

# Define your columns exactly as they appear in your Excel sheet
NUMERIC_INPUTS = [
    'temp (C)', 
    'Current Density (mA/cm^2)', 
    'Initial Concentration', 
    'Carbon_sup_flag',
    'Target_Isotope_Mass',
    'Cathode_Loading (mg/cm^2)',
    'Cathode_electronegativity',
    'Cathode_WF', 
    'Cathode_Valence',
    'Anode_loading (mg/cm^2)',
    'Anode_electronegatvity',
    'Anode_WF',
    'Anode_valence'
]

CATEGORICAL_INPUTS = [
    'Cathode_label',
    'Anode_Label'
]

OUTPUT_TARGET = 'Separation_Factor'

# Hyperparameters (Knobs to turn)
HIDDEN_SIZE = 64          # Number of neurons in hidden layers
LEARNING_RATE = 0.001     # How fast the model learns
EPOCHS = 1000             # How many times to loop through the data

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================
def load_and_process_data():
    try:
        # Try to load real data
        df = pd.read_excel(DATA_FILE)
        print(f"✅ Loaded real data from {DATA_FILE}")
    except FileNotFoundError:
        # Create Dummy Data if file doesn't exist (So you can test this script NOW)
        print("⚠️ File not found. Generating DUMMY data for testing...")
        data = {
            'Cathode_Work_Function': np.random.uniform(4.5, 5.7, 100),
            'Cathode_Electroneg': np.random.uniform(1.9, 2.3, 100),
            'Cathode_Loading': np.random.uniform(0.1, 4.0, 100),
            'Anode_Work_Function': np.full(100, 5.2),
            'Temperature': np.random.uniform(20, 80, 100),
            'Current_Density': np.random.uniform(100, 2000, 100),
            'Target_Isotope_Mass': np.random.choice([2.0, 3.0], 100),
            'Support_Material': np.random.choice(['Carbon', 'None', 'Ti-Mesh'], 100),
            'Separation_Factor': np.random.uniform(2.0, 12.0, 100) # Random targets
        }
        df = pd.DataFrame(data)

    # Separate Inputs (X) and Output (y)
    X = df[NUMERIC_INPUTS + CATEGORICAL_INPUTS]
    y = df[OUTPUT_TARGET].values.astype(np.float32).reshape(-1, 1)

    # Split: 80% for training, 20% for testing
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the Preprocessor (Scales numbers, Encodes text)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_INPUTS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_INPUTS)
        ]
    )

    # Fit logic ONLY on training data
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Convert to PyTorch Tensors (The format the NN needs)
    inputs = {
        'train': torch.tensor(X_train, dtype=torch.float32),
        'test': torch.tensor(X_test, dtype=torch.float32)
    }
    targets = {
        'train': torch.tensor(y_train, dtype=torch.float32),
        'test': torch.tensor(y_test, dtype=torch.float32)
    }
    
    return inputs, targets, preprocessor, X_train.shape[1]

# ==========================================
# 3. DEFINE THE NEURAL NETWORK ARCHITECTURE
# ==========================================
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

# ==========================================
# 4. MAIN EXECUTION BLOCK
# ==========================================
if __name__ == "__main__":
    # 1. Load Data (Raw)
    try:
        df = pd.read_excel(DATA_FILE)
        print(f"✅ Loaded real data from {DATA_FILE}")
    except FileNotFoundError:
        print("⚠️ File not found. Generating DUMMY data...")
        # ... (Same dummy data gen as before) ...
        # [Paste dummy data generation here if needed for testing]
        df = pd.DataFrame(data)

    X = df[NUMERIC_INPUTS + CATEGORICAL_INPUTS]
    y = df[OUTPUT_TARGET].values.astype(np.float32).reshape(-1, 1)

    # 2. Configure K-Fold
    K_FOLDS = 5
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    results = {} # To store metrics for each fold

    print(f"\n--- Starting {K_FOLDS}-Fold Cross-Validation ---")

    # 3. The Cross-Validation Loop
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\nFOLD {fold+1}/{K_FOLDS}")
        print("-" * 20)

        # A. Split Data for this specific fold
        # Use .iloc because we are selecting by integer index
        X_train_raw = X.iloc[train_idx]
        y_train = y[train_idx]
        X_val_raw = X.iloc[val_idx]
        y_val = y[val_idx]

        # B. Preprocessing (Fit ONLY on training data to prevent leakage)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), NUMERIC_INPUTS),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_INPUTS)
            ]
        )
        
        X_train = preprocessor.fit_transform(X_train_raw)
        X_val = preprocessor.transform(X_val_raw) # Just transform validation

        # Convert to Tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

        # C. Initialize a FRESH Model (Reset weights)
        input_dim = X_train.shape[1]
        model = IsotopeNet(input_dim) 
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # D. Training Loop (Mini-School Year)
        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # E. Evaluation for this Fold
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor)
            val_loss = criterion(val_preds, y_val_tensor)
            rmse = np.sqrt(val_loss.item())
            
        print(f"Fold {fold+1} MSE: {val_loss.item():.4f} | RMSE: {rmse:.4f}")
        results[fold] = val_loss.item()

    # 4. Final Summary
    avg_mse = np.mean(list(results.values()))
    print("\n" + "="*30)
    print(f"AVERAGE MSE ACROSS {K_FOLDS} FOLDS: {avg_mse:.4f}")
    print("="*30)
    
    # Note: For the 'final' model you save to disk, you usually 
    # re-train on the FULL dataset (100% of data) using the
    # hyperparameters you verified here.
