import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog
from pathlib import Path

# ==========================================
# 1. USER CONFIGURATION (EDIT THIS SECTION)
# ==========================================
DATA_FILE = "test_input_D_only.xlsx"  # Your Excel file name
MODEL_SAVE_PATH = "tritium_model.pth"

# Define your columns exactly as they appear in your Excel sheet
NUMERIC_INPUTS = [
    'temp (C)', 
    'Current Density (mA/cm^2)', 
    'Initial Concentration', 
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
    'Anode_Label', 
    'support'
]

OUTPUT_TARGET = 'Separation_Factor'

# Hyperparameters (Knobs to turn)
HIDDEN_SIZE = 64          # Number of neurons in hidden layers
LEARNING_RATE = 0.001     # How fast the model learns
EPOCHS = 1000             # How many times to loop through the data


class RunStatusUI:
    def __init__(self, title="Tritium Model Runner"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(False, False)

        self._status_var = tk.StringVar(value="✅ Program started. Waiting for input file selection...")
        self._file_var = tk.StringVar(value="Input file: (not selected)")
        self._fold_var = tk.StringVar(value="Fold: (not started)")

        pad = 10

        frame = tk.Frame(self.root)
        frame.pack(padx=pad, pady=pad)

        lbl_status = tk.Label(frame, textvariable=self._status_var, justify="left", anchor="w")
        lbl_status.pack(fill="x")

        lbl_file = tk.Label(frame, textvariable=self._file_var, justify="left", anchor="w")
        lbl_file.pack(fill="x", pady=(6, 0))

        lbl_fold = tk.Label(frame, textvariable=self._fold_var, justify="left", anchor="w")
        lbl_fold.pack(fill="x", pady=(6, 0))

        btn_frame = tk.Frame(frame)
        btn_frame.pack(fill="x", pady=(10, 0))

        self._select_btn = tk.Button(btn_frame, text="Select .xlsx Input", command=self._on_select_clicked)
        self._select_btn.pack(side="left")

        self._close_btn = tk.Button(btn_frame, text="Kill", command=self.close)
        self._close_btn.pack(side="right")

        self._selected_path = None
        self._default_dir = None

        self._refresh()

    def _refresh(self):
        try:
            self.root.update_idletasks()
            self.root.update()
        except tk.TclError:
            pass

    def set_status(self, text: str):
        self._status_var.set(text)
        self._refresh()

    def set_fold(self, fold_num: int, total_folds: int):
        self._fold_var.set(f"Fold: {fold_num}/{total_folds}")
        self._status_var.set(f"Running fold {fold_num}/{total_folds} ...")
        self._refresh()

    def choose_xlsx(self, default_dir: Path):
        self._default_dir = Path(default_dir).resolve()
        self.set_status("Please select an input .xlsx file (must be in the same folder as this script).")

        while True:
            path = filedialog.askopenfilename(
                initialdir=str(self._default_dir),
                title="Select an Excel (.xlsx) input file",
                filetypes=[("Excel files", "*.xlsx")],
            )

            if not path:
                self.set_status("No file selected. Program will exit.")
                raise SystemExit(0)

            p = Path(path).resolve()

            if p.suffix.lower() != ".xlsx":
                self.set_status("Selected file is not a .xlsx. Please choose a .xlsx file.")
                continue

            if p.parent != self._default_dir:
                self.set_status(f"Please select a .xlsx from: {self._default_dir}")
                continue

            self._selected_path = str(p)
            self._file_var.set(f"Input file: {p.name}")
            self.set_status("✅ Input file selected. Starting run...")
            return self._selected_path

    def _on_select_clicked(self):
        if self._default_dir is None:
            return
        try:
            self.choose_xlsx(self._default_dir)
        except SystemExit:
            self.close()

    def close(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass


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
# ==========================================
# 4. MAIN EXECUTION BLOCK (REWRITTEN)
# ==========================================
if __name__ == "__main__":
    ui = RunStatusUI()
    
    # File Selection (with GUI)
    default_path = Path(__file__).resolve().parent
    try:
        DATA_FILE = ui.choose_xlsx(default_dir=default_path)
    except SystemExit:
        print("User closed the window.")
        exit()


    # Load Data from .xlsx
    try:
        df = pd.read_excel(DATA_FILE)
        ui.set_status(f"✅ Loaded data from {Path(DATA_FILE).name}")
        print(f"✅ Loaded data from {DATA_FILE}")
    except Exception as e:
        ui.set_status(f"❌ Error loading file: {e}")
        ui.root.mainloop()
        raise e


    # Normalize capitalization/whitespace for categorical inputs
    for col in CATEGORICAL_INPUTS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()


    # Check for missing columns
    required_cols = NUMERIC_INPUTS + CATEGORICAL_INPUTS + [OUTPUT_TARGET]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        msg = f"❌ Missing columns in Excel: {missing_cols}"
        print(msg)
        ui.set_status(msg)
        ui.root.mainloop()
        raise ValueError(msg)


    # Prepare Data
    X = df[NUMERIC_INPUTS + CATEGORICAL_INPUTS]
    y = df[OUTPUT_TARGET].values.astype(np.float32).reshape(-1, 1)


    # Configure K-Fold Cross Validation
    K_FOLDS = 5
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    # Storage for aggregating results across ALL folds
    all_y_true = []
    all_y_pred = []
    fold_mses = []

    ui.set_status(f"--- Starting {K_FOLDS}-Fold Cross-Validation ---")
    print(f"\n--- Starting {K_FOLDS}-Fold Cross-Validation ---")

    # 4. The Cross-Validation Loop
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        ui.set_fold(fold+1, K_FOLDS)

        # A. Split Data
        X_train_raw = X.iloc[train_idx]
        y_train = y[train_idx]
        X_val_raw = X.iloc[val_idx]
        y_val = y[val_idx]

        # B. Preprocessing (Fit on Train, Transform Val)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), NUMERIC_INPUTS),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_INPUTS)
            ]
        )
        
        X_train = preprocessor.fit_transform(X_train_raw)
        X_val = preprocessor.transform(X_val_raw)

        # Convert to Tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        
        # C. Initialize & Train Model
        input_dim = X_train.shape[1]
        model = IsotopeNet(input_dim) 
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 0: ui.root.update()

        # D. Evaluation (Save predictions for the Aggregated Plot)
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_tensor)
            val_mse = criterion(val_preds, torch.tensor(y_val, dtype=torch.float32))
            
            # Store Actual vs Predicted for this chunk
            all_y_true.extend(y_val.flatten())
            all_y_pred.extend(val_preds.numpy().flatten())
            fold_mses.append(val_mse.item())

        ui.set_status(f"Fold {fold+1} RMSE: {np.sqrt(val_mse.item()):.4f}")

    # 5. Final Aggregated Results
    avg_mse = np.mean(fold_mses)
    ui.set_status(f"✅ CV Complete. Avg RMSE: {np.sqrt(avg_mse):.4f}")
    print("\n" + "="*40)
    print(f"CROSS-VALIDATION RESULTS (Average of {K_FOLDS} Folds)")
    print(f"Average MSE:  {avg_mse:.4f}")
    print(f"Average RMSE: {np.sqrt(avg_mse):.4f}")
    print("="*40)

    # 6. PLOT 1: Aggregated Parity Plot (The "Honest" Performance)
    ui.set_status("--- Generating Aggregated Parity Plot ---")
    
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    plt.figure(figsize=(7, 7))
    plt.scatter(all_y_true, all_y_pred, c='blue', alpha=0.6, edgecolors='k', label='Cross-Validation Predictions')
    
    min_val = min(all_y_true.min(), all_y_pred.min())
    max_val = max(all_y_true.max(), all_y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    
    plt.xlabel('Actual Separation Factor')
    plt.ylabel('Predicted Separation Factor')
    plt.title(f'Cross-Validation Parity Plot\n(n={len(all_y_true)}, RMSE={np.sqrt(avg_mse):.2f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show(block=False) # Keep open while we calculate importances

    # 7. Retrain Final Model on 100% Data for Feature Importance
    # (We can't average feature importance easily across 5 different models, 
    # so we train one "Master" model on all data to see what it learns.)
    ui.set_status("--- Training Final Model on 100% Data ---")
    print("\n--- Training Final Master Model (100% Data) ---")

    # Full Preprocessing
    final_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_INPUTS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_INPUTS)
        ]
    )
    X_full = final_preprocessor.fit_transform(X)
    X_full_tensor = torch.tensor(X_full, dtype=torch.float32)
    y_full_tensor = torch.tensor(y, dtype=torch.float32)

    # Full Training
    final_model = IsotopeNet(X_full.shape[1])
    optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        final_model.train()
        optimizer.zero_grad()
        loss = criterion(final_model(X_full_tensor), y_full_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0: ui.root.update()

    # 8. PLOT 2: Feature Importance (Permutation Method on Final Model)
    ui.set_status("--- Calculating Feature Importance ---")
    print("--- Calculating Feature Importance ---")

    ohe_names = final_preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_INPUTS)
    feature_names = NUMERIC_INPUTS + list(ohe_names)
    
    final_model.eval()
    with torch.no_grad():
        baseline_loss = criterion(final_model(X_full_tensor), y_full_tensor).item()

    importances = []
    for i in range(X_full_tensor.shape[1]):
        X_shuffled = X_full_tensor.clone()
        indices = torch.randperm(X_shuffled.size(0))
        X_shuffled[:, i] = X_shuffled[indices, i]
        
        with torch.no_grad():
            shuffled_loss = criterion(final_model(X_shuffled), y_full_tensor).item()
        
        # We use RMSE difference for intuitive scale
        importances.append(np.sqrt(shuffled_loss) - np.sqrt(baseline_loss))

    # Sort and Plot
    indices = np.argsort(importances) # Sort low to high
    sorted_names = [feature_names[i] for i in indices]
    sorted_vals = [importances[i] for i in indices]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), sorted_vals, color='teal', align='center')
    plt.yticks(range(len(indices)), sorted_names, fontsize=9)
    plt.xlabel('Increase in RMSE (Prediction Error) when feature is randomized')
    plt.title('Feature Importance (Final Model)')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show(block=True)

    ui.set_status("✅ Process Complete.")
    ui.root.mainloop()