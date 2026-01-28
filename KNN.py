import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import tkinter as tk
from tkinter import filedialog, font
from pathlib import Path
import joblib

# ==========================================
# 1. USER CONFIGURATION
# ==========================================
DATA_FILE = "test_input_D_only.xlsx"  # Your Excel file name
MODEL_SAVE_PATH = "tritium_knn_model.pkl" # Changed extension to .pkl for Scikit-Learn

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
    'support', 
    'Membrane'
]

OUTPUT_TARGET = 'Separation_Factor'

# KNN Hyperparameters
N_NEIGHBORS = 5           # Look at the 5 closest data points
WEIGHTS = 'distance'      # 'distance': Closer points count more, 'uniform': All 5 vote equally

# ==========================================
# UI HELPER CLASSES (Unchanged)
# ==========================================
def display_stats_popup(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    root = tk.Toplevel()
    root.title("Diagnostic Report (KNN)")
    root.geometry("400x250")
    root.attributes('-topmost', True)

    header_font = font.Font(family="Helvetica", size=14, weight="bold")
    body_font = font.Font(family="Helvetica", size=12)

    stats_content = (
        f"Mean Squared Error (MSE):\n{mse:.5f}\n\n"
        f"Root Mean Squared Error (RMSE):\n{rmse:.5f}"
    )

    tk.Label(root, text="KNN Performance", font=header_font, pady=10).pack()
    tk.Label(root, text=stats_content, font=body_font, pady=10, justify="center").pack()
    tk.Button(root, text="Close", command=root.destroy, width=15, bg="#dddddd").pack(pady=20)

class RunStatusUI:
    def __init__(self, title="Tritium KNN Runner"):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(False, False)

        self._status_var = tk.StringVar(value="✅ Program started. Waiting for input file...")
        self._file_var = tk.StringVar(value="Input file: (not selected)")
        self._fold_var = tk.StringVar(value="Fold: (not started)")

        pad = 10
        frame = tk.Frame(self.root)
        frame.pack(padx=pad, pady=pad)

        tk.Label(frame, textvariable=self._status_var, justify="left", anchor="w").pack(fill="x")
        tk.Label(frame, textvariable=self._file_var, justify="left", anchor="w").pack(fill="x", pady=(6, 0))
        tk.Label(frame, textvariable=self._fold_var, justify="left", anchor="w").pack(fill="x", pady=(6, 0))

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
        self.set_status("Please select an input .xlsx file.")
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
                self.set_status("Selected file is not a .xlsx.")
                continue
            self._selected_path = str(p)
            self._file_var.set(f"Input file: {p.name}")
            self.set_status("✅ Input file selected. Starting run...")
            return self._selected_path

    def _on_select_clicked(self):
        if self._default_dir:
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
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    ui = RunStatusUI()
    
    # 1. File Selection
    default_path = Path(__file__).resolve().parent
    try:
        DATA_FILE = ui.choose_xlsx(default_dir=default_path)
    except SystemExit:
        exit()

    # 2. Load Data
    try:
        df = pd.read_excel(DATA_FILE)
        ui.set_status(f"✅ Loaded data from {Path(DATA_FILE).name}")
        print(f"✅ Loaded data from {DATA_FILE}")
    except Exception as e:
        ui.set_status(f"❌ Error loading file: {e}")
        ui.root.mainloop()
        raise e

    # --- DATA CLEANING ---
    for col in CATEGORICAL_INPUTS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()

    required_cols = NUMERIC_INPUTS + CATEGORICAL_INPUTS + [OUTPUT_TARGET]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        msg = f"❌ Missing columns: {missing_cols}"
        print(msg)
        ui.set_status(msg)
        ui.root.mainloop()
        raise ValueError(msg)

    # Prepare Data
    X = df[NUMERIC_INPUTS + CATEGORICAL_INPUTS]
    y = df[OUTPUT_TARGET].values # KNN prefers shape (n,) not (n,1) usually, but handles both

    # 3. K-Fold Cross Validation
    K_FOLDS = 5
    kfold = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    all_y_true = []
    all_y_pred = []
    all_indices = []
    fold_mses = []

    ui.set_status(f"--- Starting {K_FOLDS}-Fold Cross-Validation (KNN) ---")
    print(f"\n--- Starting {K_FOLDS}-Fold Cross-Validation (KNN) ---")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        ui.set_fold(fold+1, K_FOLDS)

        # Split Data
        X_train_raw = X.iloc[train_idx]
        y_train = y[train_idx]
        X_val_raw = X.iloc[val_idx]
        y_val = y[val_idx]
        
        original_rows = df.index[val_idx].tolist()

        # Preprocessing (Standard Scaling is CRITICAL for KNN)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), NUMERIC_INPUTS),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_INPUTS)
            ]
        )
        
        X_train = preprocessor.fit_transform(X_train_raw)
        X_val = preprocessor.transform(X_val_raw)

        # --- MODEL TRAINING (KNN) ---
        # No loops, no epochs. Just fit.
        model = KNeighborsRegressor(n_neighbors=N_NEIGHBORS, weights=WEIGHTS)
        model.fit(X_train, y_train)

        # --- EVALUATION ---
        val_preds = model.predict(X_val)
        
        # Calculate MSE
        val_mse = mean_squared_error(y_val, val_preds)
        
        # Store results
        all_y_true.extend(y_val.flatten())
        all_y_pred.extend(val_preds.flatten())
        all_indices.extend(original_rows)
        fold_mses.append(val_mse)

        ui.set_status(f"Fold {fold+1} RMSE: {np.sqrt(val_mse):.4f}")

    # --- AGGREGATED RESULTS ---
    avg_mse = np.mean(fold_mses)
    ui.set_status(f"✅ CV Complete. Avg RMSE: {np.sqrt(avg_mse):.4f}")
    
    # Display Stats
    display_stats_popup(all_y_true, all_y_pred)

    # 4. OUTLIER DETECTION REPORT
    print("\n" + "="*50)
    print("🔍 OUTLIER DETECTION REPORT (KNN)")
    print("="*50)
    
    res_df = pd.DataFrame({
        'Excel_Row': [i + 2 for i in all_indices],
        'Actual': all_y_true,
        'Predicted': all_y_pred
    })
    res_df['Error'] = abs(res_df['Actual'] - res_df['Predicted'])
    
    bad_preds = res_df[res_df['Predicted'] > 30] 
    if not bad_preds.empty:
        print("⚠️ SUSPICIOUS PREDICTIONS FOUND:")
        print(bad_preds[['Excel_Row', 'Actual', 'Predicted', 'Error']].to_string(index=False))
    else:
        print("No extreme outliers (>30) found.")

    # 5. PARITY PLOT
    ui.set_status("--- Generating Parity Plot ---")
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(all_y_true, all_y_pred, c='green', alpha=0.6, edgecolors='k', label='KNN Predictions')
    
    for i, txt in enumerate(all_indices):
        true_val = all_y_true[i]
        pred_val = all_y_pred[i]
        if abs(pred_val - true_val) > 5 or pred_val > 20:
            plt.annotate(f"Row {txt+2}", (true_val, pred_val), 
                         xytext=(5, 5), textcoords='offset points', 
                         fontsize=9, color='red', fontweight='bold')

    min_val = min(all_y_true.min(), all_y_pred.min())
    max_val = max(all_y_true.max(), all_y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    
    plt.xlabel('Actual Separation Factor')
    plt.ylabel('Predicted Separation Factor')
    plt.title(f'KNN Cross-Validation Parity Plot\n(n_neighbors={N_NEIGHBORS}, weights={WEIGHTS})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show(block=False) 

    # 6. FEATURE IMPORTANCE (Permutation Importance)
    # Since KNN doesn't have "weights" to inspect, we must use Permutation Importance
    ui.set_status("--- Training Final Model ---")
    
    final_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_INPUTS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_INPUTS)
        ]
    )
    X_full = final_preprocessor.fit_transform(X)
    
    # Train final model on ALL data
    final_model = KNeighborsRegressor(n_neighbors=N_NEIGHBORS, weights=WEIGHTS)
    final_model.fit(X_full, y)

    # Calculate Importance
    ui.set_status("--- Calculating Importance ---")
    
    # 1. Baseline RMSE
    baseline_preds = final_model.predict(X_full)
    baseline_rmse = np.sqrt(mean_squared_error(y, baseline_preds))
    
    ohe_names = final_preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_INPUTS)
    feature_names = NUMERIC_INPUTS + list(ohe_names)
    
    importances = []
    
    # 2. Permute each column and check RMSE increase
    X_full_np = np.array(X_full) # Ensure it's numpy array
    
    for i in range(X_full_np.shape[1]):
        X_shuffled = X_full_np.copy()
        # Shuffle just this column
        np.random.shuffle(X_shuffled[:, i])
        
        shuffled_preds = final_model.predict(X_shuffled)
        shuffled_rmse = np.sqrt(mean_squared_error(y, shuffled_preds))
        
        # Importance = How much worse did the model get?
        importance_score = shuffled_rmse - baseline_rmse
        importances.append(importance_score)
    
    # Sort and Plot
    indices = np.argsort(importances)
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(indices)), [importances[i] for i in indices], color='teal', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=9)
    plt.xlabel('RMSE Increase (Higher is more important)')
    plt.title(f'Feature Importance (KNN Permutation)\nBaseline RMSE: {baseline_rmse:.3f}')
    plt.tight_layout()
    plt.show(block=True)

    # 7. SAVE MODEL & SCALERS
    joblib.dump(final_model, MODEL_SAVE_PATH)
    joblib.dump(final_preprocessor, 'scaler_X.pkl')
    # KNN output does not need inverse transform usually if y wasn't scaled. 
    # If you later decide to scale y, you need to save scaler_y here too.
    
    print(f"Model saved to {MODEL_SAVE_PATH}")
    print("Preprocessors saved to scaler_X.pkl")