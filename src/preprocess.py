import pandas as pd
import os

def process_data():
    # 1. Dynamically locate the project root (the parent of the 'src' folder)
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    
    # 2. Point to the correct data directories
    raw_data_dir = os.path.join(project_root, 'data', 'raw')
    output_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)

    # 3. Load files from data/raw/
    try:
        patients = pd.read_csv(os.path.join(raw_data_dir, 'patients.csv'))
        cad = pd.read_csv(os.path.join(raw_data_dir, 'cad_raw_data.csv'))
        diagnoses = pd.read_csv(os.path.join(raw_data_dir, 'diagnoses_icd.csv'))
    except FileNotFoundError as e:
        print(f"Error: Could not find raw data files. Check: {raw_data_dir}")
        return

    # 4. Processing logic
    cad['admittime'] = pd.to_datetime(cad['admittime'])
    cad['dischtime'] = pd.to_datetime(cad['dischtime'])
    cad['los'] = (cad['dischtime'] - cad['admittime']).dt.total_seconds() / (24 * 3600)

    # Aggregate diagnosis codes
    diag_pivot = pd.crosstab(diagnoses['hadm_id'], diagnoses['icd_code']).astype(int)

    # Merge demographics and diagnoses
    cad_unique = cad.drop_duplicates(subset=['hadm_id'])
    final_df = cad_unique.merge(patients, on='subject_id', how='left')
    final_df = final_df.merge(diag_pivot, on='hadm_id', how='left')

    # 5. Save the processed data
    output_path = os.path.join(output_dir, 'model_ready_data.csv')
    final_df.to_csv(output_path, index=False)
    print(f"Preprocessing complete! Data saved to: {output_path}")
    print(f"Final dataset shape: {final_df.shape}")

if __name__ == "__main__":
    process_data()