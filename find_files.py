import os

# Start searching from your Downloads folder
search_path = r'C:\Users\iamve\Downloads'

for root, dirs, files in os.walk(search_path):
    for file in files:
        if 'patients' in file.lower() or 'cad_raw' in file.lower() or 'diagnoses' in file.lower():
            print(f"FOUND: {os.path.join(root, file)}")