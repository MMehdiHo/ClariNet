import pandas as pd
import os
from pathlib import Path
import random
from config import NUM_IMG_CSV, CSV_FOLDER_PATH, OUTPUT_CSV, TARGET_IMAGES

def process_folder(folder, min_select, max_select, class_label, data_selected, selected_images):
    csv_file_path = Path(CSV_FOLDER_PATH) / f"{folder}_coverage.csv"

    if not csv_file_path.exists():
        print(f"⚠️ File {csv_file_path} not found! Skipping...")
        return selected_images

    df = pd.read_csv(csv_file_path)

    if df.shape[1] < 7:
        print(f"⚠️ File {csv_file_path} does not have the required columns! Skipping...")
        return selected_images

    selected_rows = df.sample(n=min(len(df), max_select), random_state=42)

    for _, row in selected_rows.iterrows():
        if selected_images >= TARGET_IMAGES:
            return selected_images
        image_full_path = os.path.join(folder, str(row[0]))
        data_selected.append([image_full_path, float(row[6]), float(row[5]), class_label])
        selected_images += 1

    return selected_images

def main():
    num_img_df = pd.read_csv(NUM_IMG_CSV, header=0, names=['Folder Name', 'Number of Images', 'Class'])
    final_data = []
    
    for class_label in num_img_df['Class'].unique():
        class_df = num_img_df[num_img_df['Class'] == class_label]
        data_selected = []
        selected_images = 0

        for _, row in class_df.iterrows():
            folder_name = row['Folder Name']
            img_count = int(row['Number of Images'])

            min_select = 50 if img_count < 400 else 400
            max_select = min(TARGET_IMAGES - selected_images, img_count)

            selected_images = process_folder(folder_name, min_select, max_select, class_label, data_selected, selected_images)

            if selected_images >= TARGET_IMAGES:
                break

        final_data.extend(data_selected)

    final_df = pd.DataFrame(final_data, columns=['image_path', 'healthy', 'coverage', 'class'])
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"✅ Dataset balanced and saved successfully to {OUTPUT_CSV}!")

if __name__ == "__main__":
    main()
