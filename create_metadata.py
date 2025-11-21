import os
import csv

# ====== CONFIG ======
BASE_DIR = "Data"  # update to your audio base folder
OUTPUT_CSV = "audio_metadata.csv"

CATEGORY_MAP = {
    "drum_samples": "Drum_samples",
    "key_samples": "Keys_samples"
}

# CSV Headers
headers = ["filename", "class", "title", "tags"]

rows = []

for folder, class_name in CATEGORY_MAP.items():
    folder_path = os.path.join(BASE_DIR, folder)

    if not os.path.exists(folder_path):
        print(f"⚠ Folder not found: {folder_path}")
        continue

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            title = os.path.splitext(file)[0]  # remove extension
            tag = class_name.replace("_", " ").title().replace(" ", "_")  # generate tag

            rows.append([file, class_name, title, tag])

# Write CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    writer.writerows(rows)

print(f"🎯 CSV file generated successfully: {OUTPUT_CSV}")
print(f"Total files processed: {len(rows)}")
