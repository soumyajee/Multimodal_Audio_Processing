import pandas as pd

# Paths
old_csv = r"C:\Users\Asus\Downloads\drum_samples\metadata.csv"           # your current big CSV
new_csv = r"C:\Users\Asus\Downloads\drum_samples\metadata_simple.csv"   # new clean version

df = pd.read_csv(old_csv)

# Create the simple version
simple_df = pd.DataFrame()
simple_df["filename"] = df["filename"]
simple_df["label"] = df["class"].map({"Drum_samples": 0, "Keys_samples": 1})

# Save
simple_df.to_csv(new_csv, index=False)

print("Done! New simple CSV created:")
print(simple_df.head())
print(f"\nTotal samples: Drums = {(simple_df['label']==0).sum()}, Keys = {(simple_df['label']==1).sum()}")
print(f"Saved to: {new_csv}")
