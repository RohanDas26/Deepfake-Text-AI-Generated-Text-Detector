from datasets import load_dataset

print("Downloading full RAID dataset to local cache...")
# This downloads all splits (train, test, extra) and caches them locally
dataset = load_dataset("liamdugan/raid")

print("Download complete!")
print(dataset)

# Optional: Save a local copy as Parquet/Arrow files on disk
dataset.save_to_disk("./raid_dataset_local")
print("Saved to local folder: ./raid_dataset_local")