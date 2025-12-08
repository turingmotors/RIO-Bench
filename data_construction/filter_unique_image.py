
import os
import glob
from datasets import load_from_disk

DATA_DIR = "../data/RIO-Bench/hf_dataset"
FILTERED_DATA_DIR = "../data/RIO-Bench/hf_dataset_unique_img"

# --- text
for split in ["train", "val"]:
    for t in ["txt_clean", "txt_attack"]:
        dataset_dirs = glob.glob(os.path.join(DATA_DIR, split, t, "*"))
        for _d in dataset_dirs:
            print(f"Processing {_d} ...")
            ds = load_from_disk(_d)
            print(f"  Original size: {len(ds)}")
            unique_img_ids = set()
            unique_indices = []
            for i, item in enumerate(ds):
                img_id = item["image_id"]
                if img_id not in unique_img_ids:
                    unique_img_ids.add(img_id)
                    unique_indices.append(i)
                if i % 1000 == 0:
                    print(f"    Processed {i} items, found {len(unique_img_ids)} unique images so far...")
            ds_unique = ds.select(unique_indices)
            print(f"  Unique size: {len(ds_unique)}")
            out_dir = _d.replace(DATA_DIR, FILTERED_DATA_DIR)
            os.makedirs(out_dir, exist_ok=True)
            ds_unique.save_to_disk(out_dir)
            print(f"  Saved to {out_dir}")
            print()
print("Done.")

# --- object
for split in ["train", "val"]:
    for t in ["obj_clean", "obj_attack"]:
        dataset_dirs = glob.glob(os.path.join(DATA_DIR, split, t, "*"))
        for _d in dataset_dirs:
            print(f"Processing {_d} ...")
            ds = load_from_disk(_d)
            print(f"  Original size: {len(ds)}")
            unique_img_ids = set()
            unique_indices = []
            for i, item in enumerate(ds):
                img_id = item["image_id"]
                if img_id not in unique_img_ids:
                    unique_img_ids.add(img_id)
                    unique_indices.append(i)
                if i % 1000 == 0:
                    print(f"    Processed {i} items, found {len(unique_img_ids)} unique images so far...")
            ds_unique = ds.select(unique_indices)
            print(f"  Unique size: {len(ds_unique)}")
            out_dir = _d.replace(DATA_DIR, FILTERED_DATA_DIR)
            os.makedirs(out_dir, exist_ok=True)
            ds_unique.save_to_disk(out_dir)
            print(f"  Saved to {out_dir}")
            print()
print("Done.")


