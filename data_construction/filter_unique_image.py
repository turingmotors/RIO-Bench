
import os
import glob
from datasets import load_from_disk

data_dir = "./data/RIO-Bench/hf_dataset"
filtered_dir = "./data/RIO-Bench/hf_dataset_unique_img"

# --- text
for split in ["train", "val"]:
    for t in ["txt_clean", "txt_attack"]:
        dataset_dirs = glob.glob(os.path.join(data_dir, split, t, "*"))
        # dataset_dirs = [d for d in dataset_dirs if "open_ended" in d and "_v3" in d]
        dataset_dirs = [d for d in dataset_dirs if "old" not in d]
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
            out_dir = _d.replace(data_dir, filtered_dir)
            os.makedirs(out_dir, exist_ok=True)
            ds_unique.save_to_disk(out_dir)
            print(f"  Saved to {out_dir}")
            print()
print("Done.")

# --- object
for split in ["train", "val"]:
    for t in ["obj_clean", "obj_attack"]:
        dataset_dirs = glob.glob(os.path.join(data_dir, split, t, "*"))
        dataset_dirs = [d for d in dataset_dirs if "open_ended" in d and "_v3" in d]
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
            out_dir = _d.replace(data_dir, filtered_dir)
            os.makedirs(out_dir, exist_ok=True)
            ds_unique.save_to_disk(out_dir)
            print(f"  Saved to {out_dir}")
            print()
print("Done.")


