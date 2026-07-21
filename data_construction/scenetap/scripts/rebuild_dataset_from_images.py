import argparse
import os
import re
from io import BytesIO
from datasets import load_from_disk, Image as HFImage
from PIL import Image


FNAME_RE = re.compile(r"^(?P<image_id>[^_]+)_(?P<question_id>\d+)\.[^.]+$")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_dataset_dir", required=True,
                        help="Dataset split to replace images (load_from_disk path).")
    parser.add_argument("--other_dataset_dir", action="append", default=[],
                        help="Other dataset splits to filter by available question_id.")
    parser.add_argument("--images_dir", required=True, help="Directory with attacked images.")
    parser.add_argument("--out_root", required=True, help="Output root directory.")
    parser.add_argument(
        "--out_attack_dataset_dir",
        default="",
        help="If set, save rebuilt attack dataset to this exact directory path.",
    )
    parser.add_argument("--suffix", required=True, help="Suffix appended to output dataset name.")
    parser.add_argument("--prefer_question_id", action="store_true",
                        help="Match by question_id (default). If not set, match by image_id.")
    parser.add_argument("--debug_qid", type=str, default="",
                        help="If set, print mapping info for this question_id.")
    args = parser.parse_args()

    # Build mapping from question_id or image_id to image path
    mapping = {}
    for fname in os.listdir(args.images_dir):
        m = FNAME_RE.match(fname)
        if not m:
            continue
        image_id = m.group("image_id")
        question_id = m.group("question_id")
        key = question_id if args.prefer_question_id else image_id
        mapping[str(key)] = os.path.join(args.images_dir, fname)
    if args.debug_qid:
        print(f"mapping_size={len(mapping)} debug_qid={args.debug_qid} path={mapping.get(args.debug_qid)}")

    def _out_dir(in_dir: str) -> str:
        parent = os.path.basename(os.path.dirname(in_dir))
        split = os.path.basename(in_dir)
        return os.path.join(args.out_root, f"{parent}__{args.suffix}", split)

    key_set = set(mapping.keys())

    # Attack dataset: filter to keys and replace images
    ds_attack = load_from_disk(args.attack_dataset_dir)
    def _keep(ex):
        key = str(ex.get("question_id") if args.prefer_question_id else ex.get("image_id"))
        return key in key_set
    ds_attack = ds_attack.filter(_keep)

    new_images = []
    for ex in ds_attack:
        key = str(ex.get("question_id") if args.prefer_question_id else ex.get("image_id"))
        path = mapping.get(key)
        with open(path, "rb") as f:
            new_images.append({"bytes": f.read()})

    if "image" in ds_attack.column_names:
        ds_attack = ds_attack.remove_columns("image")
    ds_attack = ds_attack.add_column("image", new_images)
    ds_attack = ds_attack.cast_column("image", HFImage())
    out_attack = args.out_attack_dataset_dir if args.out_attack_dataset_dir else _out_dir(args.attack_dataset_dir)
    os.makedirs(out_attack, exist_ok=True)
    ds_attack.save_to_disk(out_attack)

    # Other datasets: filter to keys only
    for other_dir in args.other_dataset_dir:
        ds_other = load_from_disk(other_dir)
        ds_other = ds_other.filter(_keep)
        out_other = _out_dir(other_dir)
        os.makedirs(out_other, exist_ok=True)
        ds_other.save_to_disk(out_other)


if __name__ == "__main__":
    main()
