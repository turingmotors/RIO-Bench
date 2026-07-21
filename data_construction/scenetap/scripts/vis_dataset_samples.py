import argparse
import os
import textwrap

from datasets import load_from_disk
from PIL import Image, ImageDraw, ImageFont


def _get_font(size: int):
    font_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fonts", "arial.ttf"))
    try:
        return ImageFont.truetype(font_path, size)
    except OSError:
        return ImageFont.load_default()


def _annotate(img: Image.Image, lines, font):
    if img.mode != "RGB":
        img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    y = 5
    for line in lines:
        draw.text((5, y), line, font=font, fill=(255, 0, 0))
        y += font.size + 2
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True, help="Path to load_from_disk dataset split.")
    parser.add_argument("--out_dir", default="vis_samples", help="Output directory for images.")
    parser.add_argument("--num", type=int, default=20, help="Number of samples to export.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_from_disk(args.dataset_dir)

    font = _get_font(16)
    count = min(args.num, len(ds))
    for i in range(count):
        ex = ds[i]
        img = ex.get("image")
        if isinstance(img, str):
            img = Image.open(img).convert("RGB")
        elif hasattr(img, "convert"):
            img = img.convert("RGB")
        else:
            continue

        q = ex.get("question", "")
        aw = ex.get("attack_word", "")
        ans = ex.get("answer", "")
        q_wrapped = textwrap.wrap(q, width=50)
        lines = [f"id={ex.get('image_id','')}", f"attack_word={aw}", f"answer={ans}"] + q_wrapped[:6]
        img = _annotate(img, lines, font)

        out_path = os.path.join(args.out_dir, f"sample_{i:04d}.jpg")
        img.save(out_path)


if __name__ == "__main__":
    main()
