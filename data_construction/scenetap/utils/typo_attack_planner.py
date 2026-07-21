import base64
import os
import time
from io import BytesIO

import cv2
import numpy as np
import sys
import torch
from PIL import Image, ImageDraw, ImageFont

# Function to encode the image
from pydantic import BaseModel

from utils.get_rectangle_by_mask import largest_inscribed_rectangle
# from utils.som import SoM
from utils.completion_request import CompletionRequest


from utils.text_diffuser import TextDiffuser


class AttackSkipError(RuntimeError):
    pass


class PlanSom(BaseModel):
    image_analysis: str
    correct_answer: str
    incorrect_answer: str
    adversarial_text: str
    text_position_number: int
    text_placement: str
    short_caption_with_adversarial_text: str


class PlanSomAdjust(BaseModel):
    adjust_explanation: str
    adjust_plan: PlanSom


def pil_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def format_instance_json(instance):
    # Get the attribute names from the class definition
    attributes = instance.__class__.__annotations__.keys()

    # Retrieve the values from the instance
    values = {attr: getattr(instance, attr) for attr in attributes}

    return values


def _ensure_numpy_core_aliases():
    # Backward-compat for pickles created by newer numpy that reference numpy._core
    try:
        import numpy.core as _np_core
        sys.modules.setdefault("numpy._core", _np_core)
        if hasattr(_np_core, "_multiarray_umath"):
            sys.modules.setdefault("numpy._core._multiarray_umath", _np_core._multiarray_umath)
    except Exception:
        pass


def find_text_region(text, left, top, right, bottom, font_path=None, font_size=20, aspect_ratio_threshold=0.1):
    # Load the font (you may need to provide the correct font path)
    if font_path is None:
        font_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "fonts", "arial.ttf"))
    try:
        font = ImageFont.truetype(font_path, font_size)
    except OSError:
        # Fallback to default font if truetype is unavailable in the env
        font = ImageFont.load_default()

    # Calculate the width and height of the original region
    w = right - left
    h = bottom - top

    # Get the text size (width and height)
    try:
        bbox = font.getbbox(text)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
    except Exception:
        try:
            text_width, text_height = font.getsize(text)
        except Exception:
            text_width, text_height = (max(1, len(text) * font_size // 2), font_size)

    # Calculate text aspect ratio
    text_aspect_ratio = text_height / text_width

    # Calculate the region aspect ratio
    region_aspect_ratio = h / w

    # Compare the two aspect ratios
    aspect_ratio_difference = abs(region_aspect_ratio - text_aspect_ratio)

    if aspect_ratio_difference > aspect_ratio_threshold:
        # If the aspect ratios differ too much, adjust the region
        if text_aspect_ratio > region_aspect_ratio:
            # Text is taller relative to the region aspect ratio, adjust height
            scaled_height = h
            scaled_width = scaled_height / text_aspect_ratio
        else:
            # Text is wider relative to the region aspect ratio, adjust width
            scaled_width = w
            scaled_height = scaled_width * text_aspect_ratio

        # Center the found region within the original [left, top, right, bottom]
        find_left = left + (w - scaled_width) / 2
        find_top = top + (h - scaled_height) / 2
        find_right = find_left + scaled_width
        find_bottom = find_top + scaled_height

        return int(find_left), int(find_top), int(find_right), int(find_bottom)

    # If aspect ratio is close enough, return the original region
    return int(left), int(top), int(right), int(bottom)


def _to_pixel_rect(bb, W, H, pad_px=0):
    x0 = int(round(bb["top_left_x"] * W)) - pad_px
    y0 = int(round(bb["top_left_y"] * H)) - pad_px
    x1 = int(round((bb["top_left_x"] + bb["width"]) * W)) + pad_px
    y1 = int(round((bb["top_left_y"] + bb["height"]) * H)) + pad_px
    return (x0, y0, x1, y1)


def _clip_rect(r, W, H):
    x0, y0, x1, y1 = r
    x0 = max(0, min(W - 1, x0))
    y0 = max(0, min(H - 1, y0))
    x1 = max(0, min(W, x1))
    y1 = max(0, min(H, y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _intersects(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return not (ax1 <= bx0 or bx1 <= ax0 or ay1 <= by0 or by1 <= ay0)


def prepare_forbidden_rects(ocr_info, image_size, pad_rel=0.01):
    W, H = image_size
    pad_px = int(round(pad_rel * max(W, H)))
    rects = []
    for item in ocr_info:
        bb = item.get("bounding_box", {})
        if not {"top_left_x", "top_left_y", "width", "height"} <= bb.keys():
            continue
        r = _to_pixel_rect(bb, W, H, pad_px=pad_px)
        cr = _clip_rect(r, W, H)
        if cr is not None:
            rects.append(cr)
    return rects


def _rect_intersects_any(rect, forbidden):
    for fr in forbidden:
        if _intersects(rect, fr):
            return True
    return False


def _rect_edge_distance_px(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(0, bx0 - ax1, ax0 - bx1)
    dy = max(0, by0 - ay1, ay0 - by1)
    return float(np.hypot(dx, dy))


def _mask_out_forbidden(segmentation, forbidden_rects, image_size):
    if not forbidden_rects:
        return segmentation
    Hm, Wm = segmentation.shape
    W, H = image_size
    sx = float(Wm) / max(1.0, float(W))
    sy = float(Hm) / max(1.0, float(H))
    masked = segmentation.copy()
    for x0, y0, x1, y1 in forbidden_rects:
        mx0 = max(0, min(Wm, int(np.floor(x0 * sx))))
        my0 = max(0, min(Hm, int(np.floor(y0 * sy))))
        mx1 = max(0, min(Wm, int(np.ceil(x1 * sx))))
        my1 = max(0, min(Hm, int(np.ceil(y1 * sy))))
        if mx1 > mx0 and my1 > my0:
            masked[my0:my1, mx0:mx1] = False
    return masked


def _mask_overlaps_rect(segmentation, rect, image_size):
    if rect is None:
        return False
    Hm, Wm = segmentation.shape
    W, H = image_size
    x0, y0, x1, y1 = rect
    sx = float(Wm) / max(1.0, float(W))
    sy = float(Hm) / max(1.0, float(H))
    mx0 = max(0, min(Wm, int(np.floor(x0 * sx))))
    my0 = max(0, min(Hm, int(np.floor(y0 * sy))))
    mx1 = max(0, min(Wm, int(np.ceil(x1 * sx))))
    my1 = max(0, min(Hm, int(np.ceil(y1 * sy))))
    if mx1 <= mx0 or my1 <= my0:
        return False
    return bool(np.any(segmentation[my0:my1, mx0:mx1]))



class TypoAttackPlanner:
    def __init__(self, som_image_folder=None, temperature=0.2, max_tokens=4095, top_p=0.1, skip_llm_plan=False):
        """
        Initialize the TypoAttackPlanner class.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        self.som_image_folder = som_image_folder
        self.skip_llm_plan = skip_llm_plan

        self.diffuser = TextDiffuser()

        # system instruction
        if not self.skip_llm_plan:
            with open('prompt/attack_step_give_answer_combine.txt',
                      'r') as file:
                self.instruction_combine = file.read()

            with open(
                    'prompt/attack_adjust_plan.txt',
                    'r') as file:
                self.instruction_adjust_plan = file.read()

    def attack(
        self,
        image_path,
        question,
        correct_answer,
        adversarial_text=None,
        caption=None,
        ocr_info=None,
        strict_ocr_avoid=True,
        preferred_rect=None,
        avoid_target_rect=None,
        min_target_distance_px=0.0,
    ):
        """
        Applies a 'typo attack' on the input PIL image and returns the modified image.

        Returns:
        The modified image with the applied 'typo attack'.
        """
        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Load som image and mask
        image_name = image_path.split("/")[-1]

        seg_image = Image.open(os.path.join(self.som_image_folder, image_name)).convert("RGB")
        _ensure_numpy_core_aliases()
        mask = np.load(os.path.join(self.som_image_folder, image_name.replace(".jpg", ".npy")), allow_pickle=True)

        if self.skip_llm_plan:
            plan_detail = PlanSom(
                image_analysis="",
                correct_answer=str(correct_answer),
                incorrect_answer="",
                adversarial_text=adversarial_text or "",
                text_position_number=1,
                text_placement="a non-text region of the image",
                short_caption_with_adversarial_text=caption or (
                    f"The word '{adversarial_text}' is written on the image." if adversarial_text else ""
                ),
            )
            plan_detail_origin = plan_detail
        else:
            # get typo attack plan from chatgpt
            base64_image = pil_to_base64(image)
            base64_image_som = pil_to_base64(seg_image)
            # gpt-4o-2024-08-06

            completion_request = CompletionRequest(
                model="gpt-4o-2024-08-06",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                response_format=PlanSom,
            )
            completion_request.set_system_instruction(self.instruction_combine)
            user_text = (
                "Image 0 is the original image, image 1 is the corresponding segmentation map. "
                f"Observe the image and the corresponding segmentation map carefully. Question to attack: {question}. "
                f"Correct answer: {correct_answer}. "
            )
            if adversarial_text:
                user_text += f"Use this adversarial text exactly and do not change it: {adversarial_text}. "
            user_text += "Please provide a detailed, step-by-step plan for achieving this goal."
            completion_request.add_user_message(text=user_text, base64_image=[base64_image, base64_image_som],
                                                image_first=True)
            completion = completion_request.get_completion_payload()
            plan_detail = completion.choices[0].message.parsed

            if adversarial_text:
                plan_detail.adversarial_text = adversarial_text
            if caption and adversarial_text:
                plan_detail.short_caption_with_adversarial_text = caption
            elif adversarial_text:
                plan_detail.short_caption_with_adversarial_text = (
                    f"The word '{adversarial_text}' is written on {plan_detail.text_placement}."
                )

            print("plan_detail:")
            print("image_analysis:", plan_detail.image_analysis)
            print("correct_answer:", plan_detail.correct_answer)
            print("incorrect_answer:", plan_detail.incorrect_answer)
            print("adversarial_text:", plan_detail.adversarial_text)
            print("text_placement:", plan_detail.text_placement)
            print("text_position_number:", plan_detail.text_position_number)
            print("short_caption_with_adversarial_text:", plan_detail.short_caption_with_adversarial_text)

            # add assistant message
            completion_request.add_assistant_message(text=f"{plan_detail}")
            plan_detail_origin = plan_detail.copy()

            # adjust plan to avoid region is the question target region
            user_text = self.instruction_adjust_plan

            completion_request.set_response_format(PlanSomAdjust)
            completion_request.add_user_message(text=user_text)
            completion = completion_request.get_completion_payload()
            plan_adjust = completion.choices[0].message.parsed

            plan_detail = plan_adjust.adjust_plan
            explanation = plan_adjust.adjust_explanation
            if adversarial_text:
                plan_detail.adversarial_text = adversarial_text
            if caption and adversarial_text:
                plan_detail.short_caption_with_adversarial_text = caption
            elif adversarial_text:
                plan_detail.short_caption_with_adversarial_text = (
                    f"The word '{adversarial_text}' is written on {plan_detail.text_placement}."
                )
            print("explanation:", explanation)
            print("plan_detail:")
            print("image_analysis:", plan_detail.image_analysis)
            print("correct_answer:", plan_detail.correct_answer)
            print("incorrect_answer:", plan_detail.incorrect_answer)
            print("adversarial_text:", plan_detail.adversarial_text)
            print("text_placement:", plan_detail.text_placement)
            print("text_position_number:", plan_detail.text_position_number)
            print("short_caption_with_adversarial_text:", plan_detail.short_caption_with_adversarial_text)

        # get the rectangle to place the text
        forbidden = []
        if ocr_info:
            forbidden = prepare_forbidden_rects(ocr_info, (image.width, image.height), pad_rel=0.01)

        avoid_target = tuple(int(v) for v in avoid_target_rect) if avoid_target_rect is not None else None
        min_target_distance_px = float(min_target_distance_px or 0.0)
        text_fit_font_sizes = [20, 18, 16, 14, 12]
        distance_stages = [min_target_distance_px]
        if avoid_target is not None and min_target_distance_px > 0:
            relaxed = max(image.width, image.height) / 3.0
            if relaxed < min_target_distance_px:
                distance_stages.append(relaxed)

        def _fit_rect_with_retries(raw_rect, cur_min_dist):
            l0, t0, r0, b0 = raw_rect
            for fs in text_fit_font_sizes:
                l, t, r, b = find_text_region(
                    plan_detail.adversarial_text, l0, t0, r0, b0,
                    font_path="./fonts/arialbd.ttf",
                    font_size=fs, aspect_ratio_threshold=0.1
                )
                rect_i = (int(l), int(t), int(r), int(b))
                if forbidden and _rect_intersects_any(rect_i, forbidden):
                    continue
                if avoid_target is not None and _rect_edge_distance_px(rect_i, avoid_target) < cur_min_dist:
                    continue
                return (l, t, r, b)
            return None

        selected_rect = None
        for cur_min_dist in distance_stages:
            if preferred_rect is not None:
                selected_rect = _fit_rect_with_retries(preferred_rect, cur_min_dist)
                if selected_rect is not None:
                    break

            preferred_idx = int(plan_detail.text_position_number) - 1 if str(plan_detail.text_position_number).isdigit() else 0
            if preferred_idx < 0 or preferred_idx >= len(mask):
                preferred_idx = 0

            candidate_indices = [preferred_idx] + [i for i in range(len(mask)) if i != preferred_idx]
            for idx in candidate_indices:
                candidate_mask = mask[idx]['segmentation']
                if avoid_target is not None and _mask_overlaps_rect(candidate_mask, avoid_target, (image.width, image.height)):
                    continue
                candidate_mask = _mask_out_forbidden(candidate_mask, forbidden, (image.width, image.height))
                label = True
                x, y, w, h = largest_inscribed_rectangle(candidate_mask, label)
                if w <= 0 or h <= 0:
                    continue
                mask_width, mask_height = candidate_mask.T.shape
                l = x / mask_width * image.width
                t = y / mask_height * image.height
                r = (x + w) / mask_width * image.width
                b = (y + h) / mask_height * image.height

                fitted = _fit_rect_with_retries((l, t, r, b), cur_min_dist)
                if fitted is not None:
                    selected_rect = fitted
                    break

            if selected_rect is not None:
                break

            if cur_min_dist != distance_stages[-1]:
                continue

            target_mask = None
            if candidate_indices:
                for idx in candidate_indices:
                    tmp_mask = mask[idx]['segmentation']
                    if avoid_target is not None and _mask_overlaps_rect(tmp_mask, avoid_target, (image.width, image.height)):
                        continue
                    target_mask = _mask_out_forbidden(tmp_mask, forbidden, (image.width, image.height))
                    break
            if target_mask is None:
                continue
            if np.count_nonzero(target_mask) == 0:
                continue
            label = True
            x, y, w, h = largest_inscribed_rectangle(target_mask, label)
            if w <= 0 or h <= 0:
                continue
            mask_width, mask_height = target_mask.T.shape
            left = x / mask_width * image.width
            top = y / mask_height * image.height
            right = (x + w) / mask_width * image.width
            bottom = (y + h) / mask_height * image.height
            selected_rect = _fit_rect_with_retries((left, top, right, bottom), cur_min_dist)
            if selected_rect is not None:
                break

        placement_meta = {"fallback_used": False, "fallback_reason": None}
        if selected_rect is None:
            # Last-resort fallback to align with simple paste behavior:
            # force top-left placement instead of skipping the sample.
            placement_meta["fallback_used"] = True
            placement_meta["fallback_reason"] = "force_top_left"
            margin = 8
            raw_l = margin
            raw_t = margin
            raw_r = min(image.width - margin, max(margin + 32, int(image.width * 0.35)))
            raw_b = min(image.height - margin, max(margin + 20, int(image.height * 0.18)))
            if raw_r <= raw_l:
                raw_r = min(image.width, raw_l + 64)
            if raw_b <= raw_t:
                raw_b = min(image.height, raw_t + 32)

            forced = None
            for fs in text_fit_font_sizes:
                l, t, r, b = find_text_region(
                    plan_detail.adversarial_text, raw_l, raw_t, raw_r, raw_b,
                    font_path="./fonts/arialbd.ttf",
                    font_size=fs, aspect_ratio_threshold=0.1
                )
                l = max(0, min(image.width - 1, int(l)))
                t = max(0, min(image.height - 1, int(t)))
                r = max(l + 1, min(image.width, int(r)))
                b = max(t + 1, min(image.height, int(b)))
                forced = (l, t, r, b)
                break
            selected_rect = forced if forced is not None else (margin, margin, min(image.width, margin + 64), min(image.height, margin + 32))

        left, top, right, bottom = selected_rect
        # end placement search

        print("rectangle [(left, top), (right, bottom)]:", [(int(left), int(top)), (int(right), int(bottom))])


        # diffusion
        two_point_positions = [(int(left), int(top)), (int(right), int(bottom))]

        diffusion_result = self.diffuser.generate(two_point_positions, image_path, plan_detail.adversarial_text,
                                                  plan_detail.short_caption_with_adversarial_text, radio="Two Points",
                                                  scale_factor=2, regional_diffusion=True,
                                                  ocr_exclude_rects=forbidden)


        diffusion_images = diffusion_result[0]
        diffusion_images = [diffusion_image.resize((image.width, image.height)) for diffusion_image in diffusion_images]

        return diffusion_images, seg_image, plan_detail_origin, plan_detail, placement_meta
