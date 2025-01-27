import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import traceback
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Optional

import PIL.Image
import torch

# PYTHONPATH=PPTAgent/src:$PYTHONPATH
os.sys.path.append('/Users/yyyang/Codes/PPTAgent/src')

# --- imports from your project ---
from FlagEmbedding import BGEM3FlagModel
from marker.models import create_model_dict
import induct
import llms
import pptgen
from model_utils import get_image_model, parse_pdf
from multimodal import ImageLabler
from presentation import Presentation
from utils import Config, is_image_path, pjoin, ppt_to_images, tenacity

# ---------------
# Global settings
# ---------------
RUNS_DIR = "runs"
STAGES = [
    "PPT Parsing",
    "PDF/Topic Parsing",
    "Slide Induction",
    "PPT Generation",
    "Success!",
]
REFINE_TEMPLATE_PATH = "prompts/document_refine.txt"
CAPTION_PROMPT_PATH = "prompts/caption.txt"

# For demonstration, let's load exactly 1 model (instead of multiple).
NUM_MODELS = 1
DEVICE_COUNT = torch.cuda.device_count() or torch.mps.device_count()

# Set up your fallback logic: if you have multiple possible LLMs, handle that in code below
def setup_models():
    """
    Try to connect to your primary model, and if unsuccessful,
    possibly fall back to a different model.
    """
    if llms.language_model.test_connection() and llms.vision_model.test_connection():
        print("Primary models connected successfully.")
        return

    if llms.gpt4o.test_connection():
        print("Switching to OpenAI GPT-4o models as fallback.")
        llms.language_model = llms.gpt4o
        llms.vision_model = llms.gpt4o
        return

    raise RuntimeError(
        "No working model connections available. Check your API keys and environment."
    )

# -----------
# Main logic
# -----------
def topic_generate(topic: str):
    """Generate a JSON doc structure from a given text topic using your language model."""
    # Example prompt
    prompt = (
        "Please generate a detailed presentation planning document about "
        + topic
        + ", detail to about 1000 words. "
        + "Follow the format of the example output.\n"
        + """
{
    "title": "title of document",
    "sections": [
        {
            "title": "title of section1",
            "subsections": [
                {
                    "title": "title of subsection1.1",
                    "content": "content of subsection1.1"
                },
                {
                    "title": "title of subsection1.2",
                    "content": "content of subsection1.2"
                }
            ]
        },
        {
            "title": "title of section2",
            "subsections": [
                {
                    "title": "title of subsection2.1",
                    "content": "content of subsection2.1"
                }
            ]
        }
    ]
}
"""
    )
    text = llms.language_model(prompt, return_json=True)
    if not isinstance(text, dict):
        raise ValueError("Text is not in JSON format or could not parse model output.")
    return text


@tenacity
def refine_document(markdown_document: str):
    """
    Use your refine prompt to convert raw parsed PDF text
    into a structured JSON (doc_json).
    """
    with open(REFINE_TEMPLATE_PATH, "r", encoding="utf-8") as f:
        refine_template = f.read()

    prompt = refine_template.replace("{{markdown_document}}", markdown_document)
    doc_json = llms.language_model(prompt, return_json=True)
    if not isinstance(doc_json, dict):
        raise ValueError("Refined document is not in valid JSON format.")
    return doc_json


def generate_slides(
    ppt_template_path: str,
    pdf_path: Optional[str],
    topic: Optional[str],
    slides_count: int,
    output_dir: str,
):
    """
    Given a PPT template and either a PDF file or a topic string,
    generate a final presentation (pptx).
    """

    # Create the run folder
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Using run directory: {output_dir}")

    # -- 1. Load models (text, image, marker) --
    # Here we do a simpler single-model approach
    text_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=0)
    image_model = get_image_model(device=0)
    marker_model = create_model_dict(device=0, dtype=torch.float16)

    # -- 2. Prepare the config objects --
    generation_config = Config(output_dir)
    # We'll store the "pptx" in a unique subfolder for safety.
    pptx_md5 = hashlib.md5(open(ppt_template_path, "rb").read()).hexdigest()
    pptx_config = Config(pjoin("runs", "pptx", pptx_md5))
    os.makedirs(pptx_config.RUN_DIR, exist_ok=True)

    # If you want to copy the PPT template into the "pptx" folder for caching:
    if not os.path.exists(pjoin(pptx_config.RUN_DIR, "source.pptx")):
        os.system(f"cp '{ppt_template_path}' '{pjoin(pptx_config.RUN_DIR, 'source.pptx')}'")

    print(f"[INFO] PPT Template MD5: {pptx_md5}")

    # -- 3. Parse the PPT Template into a Presentation object --
    print("[STAGE] PPT Parsing")
    presentation = Presentation.from_file(
        pjoin(pptx_config.RUN_DIR, "source.pptx"), pptx_config
    )
    ppt_image_folder = pjoin(pptx_config.RUN_DIR, "slide_images")

    if not os.path.exists(ppt_image_folder) or len(os.listdir(ppt_image_folder)) == 0:
        ppt_to_images(pjoin(pptx_config.RUN_DIR, "source.pptx"), ppt_image_folder)
        # Because your original code expects the slides to be renamed to slide_{i:04d}.jpg,
        # handle index alignment & remove error slides if any:
        for err_idx, _ in presentation.error_history:
            err_path = pjoin(ppt_image_folder, f"slide_{err_idx:04d}.jpg")
            if os.path.exists(err_path):
                os.remove(err_path)
        # rename real_idx => slide_idx
        for i, slide in enumerate(presentation.slides, start=1):
            slide.slide_idx = i
            old_name = pjoin(ppt_image_folder, f"slide_{slide.real_idx:04d}.jpg")
            new_name = pjoin(ppt_image_folder, f"slide_{slide.slide_idx:04d}.jpg")
            if os.path.exists(old_name):
                os.rename(old_name, new_name)

    # You may optionally caption each slide image:
    labler = ImageLabler(presentation, pptx_config)
    labler.caption_images()

    # -- 4. Parse PDF or use a text topic to get doc_json --
    print("[STAGE] PDF/Topic Parsing")
    if pdf_path:
        # We’ll store PDF results under runs/pdf/<md5>/...
        pdf_md5 = hashlib.md5(open(pdf_path, "rb").read()).hexdigest()
        parsedpdf_dir = pjoin("runs", "pdf", pdf_md5)
        os.makedirs(parsedpdf_dir, exist_ok=True)
        # If the refined document is not cached, parse the PDF text & refine
        refined_doc_json_path = pjoin(parsedpdf_dir, "refined_doc.json")
        if not os.path.exists(refined_doc_json_path):
            print("[INFO] Parsing PDF ...")
            text_content = parse_pdf(pdf_path, parsedpdf_dir, marker_model)
            # Optional: also caption images found in PDF
            caption_json_path = pjoin(parsedpdf_dir, "caption.json")
            if not os.path.exists(caption_json_path):
                with open(CAPTION_PROMPT_PATH, "r", encoding="utf-8") as f:
                    caption_prompt = f.read()
                images_info = {}
                for k in os.listdir(parsedpdf_dir):
                    if is_image_path(k):
                        img_path = pjoin(parsedpdf_dir, k)
                        try:
                            text_cap = llms.vision_model(caption_prompt, [img_path])
                            with PIL.Image.open(img_path) as img:
                                size = img.size
                            images_info[img_path] = [text_cap, size]
                        except Exception as e:
                            print(f"[ERROR] Could not caption {k}: {str(e)}")
                with open(caption_json_path, "w", encoding="utf-8") as f:
                    json.dump(images_info, f, ensure_ascii=False, indent=4)

            # Now refine the markdown doc => JSON
            doc_json = refine_document(text_content)
            json.dump(doc_json, open(refined_doc_json_path, "w"), indent=4)
        else:
            print("[INFO] Using cached refined_doc.json")
            doc_json = json.load(open(refined_doc_json_path, "r"))
        # Also load the PDF's image captions (if you want them for generation):
        caption_json_path = pjoin(parsedpdf_dir, "caption.json")
        images = json.load(open(caption_json_path)) if os.path.exists(caption_json_path) else {}
    else:
        # If no PDF, we assume we have a text topic
        print(f"[INFO] Generating from topic: '{topic}'")
        pdf_md5 = topic  # just reuse the "pdf" variable for your code’s logic
        doc_json = topic_generate(topic)
        images = {}

    # -- 5. Slide Induction (looking at the PPT structure) --
    print("[STAGE] Slide Induction")
    template_img_dir = pjoin(pptx_config.RUN_DIR, "template_images")
    if not os.path.exists(template_img_dir) or len(os.listdir(template_img_dir)) == 0:
        # Save a stripped version of the PPT
        deepcopy(presentation).save(
            pjoin(pptx_config.RUN_DIR, "template.pptx"), layout_only=True
        )
        ppt_to_images(
            pjoin(pptx_config.RUN_DIR, "template.pptx"), template_img_dir
        )

    slide_inducter = induct.SlideInducter(
        presentation,
        ppt_image_folder,
        template_img_dir,
        pptx_config,
        image_model,
        "inference_script",  # or "backend", up to you
    )
    slide_induction = slide_inducter.content_induct()

    # -- 6. PPT Generation --
    print("[STAGE] PPT Generation")
    # instantiate the “crew” that handles text generation for slides
    crew = pptgen.PPTCrew(text_model, error_exit=False, retry_times=5)
    crew.set_reference(presentation, slide_induction)
    # Actually generate the new PPT
    # (the code will produce final.pptx in output_dir)
    crew.generate_pres(
        generation_config,
        images,         # PDF or custom images (captions, if any)
        slides_count,   # number of pages to generate
        doc_json        # the JSON structure
    )

    print("[STAGE] Success!")
    print(f"[INFO] Output PPT stored at {pjoin(output_dir, 'final.pptx')}")


def main():
    """
    CLI entry point:
      1) Parse arguments
      2) Load fallback models if needed
      3) Run the generate_slides pipeline
    """
    parser = argparse.ArgumentParser(description="PPT agent inference script.")
    parser.add_argument(
        "--ppt",
        required=True,
        help="Path to the PPTX template to use",
    )
    parser.add_argument(
        "--pdf",
        default=None,
        help="Path to a PDF file (if you prefer PDF-based doc generation). Omit if using a --topic.",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="Topic string (if no PDF). E.g. 'Quantum Mechanics Overview'.",
    )
    parser.add_argument(
        "--slides",
        type=int,
        default=5,
        help="Number of slides/pages to generate in the final PPT.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to store the final PPT. Defaults to runs/<DATE_TIME>_<UUID>/",
    )
    args = parser.parse_args()

    # If user doesn't specify output_dir, create a unique one.
    if not args.output_dir:
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        args.output_dir = os.path.join(RUNS_DIR, f"{dt_str}_{unique_id}")

    # Basic checks
    if args.pdf is None and args.topic is None:
        print("[ERROR] Must provide either --pdf or --topic.")
        sys.exit(1)
    if args.pdf and not os.path.exists(args.pdf):
        print(f"[ERROR] PDF file not found: {args.pdf}")
        sys.exit(1)

    print("[INFO] Setting up models...")
    setup_models()  # your fallback or direct approach

    # Now run the pipeline
    generate_slides(
        ppt_template_path=args.ppt,
        pdf_path=args.pdf,
        topic=args.topic,
        slides_count=args.slides,
        output_dir=args.output_dir,
    )
    print("[DONE] Presentation generation complete.")


if __name__ == "__main__":
    main()

"""
export OPENAI_API_KEY=sk-xxx

python run_pptgen.py \
--ppt /Users/yyyang/Downloads/pptagent_test/source.pptx \
--topic "An Introduction to PPT Agent" \
--slides 6
"""