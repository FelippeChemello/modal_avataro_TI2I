import base64
import binascii
import io
import secrets
from pathlib import Path
from typing import Any

import modal

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"
CACHE_DIR = "/cache"
MAX_INPUT_BYTES = 25 * 1024 * 1024
MAX_INPUT_PIXELS = 40_000_000
MAX_REFERENCE_IMAGES = 8
MAX_PROMPT_LENGTH = 4_000
MIN_OUTPUT_DIMENSION = 256
MAX_OUTPUT_DIMENSION = 2_048
MAX_OUTPUT_PIXELS = 4_194_304


def download_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=MODEL_ID, cache_dir=CACHE_DIR)


hf_cache = modal.Volume.from_name("hf_hub_cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "accelerate>=1.10.0,<2",
        "fastapi[standard]>=0.115.0,<1",
        "huggingface_hub[hf_xet]>=0.34.0,<1",
        "pillow>=11.0.0,<13",
        "sentencepiece>=0.2.0,<1",
        "torch==2.8.0",
        "torchvision==0.23.0",
        "transformers>=4.57.0,<5",
        "git+https://github.com/huggingface/diffusers.git",
        "cbor2",
    )
    .env(
        {
            "HF_HOME": CACHE_DIR,
            "HF_HUB_CACHE": CACHE_DIR,
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .run_function(download_model, volumes={CACHE_DIR: hf_cache})
)

app = modal.App("qwen-image-edit", image=image)

with image.imports():
    import torch
    from diffusers import QwenImageEditPlusPipeline
    from fastapi import HTTPException
    from huggingface_hub import snapshot_download
    from PIL import Image, ImageOps, UnidentifiedImageError


def _decode_image(image_base64: str):
    if not isinstance(image_base64, str) or not image_base64.strip():
        raise ValueError("image_base64 must be a non-empty base64 string")

    encoded = image_base64.strip()
    if encoded.startswith("data:"):
        try:
            header, encoded = encoded.split(",", 1)
        except ValueError as exc:
            raise ValueError("Invalid image data URL") from exc
        if ";base64" not in header:
            raise ValueError("The image data URL must use base64 encoding")

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc

    if not image_bytes:
        raise ValueError("Decoded image is empty")
    if len(image_bytes) > MAX_INPUT_BYTES:
        raise ValueError(f"Decoded image exceeds the {MAX_INPUT_BYTES // (1024 * 1024)} MiB limit")

    try:
        input_image = Image.open(io.BytesIO(image_bytes))
        if input_image.width * input_image.height > MAX_INPUT_PIXELS:
            raise ValueError(f"Input image exceeds the {MAX_INPUT_PIXELS:,}-pixel limit")
        input_image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Decoded data is not a supported image") from exc

    return ImageOps.exif_transpose(input_image).convert("RGB")


def _decode_images(images_base64: list[str]):
    if not isinstance(images_base64, list) or not images_base64:
        raise ValueError("images_base64 must be a non-empty list of base64 strings")
    if len(images_base64) > MAX_REFERENCE_IMAGES:
        raise ValueError(f"images_base64 supports at most {MAX_REFERENCE_IMAGES} images")

    return [_decode_image(encoded_image) for encoded_image in images_base64]


def _validate_dimensions(width: int | None, height: int | None) -> None:
    for name, value in (("width", width), ("height", height)):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        if not MIN_OUTPUT_DIMENSION <= value <= MAX_OUTPUT_DIMENSION:
            raise ValueError(
                f"{name} must be between {MIN_OUTPUT_DIMENSION} and {MAX_OUTPUT_DIMENSION}"
            )
        if value % 16 != 0:
            raise ValueError(f"{name} must be divisible by 16")

    if width is not None and height is not None and width * height > MAX_OUTPUT_PIXELS:
        raise ValueError(f"width * height must not exceed {MAX_OUTPUT_PIXELS:,} pixels")


@app.cls(
    gpu="RTX-PRO-6000",
    timeout=900,
    scaledown_window=180,
    volumes={CACHE_DIR: hf_cache},
)
class Model:
    @modal.enter()
    def load_model(self) -> None:
        model_path = snapshot_download(
            repo_id=MODEL_ID,
            cache_dir=CACHE_DIR,
        )
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        self.pipe.to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def _generate(
        self,
        prompt: str,
        images_base64: list[str],
        negative_prompt: str = " ",
        num_inference_steps: int = 20,
        true_cfg_scale: float = 4.0,
        guidance_scale: float = 1.0,
        seed: int | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> tuple[bytes, int]:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must be non-empty")
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise ValueError(f"prompt exceeds the {MAX_PROMPT_LENGTH}-character limit")
        _validate_dimensions(width, height)

        input_images = _decode_images(images_base64)
        resolved_seed = seed if seed is not None else secrets.randbelow(2**63)
        generator = torch.Generator(device="cuda").manual_seed(resolved_seed)

        with torch.inference_mode():
            output_image = self.pipe(
                image=input_images,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                guidance_scale=guidance_scale,
                generator=generator,
                num_images_per_prompt=1,
                width=width,
                height=height,
            ).images[0]

        output = io.BytesIO()
        output_image.save(output, format="PNG")
        return output.getvalue(), resolved_seed

    @modal.method()
    def generate(
        self,
        prompt: str,
        images_base64: list[str],
        width: int | None = None,
        height: int | None = None,
        negative_prompt: str = "Blurry, low quality, deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limbs, extra fingers, missing limbs, blurry, floating limbs, disconnected limbs, malformed hands, blur, out of focus",
        seed: int | None = None,
        num_inference_steps: int = 20,
        true_cfg_scale: float = 4.0,
        guidance_scale: float = 1.0,
    ) -> bytes:
        image_bytes, _ = self._generate(
            images_base64=images_base64,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            true_cfg_scale=true_cfg_scale,
            guidance_scale=guidance_scale,
            seed=seed,
            width=width,
            height=height,
        )
        return image_bytes


@app.local_entrypoint()
def main(
    input_path: str,
    prompt: str,
    output_path: str = "output.png",
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> None:
    input_bytes = Path(input_path).read_bytes()
    images_base64 = [base64.b64encode(input_bytes).decode("ascii")]
    output_bytes = Model().generate.remote(
        prompt,
        images_base64,
        seed=seed,
        width=width,
        height=height,
    )
    Path(output_path).write_bytes(output_bytes)
    print(f"Saved {output_path} ({len(output_bytes):,} bytes)")
