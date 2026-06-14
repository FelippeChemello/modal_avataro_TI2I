# Qwen Image Edit on Modal

`app.py` serves `Qwen/Qwen-Image-Edit-2511` on an RTX PRO 6000 GPU. It accepts
an edit prompt and one or more base64-encoded reference images, then returns
PNG bytes.

## Deploy

```bash
modal deploy app.py
```

The initial build downloads the model to the shared `hf_hub_cache` Modal
Volume. Subsequent container starts load it from that cache.

`width` and `height` are optional integers from 256 through 2048 and must be
divisible by 16. If omitted, the pipeline generates approximately one
megapixel while preserving the input image's aspect ratio.

## Local invocation

The local entrypoint uploads an image to the Modal function and writes the
returned PNG:

```bash
modal run app.py --input-path cat.png \
  --prompt "Turn this cat into a dog" \
  --output-path output.png \
  --width 1024 \
  --height 1024 \
  --seed 0
```

## Multiple reference images

Call `generate` with `images_base64` to use up to eight ordered reference
images:

```python
output_bytes = Model().generate.remote(
    prompt=(
        "Use Picture 1 for the character and Picture 2 for the clothing. "
        "Place the character in the setting from Picture 3."
    ),
    images_base64=[character_base64, clothing_base64, setting_base64],
    width=1024,
    height=1024,
)
```

`images_base64` must be a non-empty list, including when only one reference
image is used.
