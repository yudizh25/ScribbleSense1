# ScribbleSense
The implementation of "ScribbleSense: Generative Scribble-based Texture Editing with Intent Prediction"

## Installation
```bash
pip install -r requirements.txt
```

## Running
Please feel free to use the latest / state-of-the-art MLLMs to obtain the predicted local texture patch.
Then, update original_path and tile_path in PatchToTextureIslands.py with the actual paths.
After that, run
```bash
python PatchToTextureIslands.py
```
to get input textures for inpainting step. Then, apply inpainting to obtain the final texture result.
```bash
python -m scripts.run_texture --config_path=configs/texture_edit/bunny_step3.yaml
```

## Acknowledgement
Our project is built on [TEXTure](https://github.com/TEXTurePaper/TEXTurePaper). We sincerely thank the authors for their awesome work!