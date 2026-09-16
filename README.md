# 🛡️ ComfyUI Binyuan · Ultimate Sampler V6.5 **EN (Bilingual)**

Bilingual build of the all-in-one sampler: **English UI for English users, Chinese UI for Chinese users.**

* Node: `🛡️ Binyuan Sampler V6.5 EN｜中英双语版` — category `Binyuan`
* Node key: `BinyuanUltimateSamplerEN` (different from the Chinese edition, so **both can be installed side by side**)
* Widget labels: English ASCII keys, translated by `locales/en|zh/nodeDefs.json`
* Combo values are written bilingually, e.g. `Inherit upstream ｜ 继承上游模型`
* All error messages are bilingual: `Please select a Checkpoint / 请选择 Checkpoint`

## Install
1. Copy this folder into `ComfyUI/custom_nodes/binyuan_sampler_plugin_english/` (the folder name does not matter).
2. Restart ComfyUI (or use ComfyUI Manager: search `Binyuan Sampler`).
3. Add the node: right-click → `Binyuan` → `🛡️ Binyuan Sampler V6.5 EN｜中英双语版`.

## Use
* `Load Mode` = `Checkpoint (whole file)` → pick a checkpoint, connect `IMAGE` to `SaveImage`.
* `Load Mode` = `Separate` → pick `Diffusion Model` + `CLIP 1/2` + `VAE` and set `CLIP Type` to match the architecture.
* LoRA: use the LoRA manager panel inside the node (`+ Add LoRA`), or type JSON in `LoRA JSON`.
* img2img: wire an image into `Upstream Image 1` and lower `Denoise`.
* `Chaining Mode` = `Inherit upstream` uses the optional `External Model/CLIP/VAE` inputs.

## Notes / requirements
* `.gguf` models and CLIPs need **ComfyUI-GGUF** installed (`custom_nodes/ComfyUI-GGUF`). It is optional: only used when you pick a `.gguf` file (not declared as a hard dependency).
* `CLIP Type` depends on your ComfyUI core version. Verified against local tags:
  | ComfyUI core | missing CLIP types |
  |---|---|
  | >= v0.26.2 | none |
  | v0.24.1 | boogu, krea2 |
  | v0.22.2 | lens, pixeldit, ideogram4, boogu, krea2 |
  | v0.20.3 | cogvideox, lens, pixeldit, ideogram4, boogu, krea2 |
  On an older core an unsupported type silently falls back to `stable_diffusion` (upstream `CLIPLoader` uses `getattr(..., default)`), which produces confusing encode/decode errors. Update ComfyUI if you need those architectures.
* The built-in `Upstream Image Handling` widget is kept for workflow compatibility; the effective behaviour is driven by `Latent Source`.

### If sampling fails with an attention error (NOT a plugin problem)
```
No operator found for `memory_efficient_attention_forward` ...
`fa3F`/`fa2F`/`cutlassF` is not supported because:
    requires device with capability <= (9, 0) but your GPU has capability (12, 0) (too new)
```
This happens when **xformers is installed** (ComfyUI uses it by default as soon as it is importable)
and its prebuilt kernels do not cover your GPU (e.g. RTX 50 series / sm_120). **Every** sampling node
fails the same way - the stock `KSampler` too - so it is an environment issue. Fix it at launch:
```
python main.py --use-pytorch-cross-attention --disable-xformers
```
RTX 40 series and older are unaffected.

### Chaining with the Chinese V6.5 edition
Works in both directions: wire `MODEL` / `CLIP` / `VAE` (outputs 2 / 3 / 4) into
`External Model` / `External CLIP` / `External VAE` and set `Chaining Mode = Inherit upstream`.
The inheriting node then reuses the already-loaded model (no second load, saves VRAM).
* Files: `__init__.py` (node), `js/binyuan_lora_en.js` (LoRA manager UI), `locales/en|zh/nodeDefs.json` (official ComfyUI custom-node translation), `pyproject.toml`, `LICENSE` (MIT).

## 中文说明
这是 **中英双语版**：英文界面用户看到英文标签，中文界面用户看到中文标签（通过官方 `locales/` 翻译机制）。
节点键名与中文版不同（`BinyuanUltimateSamplerEN` vs `BinyuanUltimateSampler`），**两个版本可以同时安装，不会互相覆盖**。
所有报错信息均为中英双语；下拉选项也是双语写法（如 `继承上游模型`），不依赖翻译文件也能看懂。
