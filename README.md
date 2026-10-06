> 2026-09-28 更新：新增高级分段采样和可调初始噪声强度。请先阅读 [完整使用说明（Markdown）](Binyuan采样器V2.2EN_详细使用说明.md) 或 [TXT说明](Binyuan采样器V2.2EN_详细使用说明.txt)。
>
> 本机核对环境为 ComfyUI 0.37.0；新增功能完成接口与逻辑测试，不代表所有模型、整合包和内核版本通用。普通模式下重绘强度与初始噪声强度可同时使用；高级模式忽略重绘强度，只有“添加噪声”开启时初始噪声强度才生效。

# 🛡️ ComfyUI Binyuan · Ultimate Sampler V2.2 **EN (Bilingual)**

Bilingual build of the all-in-one sampler: **English UI for English users, Chinese UI for Chinese users.**

* Node: `🛡️ Binyuan Sampler V2.2 EN｜中英双语版` — category `Binyuan`
* Node key: `BinyuanUltimateSamplerEN` (different from the Chinese edition, so **both can be installed side by side**)
* Widget labels: English ASCII keys, translated by `locales/en|zh/nodeDefs.json`
* Combo values are written bilingually, e.g. `Inherit upstream ｜ 继承上游模型`
* All error messages are bilingual: `Please select a Checkpoint / 请选择 Checkpoint`

## Qwen-Image-2.1 edit modes

* **Auto** keeps existing workflows compatible: an upstream image uses native reference editing; without one it is text-to-image.
* **Native reference edit** follows the official Qwen-Image-2.1 path. The image is a reference condition and sampling starts from an empty latent, so Denoise is fixed to 1.0. `Qwen reference resolution = 0` preserves the source size (rounded to a multiple of 32); values such as 1024 use a pixel budget and can change the canvas.
* **VAE redraw** reuses the first reference image's Qwen VAE latent as the starting canvas. It preserves the source canvas (rounded to a multiple of 32) and makes Denoise effective. Typical second-pass values are 0.25-0.55.
* An explicitly connected **External Latent** still has priority when requested by Latent Source/chaining mode. Only connect latents compatible with the active model; across different model families, pass IMAGE and use VAE redraw.

These paths use ComfyUI public node/model APIs and contain no integration-pack-specific paths.

## Attention backend

`Attention Backend` defaults to **Auto safe**. For a Krea2 or Qwen-Image-2.1 model loaded inside this sampler, it selects Comfy Kitchen when available, or the ComfyUI sub-quadratic backend otherwise. This is a model-local setting: it does not change other nodes or ComfyUI startup flags. Externally connected models retain their existing attention patches. Choose **Core default** to follow ComfyUI's startup setting, **Comfy Kitchen** to require that backend, or **Sub-quadratic** to force the lower-memory fallback. The latter can be slower. This option is appended after existing widgets so saved workflow values keep their positions.

## Install
1. Copy this folder into `ComfyUI/custom_nodes/binyuan_sampler_plugin_english/` (the folder name does not matter).
2. Restart ComfyUI (or use ComfyUI Manager: search `Binyuan Sampler`).
3. Add the node: right-click → `Binyuan` → `🛡️ Binyuan Sampler V2.2 EN｜中英双语版`.

## Use
* `Load Mode` = `Checkpoint (whole file)` → pick a checkpoint, connect `IMAGE` to `SaveImage`.
* `Load Mode` = `Separate` → pick `Diffusion Model` + `CLIP 1/2` + `VAE` and set `CLIP Type` to match the architecture.
* LoRA: use the LoRA manager panel inside the node (`+ Add LoRA`), or type JSON in `LoRA JSON`.
* img2img: wire an image into `Upstream Image 1` and lower `Denoise`.
* `Chaining Mode` = `Inherit upstream` uses the optional `External Model/CLIP/VAE` inputs.

## Notes / requirements
* `.gguf` models and CLIPs need **ComfyUI-GGUF** installed (`custom_nodes/ComfyUI-GGUF`). It is optional: only used when you pick a `.gguf` file (not declared as a hard dependency).
* Model/CLIP support depends on the installed ComfyUI core and matching dependencies. No universal or minimum-version compatibility guarantee has been established.
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
Compatibility depends on the installed attention backend, GPU, PyTorch and CUDA versions; this is not limited to one GPU generation.

### Chaining with the Chinese edition
Works in both directions: wire `MODEL` / `CLIP` / `VAE` (outputs 2 / 3 / 4) into
`External Model` / `External CLIP` / `External VAE` and set `Chaining Mode = Inherit upstream`.
The inheriting node then reuses the already-loaded model (no second load, saves VRAM).
* Files: `__init__.py` (node), `js/binyuan_lora_en.js` (LoRA manager UI), `locales/en|zh/nodeDefs.json` (official ComfyUI custom-node translation), `pyproject.toml`, `LICENSE` (MIT).

## 中文说明
这是 **中英双语版**：英文界面用户看到英文标签，中文界面用户看到中文标签（通过官方 `locales/` 翻译机制）。
节点键名与中文版不同（`BinyuanUltimateSamplerEN` vs `BinyuanUltimateSampler`），**两个版本可以同时安装，不会互相覆盖**。
所有报错信息均为中英双语；下拉选项也是双语写法（如 `继承上游模型`），不依赖翻译文件也能看懂。


## Qwen-Image-2.1 / 千问图像2.1

Requires a ComfyUI core with `TextEncodeQwenImage21`. The sampler detects the loaded model's latent format, and uses native Qwen-Image-2.1 conditioning for both positive and negative prompts, including image slots and reference latents. Native prefix KV caching and model sampling settings remain under ComfyUI's control.

需要内置 `TextEncodeQwenImage21` 的 ComfyUI。采样器根据实际模型的 Latent 格式自动识别2.1，正负提示词均使用原生编码流程，保留图像位置和参考 Latent；前缀 KV 缓存和模型采样设置由 ComfyUI 原生实现管理。

- **Loading / 加载**: `Separate` + Qwen-Image-2.1 diffusion model + matching Qwen3-VL 8B in `clip_1` + `clip_2=None` + `clip_type=qwen_image_2.1` (maps to native `qwen_image`) + matching 64-channel RGBA VAE. / 分离式加载：扩散模型、配套 Qwen3-VL 8B 编码器、专用64通道 RGBA VAE；不要使用旧版 Qwen 的 Wan VAE 或提示词扩写模型代替条件编码器。
- **Text to image / 文生图**: no upstream images; width, height and batch size control the empty latent. / 不接上游图像，宽高和生成数量控制空 Latent。
- **Editing / 编辑**: upstream image ports grow automatically: two are initially visible; connecting the spare port adds another. Disconnecting removes unused ports and compacts the remaining connections, keeping one spare (at least two ports). Each port uses its first image. Use `denoise=1.0`. / 上游图像端口动态增减：默认显示两个，接上备用端口后自动新增；断开后收起多余端口，其余连线顺次前移，保留一个备用端口（最少两个）。每个端口取首张图，重绘强度设1.0。
- **Reference size / 参考尺寸**: `qwen_reference_resolution=1024` uses about one megapixel per reference; `0` preserves original dimensions rounded to 32. Editing output follows the first processed reference even if width/height differ. Batch size still applies. Stitch mode first combines references horizontally. / 默认按约一百万像素等比缩放；0按原尺寸对齐32。编辑输出跟随处理后的首张参考图，生成数量仍有效；拼接模式先横向合成参考图。
- **External inputs / 外接**: inherited positive and negative conditioning are preserved; supplying both permits operation without CLIP. Connect the external latent to override the initial latent. / 继承模式保留外部正负条件，两路均提供时无需 CLIP；外部 Latent 可覆盖初始 Latent。
- **Transparency / 透明图**: RGBA references and decoded output retain alpha. If Load Image supplies a separate mask, first combine image and mask using `JoinImageWithAlpha`. / 参考图与解码输出保留透明通道；加载图片节点若分开输出图像和蒙版，先用 `JoinImageWithAlpha` 合成 RGBA。

The node ID, non-image ports, widget order, bilingual translations and LoRA manager are preserved. Old image connections are restored before dynamic compaction. Dynamic ports have English/Chinese labels. The plugin does not cap the reference count; Qwen-Image-2.1 officially describes up to 10 references, and larger counts depend on model behavior and available memory. Restart ComfyUI and refresh the browser after updating. / 节点ID、非图像端口、控件顺序、双语翻译和 LoRA 管理器保留，旧工作流的图像连线恢复后再整理端口。动态端口显示中英双语名称。插件不限制参考图数量；Qwen-Image-2.1 官方说明支持最多10张，更多图片的效果与可运行性取决于模型和显存。更新后重启 ComfyUI 并刷新浏览器。
