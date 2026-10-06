# 🛡️ Binyuan Sampler V2.2 EN - bilingual (English / 中文) build
# Includes native Qwen-Image-2.1 conditioning and reference-image support.
# This node uses its own key (BinyuanUltimateSamplerEN) so it can live next to the Chinese version.
import os
import re
import random
import sys
import gc
import json
import time
import inspect
import importlib.util
import torch
import folder_paths
import nodes
import comfy.utils
import comfy.sd
import comfy.sample
import comfy.samplers
import comfy.latent_formats
import comfy.ldm.modules.attention
import latent_preview
import torch.nn.functional as F
from PIL import Image
import numpy as np

# ========== 全局缓存 ==========
_FILE_CACHE = {}
_FILE_CACHE_TIMESTAMP = {}

def get_files_cached(folder):
    global _FILE_CACHE, _FILE_CACHE_TIMESTAMP
    if folder not in folder_paths.folder_names_and_paths:
        return []
    folder_paths_list = folder_paths.folder_names_and_paths[folder][0]
    cache_key = f"{folder}_{'_'.join(folder_paths_list)}"
    now = time.time()
    if cache_key in _FILE_CACHE and (now - _FILE_CACHE_TIMESTAMP.get(cache_key, 0)) < 5:
        return _FILE_CACHE[cache_key]
    files = sorted(list(set(folder_paths.get_filename_list(folder))))
    _FILE_CACHE[cache_key] = files
    _FILE_CACHE_TIMESTAMP[cache_key] = now
    return files

# ========== CLIP 类型规范化 ==========
# 将用户可见类型映射到 ComfyUI CLIPType 枚举对应名称（CLIPLoader/DualCLIPLoader 内部用 getattr(CLIPType, name.upper()) 解析）。
# pid / pixl 在 ComfyUI 中并非独立 CLIP 类型，而是 PixelDiffusion 解码器架构（comfy.ldm.pixeldit.pid），
# 其文本编码器与 PixelDiT 家族同源，因此归一化到 pixeldit，保证可正常加载。
CLIP_TYPE_CANONICAL = {
    "ace": "ace", "boogu": "boogu", "chroma": "chroma", "cogvideox": "cogvideox",
    "cosmos": "cosmos", "flux": "flux", "flux2": "flux2", "hidream": "hidream",
    "hunyuan_image": "hunyuan_image", "ideogram4": "ideogram4", "krea2": "krea2",
    "lens": "lens", "longcat_image": "longcat_image", "ltxv": "ltxv",
    "lumina2": "lumina2", "mochi": "mochi", "omnigen2": "omnigen2", "ovis": "ovis",
    "pid": "pixeldit", "pixart": "pixart", "pixeldit": "pixeldit", "pixl": "pixeldit",
    "qwen_image": "qwen_image", "qwen_image_2.1": "qwen_image", "sd3": "sd3", "stable_audio": "stable_audio",
    "stable_cascade": "stable_cascade", "stable_diffusion": "stable_diffusion", "wan": "wan",
}

# 需要附加 Flux 风格 guidance 的类型
GUIDANCE_CLIP_TYPES = {"flux", "flux2", "pid", "pixl", "boogu", "ideogram4", "pixeldit"}

# ============================ 动态提示词 {a|b|c} 抽签 ============================
# 前端（网页界面）的抽签只在"提交队列"时执行，而且只作用于**直接写在 widget 里**的
# 文本：连线传进来的字符串、以及用 API / 外部程序提交的请求都不会被展开，括号和竖线
# 会原样送进文本编码器，模型就会把所有选项一起画出来（典型症状：一张图好几个人）。
# 这里补一层服务端抽签，两种来源都能生效。
def binyuan_expand_dynamic_prompt(text, seed=0, label="正面"):
    """展开 {a|b|c}，行为与 ComfyUI 前端 processDynamicPrompt 对齐。

    - { } 必须成对；不配对时**不抽签**并给出警告（避免把 JSON 之类的文本吃掉）
    - 选项之间只能用半角竖线 |，支持嵌套
    - 每次运行重新随机（与网页端抽签一致），不看 seed
    """
    if not isinstance(text, str) or "{" not in text:
        return text
    if text.count("{") != text.count("}"):
        print("[binyuan] %s提示词花括号不配对（{ %d 个 / } %d 个），已跳过抽签，原文送进模型。"
              % (label, text.count("{"), text.count("}")))
        return text

    s = re.sub(r"/\*[\s\S]*?\*/|//.*", "", text)   # 与前端一致：/* */ 和 // 会被当注释删掉
    # 每次运行重新随机抽签（与 ComfyUI 前端行为一致）。
    # 注意：这里**不绑定 seed** —— 否则 seed 固定的工作流每次都抽到同一个选项，
    # 表面上就像"抽签没生效/没变化"。
    rng = random.Random()

    def walk(txt):
        out, i = "", 0
        while i < len(txt):
            c = txt[i]
            i += 1
            if c == "\\":
                if i < len(txt):
                    out += "\\" + txt[i]
                    i += 1
                else:
                    out += "\\"
                continue
            if c != "{":
                out += c
                continue
            opts, cur, depth = [], "", 0
            while i < len(txt):
                a = txt[i]
                i += 1
                if a == "\\":
                    if i < len(txt):
                        cur += "\\" + txt[i]
                        i += 1
                    else:
                        cur += "\\"
                    continue
                if a == "{":
                    depth += 1
                elif a == "}":
                    if depth == 0:
                        break
                    depth -= 1
                elif a == "|" and depth == 0:
                    opts.append(cur)
                    cur = ""
                    continue
                cur += a
            opts.append(cur)
            out += walk(opts[rng.randrange(len(opts))])
        return out

    result = re.sub(r"\\([{}|])", r"\1", walk(s))
    if result != text:
        print("[binyuan] %s提示词抽签: %s%s"
              % (label, result[:110], "..." if len(result) > 110 else ""))
    return result
# ===============================================================================

def normalize_clip_type(clip_type):
    """把用户选择的 CLIP_类型 归一化为 ComfyUI 可识别的 CLIPType 名称。"""
    if not clip_type:
        return "stable_diffusion"
    key = clip_type.lower()
    return CLIP_TYPE_CANONICAL.get(key, "stable_diffusion")

# ========== GGUF 支持 ==========
# 复用 ComfyUI-GGUF（City96）的加载器，让本节点可直接选择 .gguf 扩散模型 / CLIP。
# 不复制其实现，而是按路径动态加载 ComfyUI-GGUF 的 nodes 模块，保证随上游同步更新。
_GGUF_NODES_CACHE = None

def _register_gguf_folders():
    """幂等地把 .gguf 文件登记到 unet_gguf / clip_gguf 目录键（与 ComfyUI-GGUF 一致），
    保证即便加载顺序不同也能在下拉框列出 .gguf 文件。"""
    def _add(key, targets):
        base = folder_paths.folder_names_and_paths.get(key, ([], {}))
        base = base[0] if isinstance(base[0], (list, set, tuple)) else []
        target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
        orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
        folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    _add("unet_gguf", ["diffusion_models", "unet"])
    _add("clip_gguf", ["text_encoders", "clip"])

_register_gguf_folders()

def _get_gguf_nodes():
    """按路径加载 ComfyUI-GGUF 的 nodes 模块并缓存。返回模块含
    UnetLoaderGGUF / CLIPLoaderGGUF / DualCLIPLoaderGGUF / GGMLOps /
    gguf_sd_loader / gguf_clip_loader / GGUFModelPatcher。"""
    global _GGUF_NODES_CACHE
    if _GGUF_NODES_CACHE is not None:
        return _GGUF_NODES_CACHE
    custom_nodes_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gguf_dir = os.path.join(custom_nodes_dir, "ComfyUI-GGUF")
    init_path = os.path.join(gguf_dir, "__init__.py")
    if not os.path.isfile(init_path):
        raise RuntimeError(
            "ComfyUI-GGUF custom node not found (expected at custom_nodes/ComfyUI-GGUF). "
            "Install ComfyUI-GGUF to load .gguf models. / 未找到 ComfyUI-GGUF 节点，加载 .gguf 模型需先安装它。"
        )
    pkg_name = "binyuan_gguf_loader"
    spec = importlib.util.spec_from_file_location(
        pkg_name, init_path, submodule_search_locations=[gguf_dir]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    _GGUF_NODES_CACHE = sys.modules[pkg_name + ".nodes"]
    return _GGUF_NODES_CACHE

def load_gguf_diffusion_model(model_path):
    """通过 ComfyUI-GGUF 加载 .gguf 扩散模型，返回 ModelPatcher。
    GGUF 自带量化，不再额外套用 权重精度 的 dtype。"""
    gn = _get_gguf_nodes()
    ops = gn.GGMLOps()
    sd, extra = gn.gguf_sd_loader(model_path)
    kwargs = {}
    if "metadata" in inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters:
        kwargs["metadata"] = extra.get("metadata", {})
    model = comfy.sd.load_diffusion_model_state_dict(
        sd, model_options={"custom_operations": ops}, **kwargs
    )
    if model is None:
        raise RuntimeError(f"Unrecognized GGUF model type / 无法识别 GGUF 模型类型: {model_path}")
    model = gn.GGUFModelPatcher.clone(model)
    return model

def load_gguf_clip(clip_paths, clip_type_str):
    """通过 ComfyUI-GGUF 加载 .gguf CLIP（支持单/双编码器），返回 CLIP 对象。"""
    gn = _get_gguf_nodes()
    inst = gn.CLIPLoaderGGUF()
    clip_type = getattr(comfy.sd.CLIPType, clip_type_str.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
    paths = list(clip_paths)
    clip_data = inst.load_data(paths)
    return inst.load_patcher(paths, clip_type, clip_data)

class ReferenceImageInputs(dict):
    # Keep the legacy node schema/port IDs while accepting dynamically added ports.
    def __contains__(self, key):
        return super().__contains__(key) or bool(re.fullmatch(r"upstream_image_[1-9][0-9]*", key))

    def __getitem__(self, key):
        if re.fullmatch(r"upstream_image_[1-9][0-9]*", key):
            return ("IMAGE",)
        return super().__getitem__(key)


class BinyuanUltimateSamplerEN:
    # ===== Bilingual (EN/ZH) compatibility layer =====
    # Widget labels are English; combo values are written as "English ｜ 中文".
    # This table maps them back to the internal canonical keys/values, so all the
    # sampling logic below consumes canonical keys for both UI languages.
    EN_KEY_MAP = {
        "load_mode": "加载模式",
        "checkpoint": "Checkpoint",
        "diffusion_model": "扩散模型",
        "clip_1": "CLIP_1",
        "clip_2": "CLIP_2",
        "clip_type": "CLIP_类型",
        "vae": "VAE",
        "weight_precision": "权重精度",
        "chaining_mode": "串联模式",
        "upstream_image_handling": "上游图像处理",
        "latent_source": "Latent输入源",
        "positive_prompt": "正面提示词",
        "negative_prompt": "负面提示词",
        "size_preset": "尺寸助手",
        "width": "宽度",
        "height": "高度",
        "batch_size": "生成数量",
        "seed": "seed",
        "steps": "步数",
        "cfg": "CFG",
        "flux_guidance": "Flux引导",
        "sampler": "采样算法",
        "scheduler": "调度器",
        "denoise": "重绘强度",
        "lora_json": "lora_json",
        "lora_list": "LORA_LIST",
        "cleanup_vram": "自动清理显存",
        "external_model": "外部模型",
        "external_clip": "外部CLIP",
        "external_vae": "外部VAE",
        "external_latent": "外部Latent",
        "external_sigmas": "外部Sigmas",
        "external_positive": "外部正面条件",
        "external_negative": "外部负面条件",
        "lora_settings": "LoRA设置",
        "upstream_image_1": "上游图像_1",
        "upstream_image_2": "上游图像_2",
        "upstream_image_3": "上游图像_3",
        "qwen_reference_resolution": "Qwen参考分辨率",
        "qwen_edit_mode": "Qwen编辑模式",
        "attention_backend": "注意力后端"
    }

    EN_VALUE_MAP = {
        "加载模式": {
            "Checkpoint (whole file) ｜ 整包Checkpoint": "整包Checkpoint",
            "Separate (Flux/SD3/Diffusion) ｜ 分离式(Flux/SD3/扩散)": "分离式(Flux/SD3/扩散)"
        },
        "串联模式": {
            "Own model ｜ 使用自身模型": "使用自身模型",
            "Inherit upstream ｜ 继承上游模型": "继承上游模型",
            "Auto detect ｜ 自动检测": "自动检测"
        },
        "上游图像处理": {
            "Re-encode with VAE ｜ 重新VAE编码": "重新VAE编码",
            "Use directly as latent ｜ 直接作为Latent": "直接作为Latent",
            "Auto ｜ 自动选择": "自动选择"
        },
        "Latent输入源": {
            "Empty latent ｜ 空Latent": "空Latent",
            "External latent first ｜ 外部Latent优先": "外部Latent优先",
            "Upstream image first ｜ 上游图像优先": "上游图像优先",
            "Stitch upstream images ｜ 上游图像拼接": "上游图像拼接"
        },
        "尺寸助手": {
            "Custom ｜ 自定义": "自定义"
        },
        "Qwen编辑模式": {
            "Auto ｜ 自动": "自动",
            "Native reference edit ｜ 原生参考编辑": "原生参考编辑",
            "VAE redraw ｜ VAE传统重绘": "VAE传统重绘"
        },
        "注意力后端": {
            "Auto safe ｜ 自动稳定": "自动稳定",
            "Core default ｜ 内核默认": "内核默认",
            "Comfy Kitchen ｜ Comfy Kitchen": "Comfy Kitchen",
            "Sub-quadratic ｜ 分块省显存": "分块省显存"
        }
    }

    EN_VALUE_KEYWORDS = {
        "加载模式": [
            [
                "separate",
                "分离式(Flux/SD3/扩散)"
            ],
            [
                "checkpoint",
                "整包Checkpoint"
            ]
        ],
        "串联模式": [
            [
                "inherit",
                "继承上游模型"
            ],
            [
                "own",
                "使用自身模型"
            ],
            [
                "auto",
                "自动检测"
            ]
        ],
        "上游图像处理": [
            [
                "vae",
                "重新VAE编码"
            ],
            [
                "latent",
                "直接作为Latent"
            ],
            [
                "auto",
                "自动选择"
            ]
        ],
        "Latent输入源": [
            [
                "empty",
                "空Latent"
            ],
            [
                "external",
                "外部Latent优先"
            ],
            [
                "stitch",
                "上游图像拼接"
            ],
            [
                "upstream",
                "上游图像优先"
            ]
        ],
        "尺寸助手": [
            [
                "custom",
                "自定义"
            ]
        ]
    }

    @classmethod
    def _translate_kwargs(cls, kwargs):
        """EN widget keys/values -> canonical keys/values used by run()."""
        out = {}
        for k, v in kwargs.items():
            key = cls.EN_KEY_MAP.get(k, k)
            if re.fullmatch(r"upstream_image_[1-9][0-9]*", k):
                key = "上游图像_" + k.rsplit("_", 1)[1]
            if isinstance(v, str):
                vmap = cls.EN_VALUE_MAP.get(key)
                if vmap:
                    if v in vmap:
                        v = vmap[v]
                    else:
                        low = v.lower()
                        for word, canon in cls.EN_VALUE_KEYWORDS.get(key, []):
                            if word in low:
                                v = canon
                                break
            out[key] = v
        return out

    @classmethod
    def INPUT_TYPES(s):
        def get_files(folder):
            return get_files_cached(folder)

        all_clip_files = []
        for folder in ["clip", "text_encoders", "CLIP", "clip_gguf"]:
            if folder in folder_paths.folder_names_and_paths:
                all_clip_files.extend(get_files_cached(folder))
        all_clip_files = sorted(list(set(all_clip_files)))

        # diffusion models: diffusion_models / unet / unet_gguf (incl. .gguf), de-duplicated
        all_diffusion_files = sorted(list(set(
            get_files("diffusion_models") + get_files("unet") + get_files("unet_gguf")
        )))

        # every type supported by this plugin (aligned with ComfyUI CLIPType)
        CLIP_TYPES = [
            "ACE", "boogu", "chroma", "cogvideox", "cosmos", "flux", "flux2",
            "hidream", "hunyuan_image", "ideogram4", "krea2", "lens", "longcat_image",
            "LTXV", "lumina2", "mochi", "omnigen2", "ovis", "pid", "PixArt",
            "pixeldit", "pixl", "qwen_image", "qwen_image_2.1", "sd3", "SD3", "stable_audio",
            "stable_cascade", "stable_diffusion", "wan"
        ]

        PRESET_SIZES = ["Custom ｜ 自定义", "512x512", "512x768", "768x512", "768x768", "768x1024", "1024x768", "1024x1024", "1024x1280", "1280x1024", "1152x1152", "1152x1536", "1536x1152", "1216x832", "832x1216", "1280x1280", "1280x1920", "1920x1280", "1536x1536", "1536x2048", "2048x1536", "2048x2048", "1024x2048", "2048x1024", "4096x4096"]

        SAMPLERS = comfy.samplers.KSampler.SAMPLERS
        SCHEDULERS = comfy.samplers.KSampler.SCHEDULERS

        return {
            "required": {
                "load_mode": (["Checkpoint (whole file) ｜ 整包Checkpoint", "Separate (Flux/SD3/Diffusion) ｜ 分离式(Flux/SD3/扩散)"], {"default": "Checkpoint (whole file) ｜ 整包Checkpoint"}),
                "checkpoint": (["None"] + get_files("checkpoints"), {"default": "None"}),
                "diffusion_model": (["None"] + all_diffusion_files, {"default": "None"}),
                "clip_1": (["None"] + all_clip_files, {"default": "None"}),
                "clip_2": (["None"] + all_clip_files, {"default": "None"}),
                "clip_type": (CLIP_TYPES, {"default": "flux"}),
                "vae": (["baked_vae"] + get_files("vae"), {"default": "baked_vae"}),
                "weight_precision": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2", "nvfp4", "pid", "pixl", "bf16", "fp16"], {"default": "default"}),
                "chaining_mode": (["Own model ｜ 使用自身模型", "Inherit upstream ｜ 继承上游模型", "Auto detect ｜ 自动检测"], {"default": "Inherit upstream ｜ 继承上游模型"}),
                "upstream_image_handling": (["Re-encode with VAE ｜ 重新VAE编码", "Use directly as latent ｜ 直接作为Latent", "Auto ｜ 自动选择"], {"default": "Re-encode with VAE ｜ 重新VAE编码"}),
                "latent_source": (["Empty latent ｜ 空Latent", "External latent first ｜ 外部Latent优先", "Upstream image first ｜ 上游图像优先", "Stitch upstream images ｜ 上游图像拼接"], {"default": "Upstream image first ｜ 上游图像优先"}),
                "positive_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "masterpiece, best quality, 1girl",
                    "tooltip": "支持 {a|b|c} 动态提示词抽签：每次点 Run 重新随机抽一个。网页点 Run 由前端抽签；连线传入或 API/外部调用由节点在服务端抽签。{ } 必须成对，选项之间用半角竖线 | 。"}),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "",
                    "tooltip": "同上，负向提示词同样支持 {a|b|c} 抽签。"}),
                "size_preset": (PRESET_SIZES, {"default": "Custom ｜ 自定义"}),
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "flux_guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "sampler": (SAMPLERS, {"default": "euler"}),
                "scheduler": (SCHEDULERS, {"default": "simple"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "lora_json": ("STRING", {"multiline": True, "default": "[]"}),
                "lora_list": (["None"] + get_files("loras") + get_files("lora"),),
                "cleanup_vram": ("BOOLEAN", {"default": False, "label": "Clear VRAM after run / 生成后清理显存"}),
            },
            "optional": ReferenceImageInputs({
                "external_model": ("MODEL",),
                "external_clip": ("CLIP",),
                "external_vae": ("VAE",),
                "external_latent": ("LATENT",),
                "external_sigmas": ("SIGMAS",),
                "external_positive": ("CONDITIONING",),
                "external_negative": ("CONDITIONING",),
                "lora_settings": ("BINYUAN_LORA_SETTINGS",),
                "upstream_image_1": ("IMAGE",),
                "upstream_image_2": ("IMAGE",),
                "upstream_image_3": ("IMAGE",),
                "qwen_reference_resolution": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 32,
                    "tooltip": "Qwen-Image-2.1 reference pixel budget (resolution squared); 0 keeps original size, aligned to 32. Editing output follows the first reference. / Qwen-Image-2.1 参考图像素预算（分辨率平方）；0 保持原尺寸并对齐32，编辑输出尺寸跟随第一张参考图。"}),
                "qwen_edit_mode": (["Auto ｜ 自动", "Native reference edit ｜ 原生参考编辑", "VAE redraw ｜ VAE传统重绘"], {
                    "default": "Auto ｜ 自动",
                    "tooltip": "Auto keeps existing workflows compatible. Native reference edit uses Qwen-Image-2.1 official reference conditioning and an empty latent (denoise is fixed to 1). VAE redraw encodes the upstream image as the initial latent, preserves its canvas (32-pixel alignment), and enables adjustable denoise. / 自动模式兼容旧工作流；原生参考编辑使用官方参考条件与空Latent（重绘强度固定为1）；VAE传统重绘把上游图像编码为初始Latent，保持画布（对齐32）并允许调节重绘强度。"
                }),
                "attention_backend": (["Auto safe ｜ 自动稳定", "Core default ｜ 内核默认", "Comfy Kitchen ｜ Comfy Kitchen", "Sub-quadratic ｜ 分块省显存"], {
                    "default": "Auto safe ｜ 自动稳定",
                    "tooltip": "Auto safe uses Comfy Kitchen for internally loaded Krea2/Qwen-Image-2.1 models when available, otherwise ComfyUI's sub-quadratic backend; other models and externally patched models retain their backend. Core default follows ComfyUI startup flags. / 自动稳定为内部加载的Krea2/Qwen-Image-2.1模型选用可用的Comfy Kitchen，否则使用内核分块后端；其他模型与已外接修补的模型保持原后端。内核默认遵循启动参数。"
                }),
                "advanced_sampling": ("BOOLEAN", {"default": False, "tooltip": "Advanced step range; steps becomes total schedule steps, denoise is ignored. / 高级分段：步数为总步数，忽略重绘强度。"}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000, "tooltip": "Zero-based start; 10 starts after the first 10 steps. / 从0计数；10表示跳过前10步。"}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000, "tooltip": "Stop boundary, clamped to total steps. / 结束边界；超过总步数即运行到最后。"}),
                "add_noise": ("BOOLEAN", {"default": True, "tooltip": "Disable when continuing an upstream noisy LATENT. / 接续上游带噪LATENT时关闭；成图重绘时开启。"}),
                "return_with_leftover_noise": ("BOOLEAN", {"default": False, "tooltip": "Enable for the first segment; pass LATENT, not denoised_latent, to the next sampler. / 第一段开启，把LATENT传给下一段，不要传去噪Latent。"}),
                "noise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Initial random-noise multiplier in both modes: 1=normal, 0=no added initial noise. Advanced Add noise OFF overrides this. Does not change sigma schedule or remove existing noise. / 两种模式均有效：1为标准新增噪声，0不新增初始噪声；高级模式关闭添加噪声时本项无效。不改变采样日程，不消除输入已有噪声。"}),
            })
        }

    RETURN_TYPES = ("IMAGE", "MODEL", "CLIP", "VAE", "LATENT", "CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("IMAGE / 图像", "MODEL / 模型", "CLIP", "VAE", "LATENT / Latent", "POSITIVE / 正面条件", "NEGATIVE / 负面条件", "DENOISED / 降噪Latent")
    FUNCTION = "run"
    CATEGORY = "Binyuan"
    DESCRIPTION = "All-in-one bilingual sampler: loads a whole checkpoint or a split stack (diffusion model + dual CLIP + VAE), encodes prompts, samples and decodes - and also outputs model/CLIP/VAE/latent/conditioning for chaining. Supports Flux / SD3 / Wan / Qwen-Image / Qwen-Image-2.1 / Krea2 / Z-Image / LTXV / Hunyuan / Lumina2 ... English + Chinese UI. ｜ 一体化中英双语采样器：整包 Checkpoint 或分离式（扩散模型+CLIP+VAE）加载、条件编码、采样、解码出图，并把模型/CLIP/VAE/Latent/条件作为端口输出方便串联。支持 Flux / SD3 / Wan / Qwen-Image / Qwen-Image-2.1 / Krea2 / Z-Image / LTXV / Hunyuan / Lumina2 等架构。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return None

    def get_image_size(self, image_tensor):
        if image_tensor is None:
            return 0, 0
        if len(image_tensor.shape) == 4:
            _, h, w, _ = image_tensor.shape
        else:
            h, w = image_tensor.shape[:2]
        return w, h

    def encode_image_to_latent(self, vae, image, width, height):
        try:
            if len(image.shape) == 4:
                image = image.permute(0, 3, 1, 2)
                image = F.interpolate(image, size=(height, width), mode='bilinear')
                image = image.permute(0, 2, 3, 1)
            return nodes.VAEEncode().encode(vae, image)[0]
        except Exception as e:
            print(f"[ERROR] image encode failed: {e}")
            return None

    def stitch_images_horizontal(self, images_list, target_h):
        """将多张上游图像横向拼接为单张画布（用于「上游图像拼接」模式）。"""
        parts = []
        for img in images_list:
            if img is None:
                continue
            t = img
            if t.dim() == 4:
                t = t[0]  # 取首张
            if t.dim() != 3:
                continue
            # 按目标高度等比缩放，宽度对齐到 8
            h0 = t.shape[0]
            if h0 <= 0:
                continue
            scale = float(target_h) / float(h0)
            new_w = max(8, int(round(t.shape[1] * scale) // 8 * 8))
            x = t.permute(2, 0, 1).unsqueeze(0).float()  # [1,C,H,W]
            x = F.interpolate(x, size=(int(target_h), new_w), mode='bilinear', align_corners=False)
            parts.append(x)
        if not parts:
            return None
        stitched = torch.cat(parts, dim=3)  # [1,C,H,W_total]
        out = stitched.permute(0, 2, 3, 1)  # [1,H,W_total,C]
        return out

    def get_latent_format(self, model):
        """安全读取模型 latent 格式（通道数 / 空间下采样比 / 维度）。"""
        try:
            return model.get_model_object("latent_format")
        except Exception:
            return None

    def configure_attention_backend(self, model, selection, internally_loaded):
        """Select a model-local attention implementation without changing global flags."""
        attention = comfy.ldm.modules.attention
        if selection == "自动稳定":
            if not internally_loaded:
                return model  # Preserve upstream ModelAttentionBackend patches.
            model_class = type(getattr(model, "model", None)).__name__
            if model_class not in ("Krea2", "QwenImage21"):
                return model
            available = bool(getattr(attention, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", False))
            backend_name = "comfy_kitchen_int8" if available else "sub_quad"
        elif selection == "内核默认":
            return model
        elif selection == "Comfy Kitchen":
            backend_name = "comfy_kitchen_int8"
        elif selection == "分块省显存":
            backend_name = "sub_quad"
        else:
            return model

        getter = getattr(attention, "get_attention_function", None)
        backend_function = getter(backend_name, None) if callable(getter) else None
        if backend_function is None:
            backend_function = getattr(attention, {
                "comfy_kitchen_int8": "attention_comfy_kitchen_int8",
                "sub_quad": "attention_sub_quad",
            }[backend_name], None)
        if backend_name == "comfy_kitchen_int8" and not getattr(
            attention, "COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE", False
        ):
            backend_function = None
        if backend_function is None or not hasattr(model, "set_model_optimized_attention"):
            if selection != "自动稳定":
                raise RuntimeError(
                    f"Attention backend {selection} is unavailable in this ComfyUI installation. "
                    f"/ 当前ComfyUI环境不支持注意力后端：{selection}。"
                )
            print(f"[WARN] safe attention backend {backend_name} unavailable; using ComfyUI default")
            return model
        patched = model.clone()
        patched.set_model_optimized_attention(backend_function)
        print(f"[INFO] model-local attention backend: {backend_name} (model={type(model.model).__name__})")
        return patched

    def empty_latent_for_model(self, model, width, height, batch):
        """按模型 latent 格式创建空 Latent（通道数与下采样比与模型匹配）。"""
        fmt = self.get_latent_format(model)
        channels = getattr(fmt, "latent_channels", 4) if fmt else 4
        ratio = getattr(fmt, "spacial_downscale_ratio", 8) if fmt else 8
        dims = getattr(fmt, "latent_dimensions", 2) if fmt else 2
        h = max(1, height // int(ratio))
        w = max(1, width // int(ratio))
        device = comfy.model_management.intermediate_device()
        dtype = comfy.model_management.intermediate_dtype()
        if dims == 3:
            # 视频模型：单帧
            latent = torch.zeros([batch, int(channels), 1, h, w], device=device, dtype=dtype)
        else:
            latent = torch.zeros([batch, int(channels), h, w], device=device, dtype=dtype)
        print(f"[INFO] empty latent: {latent.shape} (channels={channels}, ratio={ratio}, dims={dims})")
        return {"samples": latent, "downscale_ratio_spacial": int(ratio)}

    def validate_vae(self, model, vae):
        """校验 VAE 与模型 latent 通道是否匹配，不匹配给出明确错误。"""
        if vae is None:
            return
        fmt = self.get_latent_format(model)
        model_ch = getattr(fmt, "latent_channels", None) if fmt else None
        vae_ch = getattr(vae, "latent_channels", None)
        if model_ch is not None and vae_ch is not None and int(model_ch) != int(vae_ch):
            raise RuntimeError(
                f"VAE does not match the model: model latent channels={model_ch}, current VAE channels={vae_ch}. "
                f"Pick the matching VAE file in the VAE dropdown. / VAE 与模型不匹配，请在 VAE 下拉框选择对应的 VAE 文件。"
            )
        vae_ratio = getattr(vae, "downscale_ratio", None)
        print(f"[INFO] VAE 校验通过 (VAE通道={vae_ch}, 下采样比={vae_ratio})")

    def get_model_name(self, model_obj):
        if model_obj is None:
            return "无"
        if hasattr(model_obj, 'model_config') and hasattr(model_obj.model_config, '_name'):
            return model_obj.model_config._name
        if hasattr(model_obj, 'model_type'):
            return str(model_obj.model_type)
        if hasattr(model_obj, '__class__'):
            return model_obj.__class__.__name__
        return "未知模型"

    def get_clip_name(self, clip_obj):
        if clip_obj is None:
            return "无"
        if hasattr(clip_obj, 'model_config') and hasattr(clip_obj.model_config, '_name'):
            return clip_obj.model_config._name
        if hasattr(clip_obj, '__class__'):
            return clip_obj.__class__.__name__
        return "未知CLIP"

    def get_vae_name(self, vae_obj):
        if vae_obj is None:
            return "无"
        if hasattr(vae_obj, '__class__'):
            return vae_obj.__class__.__name__
        return "未知VAE"

    def load_clip_internal(self, clip_1, clip_2, clip_type_canon):
        """加载 CLIP，自动按文件后缀在标准加载器与 GGUF 加载器间分流。
        任一 CLIP 文件为 .gguf 即整体走 GGUF 加载器（混用 scaled_fp8 与 gguf 不被支持）。"""
        if not clip_1 or clip_1 == "None":
            return None
        use_gguf = clip_1.lower().endswith(".gguf") or (
            clip_2 and clip_2 != "None" and clip_2.lower().endswith(".gguf")
        )
        names = [clip_1]
        if clip_2 and clip_2 != "None":
            names.append(clip_2)

        if use_gguf:
            paths = []
            for c in names:
                p = (folder_paths.get_full_path("clip", c) or
                     folder_paths.get_full_path("text_encoders", c) or
                     folder_paths.get_full_path("clip_gguf", c))
                if not p:
                    raise FileNotFoundError(f"CLIP not found / 找不到 CLIP: {c}")
                paths.append(p)
            clip = load_gguf_clip(paths, clip_type_canon)
            print(f"[INFO] loaded GGUF CLIP: {', '.join(names)} (type: {clip_type_canon})")
            return clip

        if clip_2 and clip_2 != "None":
            loader = nodes.DualCLIPLoader()
            result = loader.load_clip(clip_1, clip_2, clip_type_canon)
        else:
            loader = nodes.CLIPLoader()
            result = loader.load_clip(clip_1, clip_type_canon)
        if result is not None and len(result) > 0:
            print(f"[INFO] loaded CLIP: {clip_1} (type: {clip_type_canon})")
            return result[0]
        return None

    def detect_model_type(self, lora_name):
        """检测 LoRA 类型，用于选择加载方式"""
        if lora_name is None:
            return "standard"
        lora_name_lower = lora_name.lower()
        if "klein" in lora_name_lower or "flux2" in lora_name_lower:
            return "klein"
        if "zimage" in lora_name_lower or "lumina2" in lora_name_lower or "turbo" in lora_name_lower:
            return "zimage"
        return "standard"

    def cleanup_vram(self):
        """清理显存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("[INFO] VRAM cleared")

    def parse_lora_config(self, lora_json_str):
        """解析 LoRA 配置。兼容多种写法，保证任意内核下都能手动填写：
        - JSON 数组：[{"n":"x.safetensors","s":0.8,"sm":0.8,"sc":0.8,"e":true}, ...]
        - 单个 JSON 对象：{"n":"x.safetensors","s":0.8}
        - 简单逐行文本（每行一个 LoRA，# 或 // 开头为注释）：
            x.safetensors
            x.safetensors:0.8
            x.safetensors|0.8
        字段：n/name=文件名，s/strength=统一强度，sm=模型强度，sc=CLIP强度，e=是否启用。
        """
        if not lora_json_str:
            return []
        s = str(lora_json_str).strip()
        if not s or s in ("[]", "{}"):
            return []

        # 1) 优先按 JSON 解析
        try:
            data = json.loads(s)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
        except (json.JSONDecodeError, ValueError):
            pass

        # 2) 回退：逐行简单解析（即便前端 JS 全部失效也能手填）
        result = []
        for line in s.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            name = line
            strength = 1.0
            m = re.match(r"^(.+?)\s*[:|,]\s*(-?\d+(?:\.\d+)?)\s*$", line)
            if m:
                name = m.group(1).strip()
                strength = float(m.group(2))
            if name and name != "None":
                result.append({"n": name, "s": strength, "e": True})
        return result

    def load_loras(self, model, clip, lora_list):
        """统一的 LoRA 加载入口，支持多 LoRA 叠加。
        只依赖 ComfyUI 多年稳定的公开 API（comfy.utils.load_torch_file /
        comfy.sd.load_lora_for_models），不触碰易变的内部接口，保证跨内核可用。
        """
        loaded_count = 0
        for idx, lora in enumerate(lora_list):
            if not isinstance(lora, dict):
                continue
            if not lora.get("e", True):
                continue
            lora_name = lora.get("n") or lora.get("name")
            if not lora_name or lora_name == "None":
                continue

            # 强度：s 为统一强度；sm/sc 可分别覆盖模型/CLIP 强度
            strength = float(lora.get("s", lora.get("strength", 1.0)))
            sm = lora.get("sm")
            sc = lora.get("sc")
            sm = float(sm) if sm is not None else strength
            sc = float(sc) if sc is not None else strength

            # 查找文件（同时兼容 loras / lora 两个目录名）
            lora_path = folder_paths.get_full_path("loras", lora_name)
            if not lora_path:
                lora_path = folder_paths.get_full_path("lora", lora_name)
            if not lora_path or not os.path.exists(lora_path):
                print(f"[WARN] LoRA file not found: {lora_name}")
                continue

            print(f"[INFO] 加载 LoRA [{idx+1}]: {lora_name} (模型强度={sm}, CLIP强度={sc})")
            try:
                lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
                model, clip = comfy.sd.load_lora_for_models(model, clip, lora_sd, sm, sc)
                loaded_count += 1
                print(f"[INFO] LoRA loaded: {lora_name}")
            except Exception as e:
                print(f"[WARN] LoRA load failed {lora_name}: {e}")

        return model, clip, loaded_count

    def run(self, **kwargs):
        print("[INFO] Binyuan Sampler V2.2 EN (bilingual)")
        # ---- English UI -> internal canonical keys/values ----
        kwargs = self._translate_kwargs(kwargs)
        
        # ========== 获取并解析参数 ==========
        加载模式 = kwargs.get("加载模式", "整包Checkpoint")
        Checkpoint = kwargs.get("Checkpoint", "None")
        扩散模型 = kwargs.get("扩散模型", "None")
        CLIP_1 = kwargs.get("CLIP_1", "None")
        CLIP_2 = kwargs.get("CLIP_2", "None")
        CLIP_类型 = kwargs.get("CLIP_类型", "flux")
        CLIP_类型_规范 = normalize_clip_type(CLIP_类型)
        VAE = kwargs.get("VAE", "baked_vae")
        权重精度 = kwargs.get("权重精度", "default")
        串联模式 = kwargs.get("串联模式", "继承上游模型")
        上游图像处理 = kwargs.get("上游图像处理", "重新VAE编码")
        Latent输入源 = kwargs.get("Latent输入源", "上游图像优先")
        正面提示词 = kwargs.get("正面提示词", "masterpiece, best quality, 1girl")
        负面提示词 = kwargs.get("负面提示词", "")
        宽度 = int(kwargs.get("宽度", 1024))
        高度 = int(kwargs.get("高度", 1024))
        生成数量 = int(kwargs.get("生成数量", 1))
        seed = int(kwargs.get("seed", 0))
        # ---- 动态提示词 {a|b|c} 服务端抽签：连线传入的文本、API/外部调用都能生效 ----
        正面提示词 = binyuan_expand_dynamic_prompt(正面提示词, seed, "正面")
        负面提示词 = binyuan_expand_dynamic_prompt(负面提示词, seed, "负面")
        步数 = int(kwargs.get("步数", 20))
        CFG = float(kwargs.get("CFG", 1.0))
        Flux引导 = float(kwargs.get("Flux引导", 3.5))
        采样算法 = kwargs.get("采样算法", "euler")
        调度器 = kwargs.get("调度器", "simple")
        重绘强度 = float(kwargs.get("重绘强度", 1.0))
        advanced_sampling = bool(kwargs.get("advanced_sampling", False))
        start_at_step = max(0, int(kwargs.get("start_at_step", 0)))
        end_at_step = max(0, int(kwargs.get("end_at_step", 10000)))
        add_noise = bool(kwargs.get("add_noise", True))
        noise_strength = float(kwargs.get("noise_strength", 1.0))
        if not 0.0 <= noise_strength <= 2.0:
            raise ValueError("Noise strength must be between 0 and 2 / 初始噪声强度须在0到2之间")
        return_with_leftover_noise = bool(kwargs.get("return_with_leftover_noise", False))
        Qwen编辑模式 = kwargs.get("Qwen编辑模式", "自动")
        注意力后端 = kwargs.get("注意力后端", "自动稳定")
        lora_json_str = kwargs.get("lora_json", "[]")
        lora_settings = kwargs.get("LoRA设置") or []
        自动清理显存 = kwargs.get("自动清理显存", False)
        
        # ========== 获取可选外部输入 ==========
        external_model = kwargs.get("外部模型")
        external_clip = kwargs.get("外部CLIP")
        external_vae = kwargs.get("外部VAE")
        external_latent = kwargs.get("外部Latent")
        external_sigmas = kwargs.get("外部Sigmas")
        external_positive = kwargs.get("外部正面条件")
        external_negative = kwargs.get("外部负面条件")
        
        image_keys = sorted(
            (key for key in kwargs if re.fullmatch(r"上游图像_[1-9][0-9]*", key)),
            key=lambda key: int(key.rsplit("_", 1)[1]),
        )
        upstream_images = [kwargs[key] for key in image_keys if kwargs[key] is not None]
        
        print(f"[INFO] upstream images: {len(upstream_images)}")
        print(f"[INFO] chaining mode: {串联模式}")
        
        # ========== 打印外接状态 ==========
        print(f"[INFO] external model: {self.get_model_name(external_model)}")
        print(f"[INFO] external CLIP: {self.get_clip_name(external_clip)}")
        print(f"[INFO] external VAE: {self.get_vae_name(external_vae)}")
        print(f"[INFO] external latent: {'yes' if external_latent is not None else 'no'}")
        print(f"[INFO] external sigmas: {'yes' if external_sigmas is not None else 'no'}")
        print(f"[INFO] external positive conditioning: {'yes' if external_positive is not None else 'no'}")
        print(f"[INFO] external negative conditioning: {'yes' if external_negative is not None else 'no'}")
        
        use_inherit = (串联模式 == "继承上游模型")
        
        model = None
        clip = None
        vae = None
        
        try:
            # ========== 解析模型权重精度配置 ==========
            model_options = {}
            te_model_options = {}
            
            if 权重精度 == "fp8_e4m3fn":
                model_options["dtype"] = torch.float8_e4m3fn
                te_model_options["dtype"] = torch.float8_e4m3fn
            elif 权重精度 == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
                te_model_options["dtype"] = torch.float8_e4m3fn
            elif 权重精度 == "fp8_e5m2":
                model_options["dtype"] = torch.float8_e5m2
                te_model_options["dtype"] = torch.float8_e5m2
            elif 权重精度 in ["nvfp4", "pid", "pixl"]:
                model_options["dtype"] = "nvfp4"
                te_model_options["dtype"] = "nvfp4"
            elif 权重精度 == "bf16":
                model_options["dtype"] = torch.bfloat16
                te_model_options["dtype"] = torch.bfloat16
            elif 权重精度 == "fp16":
                model_options["dtype"] = torch.float16
                te_model_options["dtype"] = torch.float16

            # ========== 模型加载 ==========
            if use_inherit and external_model is not None:
                print("[INFO] 串联模式=继承上游模型，使用外接模型")
                model = external_model
                clip = external_clip if external_clip is not None else clip
                vae = external_vae if external_vae is not None else vae
                
                if clip is None and CLIP_1 and CLIP_1 != "None":
                    print(f"[INFO] 外接缺少CLIP，从内部补充: {CLIP_1} (类型: {CLIP_类型} → {CLIP_类型_规范})")
                    clip = self.load_clip_internal(CLIP_1, CLIP_2, CLIP_类型_规范)
                
                if vae is None and VAE and VAE != "baked_vae" and VAE != "None":
                    print(f"[INFO] 外接缺少VAE，从内部补充: {VAE}")
                    vae_path = folder_paths.get_full_path("vae", VAE)
                    if vae_path:
                        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
            
            else:
                print("[INFO] 串联模式=使用自身模型，使用内部配置")
                if 加载模式 == "整包Checkpoint":
                    if not Checkpoint or Checkpoint == "None":
                        raise ValueError("Please select a Checkpoint / 请选择 Checkpoint")
                    checkpoint_path = folder_paths.get_full_path("checkpoints", Checkpoint)
                    if not checkpoint_path:
                        raise FileNotFoundError(f"Checkpoint not found / 找不到: {Checkpoint}")
                    
                    try:
                        model, clip, vae = comfy.sd.load_checkpoint_guess_config(
                            checkpoint_path, output_vae=True, output_clip=True,
                            model_options=model_options, te_model_options=te_model_options
                        )[:3]
                    except Exception as e:
                        print(f"[WARN] 使用指定精度加载模型失败，降级回默认检测机制: {e}")
                        model, clip, vae = comfy.sd.load_checkpoint_guess_config(
                            checkpoint_path, output_vae=True, output_clip=True
                        )[:3]
                    print(f"[INFO] loaded checkpoint: {Checkpoint}")
                else:
                    if 扩散模型 and 扩散模型 != "None":
                        diffusion_path = (folder_paths.get_full_path("diffusion_models", 扩散模型) or
                                        folder_paths.get_full_path("unet", 扩散模型) or
                                        folder_paths.get_full_path("unet_gguf", 扩散模型))
                        if diffusion_path:
                            if 扩散模型.lower().endswith(".gguf"):
                                print(f"[INFO] 加载GGUF扩散模型: {扩散模型}")
                                model = load_gguf_diffusion_model(diffusion_path)
                            else:
                                try:
                                    model = comfy.sd.load_diffusion_model(diffusion_path, model_options=model_options)
                                except Exception as e:
                                    print(f"[WARN] 使用指定精度加载扩散模型失败，降级回默认检测机制: {e}")
                                    model = comfy.sd.load_diffusion_model(diffusion_path)
                                print(f"[INFO] loaded diffusion model: {扩散模型}")
                        else:
                            raise FileNotFoundError(f"Diffusion model not found / 找不到扩散模型: {扩散模型}")
                    else:
                        raise ValueError("Separate mode requires a diffusion model / 分离模式下必须选择扩散模型")
                    
                    if CLIP_1 and CLIP_1 != "None":
                        clip = self.load_clip_internal(CLIP_1, CLIP_2, CLIP_类型_规范)
                        if clip is None:
                            raise RuntimeError("CLIP load failed / CLIP 加载失败")
                    else:
                        raise ValueError("Separate mode requires CLIP_1 / 分离模式下必须选择 CLIP_1")
                    
                    if VAE and VAE != "baked_vae" and VAE != "None":
                        vae_path = folder_paths.get_full_path("vae", VAE)
                        if vae_path:
                            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
                            print(f"[INFO] loaded VAE: {VAE}")
            
            if model is None:
                raise RuntimeError("Model load failed / 模型加载失败")
            # 外部正、负条件都已提供时，采样阶段不需要 CLIP。
            # 这允许“外部模型 + 外部条件 + 内部/外部 VAE”的混合串联方式。
            # 只有仍需由本节点编码提示词时，才强制要求 CLIP。
            has_complete_external_conditioning = (
                use_inherit
                and external_positive is not None
                and external_negative is not None
            )
            if clip is None and not has_complete_external_conditioning:
                raise RuntimeError(
                    "CLIP is not loaded: external positive AND negative conditioning are not both connected. "
                    "Connect an external CLIP, pick an internal CLIP in clip_1, or connect both external conditioning inputs. "
                    "/ CLIP 加载失败：未同时接入外部正面与负面条件。请连接外部 CLIP、在 clip_1 选择内部 CLIP，或同时接入两路外部条件。"
                )
            if clip is None:
                print("[INFO] 已接入完整外部条件，跳过CLIP加载")
            if vae is None:
                if hasattr(model, 'model') and hasattr(model.model, 'vae'):
                    vae = model.model.vae
                    print("[INFO] 使用模型内置VAE")
                else:
                    raise RuntimeError(
                        "VAE is not loaded. Whole-file checkpoints usually contain a VAE; separate diffusion models do not. "
                        "Pick the matching VAE file in the VAE dropdown (e.g. ae.safetensors for Flux/SD3, the matching Wan VAE for Wan/older Qwen, the 64-channel RGBA VAE for Qwen-Image-2.1). "
                        "/ 未加载 VAE。整包 Checkpoint 通常自带 VAE；分离式模型不会自带 VAE，必须在 VAE 下拉框选择对应的 VAE 文件。"
                    )
            # 校验 VAE 与模型 latent 通道匹配，避免「能采样但解码失败」
            self.validate_vae(model, vae)
            model = self.configure_attention_backend(
                model, 注意力后端, internally_loaded=not (use_inherit and external_model is not None)
            )
            is_qwen21 = isinstance(self.get_latent_format(model), getattr(comfy.latent_formats, "QwenImage21", ()))
            
            # ========== LoRA 加载（多 LoRA 叠加，跨内核稳定） ==========
            lora_list = self.parse_lora_config(lora_json_str)
            if isinstance(lora_settings, list):
                lora_list.extend(lora_settings)
            if lora_list:
                print(f"[INFO] {len(lora_list)} LoRA(s) configured")
                model, clip, loaded_count = self.load_loras(model, clip, lora_list)
                print(f"[INFO] {loaded_count} LoRA(s) loaded")
            else:
                print("[INFO] no LoRA configured")
            
            # ========== 条件编码 ==========
            def encode_prompt(text, clip_type_val, guidance_val):
                if not text or not text.strip():
                    text = " "
                tokens = clip.tokenize(text)
                # 必须用 return_dict=True：boogu/omnigen2 等模型的 transformer 需要 num_tokens，
                # 而 num_tokens 来自 attention_mask；只有 return_dict 才会把 attention_mask 等额外条件带出来。
                out = clip.encode_from_tokens(tokens, return_dict=True)
                cond = out.get("cond")
                pooled = out.get("pooled_output", None)
                cond_dict = {}
                if pooled is None:
                    if cond is not None and len(cond) > 0 and len(cond[0]) > 0:
                        pooled = torch.zeros_like(cond[0][0])
                    else:
                        pooled = torch.zeros((1, 2048))
                cond_dict["pooled_output"] = pooled
                # 复制 TE 返回的额外条件（attention_mask 等），供 omnigen2/boogu/qwen 等使用
                for k, v in out.items():
                    if k in ("cond", "pooled_output"):
                        continue
                    cond_dict[k] = v
                if clip_type_val in GUIDANCE_CLIP_TYPES:
                    cond_dict["guidance"] = guidance_val
                return [[cond, cond_dict]]

            qwen_positive = qwen_negative = qwen_empty_latent = None
            qwen_has_references = bool(upstream_images)
            if Qwen编辑模式 == "自动":
                qwen_runtime_mode = "原生参考编辑" if qwen_has_references else "文生图"
            else:
                qwen_runtime_mode = Qwen编辑模式
            if is_qwen21 and qwen_runtime_mode == "VAE传统重绘" and not qwen_has_references and external_latent is None:
                raise RuntimeError(
                    "Qwen VAE redraw requires an upstream image or external latent. "
                    "/ Qwen VAE传统重绘需要连接上游图像或外部Latent。"
                )
            if is_qwen21 and not has_complete_external_conditioning:
                encoder = nodes.NODE_CLASS_MAPPINGS.get("TextEncodeQwenImage21")
                if encoder is None:
                    raise RuntimeError("TextEncodeQwenImage21 is unavailable; update ComfyUI. / 缺少 Qwen-Image-2.1 原生编码节点，请更新 ComfyUI。")
                references = upstream_images
                if Latent输入源 == "上游图像拼接" and references:
                    _, target_h = self.get_image_size(references[0])
                    references = [self.stitch_images_horizontal(references, target_h)]
                # VAE redraw deliberately keeps the source canvas. Native reference
                # mode follows the official Qwen resolution control.
                reference_resolution = 0 if qwen_runtime_mode == "VAE传统重绘" else int(kwargs.get("Qwen参考分辨率", 1024))
                encoded = encoder.execute(
                    clip, 正面提示词, 负面提示词, vae=vae,
                    resolution=reference_resolution,
                    images={f"image_{i}": image for i, image in enumerate(references, 1)},
                )
                qwen_positive, qwen_negative, qwen_empty_latent = encoded[0], encoded[1], encoded[2]

            if use_inherit and external_positive is not None:
                positive = external_positive
                print("[INFO] using external positive conditioning")
            elif is_qwen21:
                positive = qwen_positive
            else:
                positive = encode_prompt(正面提示词, CLIP_类型_规范, Flux引导)

            if use_inherit and external_negative is not None:
                negative = external_negative
                print("[INFO] using external negative conditioning")
            elif is_qwen21:
                negative = qwen_negative
            elif 负面提示词 and 负面提示词.strip():
                negative = encode_prompt(负面提示词, CLIP_类型_规范, Flux引导)
            else:
                # 空负面也要走 return_dict，保证 attention_mask/num_tokens 与正面一致
                negative = encode_prompt(" ", CLIP_类型_规范, Flux引导)
            
            # ========== Latent ========== 
            latent = None
            latent_origin = "unknown"
            if use_inherit and external_latent is not None:
                latent = external_latent
                latent_origin = "external"
                print("[INFO] using external latent")
            elif Latent输入源 == "外部Latent优先" and external_latent is not None:
                latent = external_latent
                latent_origin = "external"
                print("[INFO] using external latent (preferred)")
            elif is_qwen21 and qwen_runtime_mode == "VAE传统重绘" and upstream_images:
                # TextEncodeQwenImage21 already VAE-encodes the references. Reuse the
                # first official reference latent as the noisy starting image instead
                # of encoding it a second time. This is real img2img/redraw behavior.
                refs = positive[0][1].get("reference_latents", []) if positive else []
                if not refs:
                    w, h = self.get_image_size(upstream_images[0])
                    fallback_latent = self.encode_image_to_latent(vae, upstream_images[0], w, h)
                    if fallback_latent is None:
                        raise RuntimeError(
                            "Qwen VAE redraw could not encode the upstream image. Check the matching Qwen VAE. "
                            "/ Qwen VAE传统重绘无法编码上游图像，请检查是否使用匹配的Qwen VAE。"
                        )
                    redraw_samples = fallback_latent["samples"]
                else:
                    redraw_samples = refs[0]
                if redraw_samples.shape[0] == 1 and 生成数量 > 1:
                    redraw_samples = redraw_samples.repeat(生成数量, *([1] * (redraw_samples.dim() - 1)))
                latent = {"samples": redraw_samples}
                latent_origin = "qwen21_vae_redraw"
                高度, 宽度 = redraw_samples.shape[-2] * 16, redraw_samples.shape[-1] * 16
                print(f"[INFO] Qwen-Image-2.1 VAE redraw: {宽度}x{高度}; denoise is adjustable")
            elif is_qwen21:
                refs = positive[0][1].get("reference_latents", []) if positive else []
                if refs:
                    # Native reference editing must use the empty latent returned by
                    # the official encoder so its canvas matches the resized reference.
                    latent = qwen_empty_latent
                    if latent is None:
                        latent = self.empty_latent_for_model(model, 宽度, 高度, 生成数量)
                    latent_origin = "qwen21_reference_empty"
                    print("[INFO] Qwen-Image-2.1 native reference editing uses the reference canvas")
                else:
                    # Text-to-image has no reference canvas. The encoder's `resolution`
                    # fallback is always square, so it must not override the sampler's
                    # user-selected width and height.
                    latent = self.empty_latent_for_model(model, 宽度, 高度, 生成数量)
                    latent_origin = "empty"
                    print(f"[INFO] Qwen-Image-2.1 text-to-image uses custom canvas: {宽度}x{高度}")
                if latent["samples"].shape[0] == 1 and 生成数量 > 1:
                    latent = latent.copy()
                    samples = latent["samples"]
                    latent["samples"] = samples.repeat(生成数量, *([1] * (samples.dim() - 1)))
            elif Latent输入源 == "空Latent":
                latent = self.empty_latent_for_model(model, 宽度, 高度, 生成数量)
                latent_origin = "empty"
                print(f"[INFO] using empty latent: {宽度}x{高度}")
            elif Latent输入源 == "上游图像拼接" and len(upstream_images) >= 1:
                # 将多张上游图像横向拼接为一张画布后编码为 Latent
                w0, h0 = self.get_image_size(upstream_images[0])
                target_h = h0 if h0 and h0 > 0 else 高度
                stitched = self.stitch_images_horizontal(upstream_images, target_h)
                if stitched is not None:
                    sh, sw = stitched.shape[1], stitched.shape[2]
                    print(f"[INFO] 拼接上游图像: {len(upstream_images)} 张 → 画布 {sw}x{sh}")
                    latent = self.encode_image_to_latent(vae, stitched, sw, sh)
                    if latent is not None:
                        latent_origin = "vae_image"
                if latent is None:
                    w, h = self.get_image_size(upstream_images[0])
                    latent = self.encode_image_to_latent(vae, upstream_images[0], w, h)
                    if latent is not None:
                        latent_origin = "vae_image"
                if latent is None:
                    latent = self.empty_latent_for_model(model, 宽度, 高度, 生成数量)
                    latent_origin = "empty"
                    print(f"[INFO] 拼接编码失败，使用空Latent")
            elif len(upstream_images) >= 1:
                # 上游图像优先 / 自动选择
                w, h = self.get_image_size(upstream_images[0])
                print(f"[INFO] 编码上游图像: {w}x{h}")
                latent = self.encode_image_to_latent(vae, upstream_images[0], w, h)
                if latent is not None:
                    latent_origin = "vae_image"
                if latent is None:
                    latent = self.empty_latent_for_model(model, 宽度, 高度, 生成数量)
                    latent_origin = "empty"
                    print(f"[INFO] 编码失败，使用空Latent")
            else:
                latent = self.empty_latent_for_model(model, 宽度, 高度, 生成数量)
                latent_origin = "empty"
                print(f"[INFO] using empty latent: {宽度}x{高度}")
            
            if latent is None:
                raise RuntimeError("Latent creation failed / Latent 生成失败")
            
            # ========== 自定义采样（支持外部 SIGMAS + 真正的 x0 降噪 Latent） ==========
            latent_image = latent["samples"]
            latent_for_sample = latent.copy()
            latent_image = comfy.sample.fix_empty_latent_channels(
                model, latent_image,
                latent.get("downscale_ratio_spacial", None),
                latent.get("downscale_ratio_temporal", None),
            )
            latent_for_sample["samples"] = latent_image

            if external_sigmas is not None:
                sigmas = external_sigmas
                print(f"[INFO] 使用外部Sigmas，采样步数: {max(int(sigmas.shape[-1]) - 1, 0)}")
            elif advanced_sampling:
                sigmas = comfy.samplers.KSampler(
                    model, 步数, "cpu", sampler=采样算法, scheduler=调度器, denoise=1.0
                ).sigmas
            else:
                effective_denoise = 1.0 if advanced_sampling else 重绘强度
                if latent_origin in ("empty", "qwen21_reference_empty") and effective_denoise < 1.0:
                    print(
                        f"[WARN] denoise={effective_denoise} cannot be used with an empty initial latent; "
                        "using 1.0 to prevent corrupted output. Connect external_latent for second-pass sampling. "
                        f"/ 当前为{('Qwen-Image-2.1原生参考编辑' if latent_origin == 'qwen21_reference_empty' else '空Latent文生图')}，"
                        "重绘强度已自动改为1.0；二次采样请连接外部Latent。"
                    )
                    effective_denoise = 1.0
                elif latent_origin in ("external", "vae_image", "qwen21_vae_redraw"):
                    print(f"[INFO] latent redraw mode ({latent_origin}); denoise={effective_denoise} is enabled")

                if effective_denoise <= 0.0:
                    sigmas = torch.FloatTensor([])
                else:
                    total_steps = 步数 if effective_denoise >= 1.0 else int(步数 / effective_denoise)
                    sigmas = comfy.samplers.calculate_sigmas(
                        model.get_model_object("model_sampling"), 调度器, total_steps
                    ).cpu()
                    sigmas = sigmas[-(步数 + 1):]
                print(f"[INFO] 使用内部Sigmas: {调度器}, {步数}步, 请求重绘强度={重绘强度}, 实际重绘强度={effective_denoise}")

            if advanced_sampling:
                schedule_steps = max(len(sigmas) - 1, 0)
                stop = min(end_at_step, schedule_steps)
                # Match KSamplerAdvanced: truncate the end, then slice the start.
                sigmas = sigmas[:stop + 1].clone()
                if stop < schedule_steps and not return_with_leftover_noise and len(sigmas):
                    sigmas[-1] = 0
                sigmas = sigmas[start_at_step:]
                print(f"[INFO] Advanced sampling / 高级分段：总步数={schedule_steps}, 起始={start_at_step}, 结束={stop}, 实际步数={max(len(sigmas)-1, 0)}, 添加噪声={add_noise}, 保留剩余噪声={return_with_leftover_noise}")

            batch_inds = latent_for_sample.get("batch_index", None)
            if (advanced_sampling and not add_noise) or noise_strength == 0.0:
                noise = torch.zeros_like(latent_image)
            else:
                noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
                if noise_strength != 1.0:
                    noise = noise * noise_strength
            noise_mask = latent_for_sample.get("noise_mask", None)
            sampler = comfy.samplers.sampler_object(采样算法)
            x0_output = {}
            callback = latent_preview.prepare_callback(
                model, max(int(sigmas.shape[-1]) - 1, 0), x0_output
            )
            if len(sigmas) <= 1:
                sampled_tensor = latent_image
            else:
                sampled_tensor = comfy.sample.sample_custom(
                    model, noise, CFG, sampler, sigmas, positive, negative, latent_image,
                    noise_mask=noise_mask, callback=callback,
                    disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed,
                )

            sampled_latent = latent_for_sample.copy()
            sampled_latent.pop("downscale_ratio_spacial", None)
            sampled_latent.pop("downscale_ratio_temporal", None)
            sampled_latent["samples"] = sampled_tensor

            if "x0" in x0_output:
                x0 = x0_output["x0"]
                if getattr(sampled_tensor, "is_nested", False) and not getattr(x0, "is_nested", False):
                    latent_shapes = [x.shape for x in sampled_tensor.unbind()]
                    x0 = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0, latent_shapes))
                denoised_latent = latent_for_sample.copy()
                denoised_latent["samples"] = model.model.process_latent_out(x0.cpu())
            else:
                denoised_latent = sampled_latent

            samples = (sampled_latent,)
            
            preview_latent = denoised_latent if advanced_sampling and return_with_leftover_noise else sampled_latent
            images = vae.decode(preview_latent["samples"])
            # 视频 VAE（如 Wan/Qwen/Krea2 用的 Wan21 VAE，latent_dim=3）解码后是 5D [B,T,H,W,C]，
            # 需要把时间维合并进 batch，得到标准 4D IMAGE [B,H,W,C]，否则 SaveImage 会报
            # "Cannot handle this data type"。与 ComfyUI 官方 VAEDecode 处理一致。
            if len(images.shape) == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            elif len(images.shape) == 3:
                images = images.unsqueeze(0)
            print(f"[INFO] done, images: {images.shape[0]}")
            
            if 自动清理显存:
                self.cleanup_vram()
            
            return (images, model, clip, vae, samples[0], positive, negative, denoised_latent)
            
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
            if 自动清理显存:
                self.cleanup_vram()
            # 不返回伪造图像和 None 输出；让错误停在本节点，避免下游节点
            # 报出 "NoneType has no attribute copy" 等误导性异常。
            raise RuntimeError(f"Binyuan Sampler V2.2 EN failed / 执行失败: {e}") from e

NODE_CLASS_MAPPINGS = {"BinyuanUltimateSamplerEN": BinyuanUltimateSamplerEN}
NODE_DISPLAY_NAME_MAPPINGS = {"BinyuanUltimateSamplerEN": "🛡️ Binyuan Sampler V2.2 EN｜中英双语版"}
WEB_DIRECTORY = "js"
