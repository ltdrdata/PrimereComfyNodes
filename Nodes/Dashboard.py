from custom_nodes.ComfyUI_Primere_Nodes.components.tree import TREE_DASHBOARD
from custom_nodes.ComfyUI_Primere_Nodes.components.tree import PRIMERE_ROOT
import comfy.samplers
import folder_paths
import nodes
import torch
import torch.nn.functional as F
from .modules.latent_noise import PowerLawNoise
import random
import os
import tomli
import math
from .modules.adv_encode import advanced_encode, advanced_encode_XL
from nodes import MAX_RESOLUTION

class PrimereSamplers:
    CATEGORY = TREE_DASHBOARD
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS)
    RETURN_NAMES = ("sampler_name", "scheduler_name")
    FUNCTION = "get_sampler"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler_name": (comfy.samplers.KSampler.SCHEDULERS,)
            }
        }

    def get_sampler(self, sampler_name, scheduler_name):
        return sampler_name, scheduler_name


class PrimereVAE:
    RETURN_TYPES = ("VAE_NAME",)
    RETURN_NAMES = ("vae_name",)
    FUNCTION = "load_vae_list"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_model": (folder_paths.get_filename_list("vae"),)
            },
        }

    def load_vae_list(self, vae_model):
        return vae_model,

class PrimereCKPT:
    RETURN_TYPES = ("CHECKPOINT_NAME", "INT", "STRING")
    RETURN_NAMES = ("ckpt_name", "is_sdxl", "sdxl_path")
    FUNCTION = "load_ckpt_list"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_model": (folder_paths.get_filename_list("checkpoints"),)
            },
            "optional": {
                "sdxl_path": ("STRING", {"default": 'SDXL'}),
            },
        }

    def load_ckpt_list(self, base_model, sdxl_path):
        is_sdxl = 0
        sdxl_path_string = sdxl_path
        if sdxl_path:
            if not sdxl_path.endswith(os.sep):
                sdxl_path = sdxl_path + os.sep
            if (base_model.startswith(sdxl_path) == True):
                is_sdxl = 1

        return (base_model, is_sdxl, sdxl_path_string,)

class PrimereVAELoader:
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("VAE",)
    FUNCTION = "load_primere_vae"
    CATEGORY = TREE_DASHBOARD

    def __init__(self):
        self.vae_loader = nodes.VAELoader()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": ("VAE_NAME",)
            },
        }

    def load_primere_vae(self, vae_name, ):
        return self.vae_loader.load_vae(vae_name)

class PrimereCKPTLoader:
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "INT",)
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "IS_SDXL",)
    FUNCTION = "load_primere_ckpt"
    CATEGORY = TREE_DASHBOARD

    def __init__(self):
        self.chkp_loader = nodes.CheckpointLoaderSimple()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": ("CHECKPOINT_NAME",),
            },
            "optional": {
                "sdxl_path": ("STRING", {"default": 'SDXL'}),
            },
        }

    def load_primere_ckpt(self, ckpt_name, sdxl_path):
        is_sdxl = 0
        if sdxl_path:
            if not sdxl_path.endswith(os.sep):
                sdxl_path = sdxl_path + os.sep
            if (ckpt_name.startswith(sdxl_path) == True):
                is_sdxl = 1

        return self.chkp_loader.load_checkpoint(ckpt_name) + (is_sdxl,)

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

class PrimerePromptSwitch:
    any_typ = AnyType("*")

    RETURN_TYPES = (any_typ, any_typ, "INT")
    RETURN_NAMES = ("selected_pos", "selected_neg", "selected_index")
    FUNCTION = "promptswitch"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(cls):
        any_typ = AnyType("*")

        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
            },
            "optional": {
                "prompt_pos_1": (any_typ,),
                "prompt_neg_1": (any_typ,),
            },
        }

    def promptswitch(self, *args, **kwargs):
        selected_index = int(kwargs['select'])
        input_namep = f"prompt_pos_{selected_index}"
        input_namen = f"prompt_neg_{selected_index}"
        # selected_label = input_namep

        if input_namep in kwargs:
            return (kwargs[input_namep], kwargs[input_namen], selected_index)
        else:
            print(f"ImpactSwitch: invalid select index (ignored)")
            return (None, None, selected_index)

class PrimereSeed:
  RETURN_TYPES = ("INT",)
  RETURN_NAMES = ("SEED",)
  FUNCTION = "seed"
  CATEGORY = TREE_DASHBOARD

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "seed": ("INT", {
          "default": -1,
          "min": -18446744073709551615, # -1125899906842624,
          "max": 18446744073709551615, # 1125899906842624
        }),
      },
    }

  def seed(self, seed = 0):
    return (seed,)


class PrimereFractalLatent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        pln = PowerLawNoise('cpu')
        return {
            "required": {
                # "batch_size": ("INT", {"default": 1, "max": 64, "min": 1, "step": 1}),
                "width": ("INT", {"default": 512, "max": 8192, "min": 64, "forceInput": True}),
                "height": ("INT", {"default": 512, "max": 8192, "min": 64, "forceInput": True}),
                # "resampling": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],),
                "rand_noise_type": ("BOOLEAN", {"default": False}),
                "noise_type": (pln.get_noise_types(),),
                # "scale": ("FLOAT", {"default": 1.0, "max": 1024.0, "min": 0.01, "step": 0.001}),
                "rand_alpha_exponent": ("BOOLEAN", {"default": True}),
                "alpha_exponent": ("FLOAT", {"default": 1.0, "max": 12.0, "min": -12.0, "step": 0.001}),
                "alpha_exp_rand_min": ("FLOAT", {"default": 0.5, "max": 12.0, "min": -12.0, "step": 0.001}),
                "alpha_exp_rand_max": ("FLOAT", {"default": 1.5, "max": 12.0, "min": -12.0, "step": 0.001}),
                "rand_modulator": ("BOOLEAN", {"default": True}),
                "modulator": ("FLOAT", {"default": 1.0, "max": 2.0, "min": 0.1, "step": 0.01}),
                "modulator_rand_min": ("FLOAT", {"default": 0.8, "max": 2.0, "min": 0.1, "step": 0.01}),
                "modulator_rand_max": ("FLOAT", {"default": 1.4, "max": 2.0, "min": 0.1, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615, "forceInput": True}),
                "rand_device": ("BOOLEAN", {"default": False}),
                "device": (["cpu", "cuda"],),
            },
            "optional": {
                "optional_vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latents", "previews")
    FUNCTION = "primere_latent_noise"
    CATEGORY = TREE_DASHBOARD

    def primere_latent_noise(self, width, height, rand_noise_type, noise_type, rand_alpha_exponent, alpha_exponent, alpha_exp_rand_min, alpha_exp_rand_max, rand_modulator, modulator, modulator_rand_min, modulator_rand_max, seed, rand_device, device, optional_vae = None):
        if rand_device == True:
            device = random.choice(["cpu", "cuda"])

        power_law = PowerLawNoise(device = device)

        if rand_alpha_exponent == True:
            alpha_exponent = round(random.uniform(alpha_exp_rand_min, alpha_exp_rand_max), 3)

        if rand_modulator == True:
            modulator = round(random.uniform(modulator_rand_min, modulator_rand_max), 2)

        if rand_noise_type == True:
            pln = PowerLawNoise(device)
            noise_type = random.choice(pln.get_noise_types())

        tensors = power_law(1, width, height, scale = 1, alpha = alpha_exponent, modulator = modulator, noise_type = noise_type, seed = seed)
        alpha_channel = torch.ones((1, height, width, 1), dtype = tensors.dtype, device = "cpu")
        tensors = torch.cat((tensors, alpha_channel), dim = 3)

        if optional_vae is None:
            latents = tensors.permute(0, 3, 1, 2)
            latents = F.interpolate(latents, size=((height // 8), (width // 8)), mode = 'nearest-exact')
            return {'samples': latents}, tensors

        encoder = nodes.VAEEncode()
        latents = []
        for tensor in tensors:
            tensor = tensor.unsqueeze(0)
            latents.append(encoder.encode(optional_vae, tensor)[0]['samples'])

        latents = torch.cat(latents)
        return {'samples': latents}, tensors

class PrimereCLIP:
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("COND+", "COND-", "PROMPT+", "PROMPT-")
    FUNCTION = "clip_encode"
    CATEGORY = TREE_DASHBOARD

    @staticmethod
    def get_default_neg(toml_path: str):
        with open(toml_path, "rb") as f:
            style_def_neg = tomli.load(f)
        return style_def_neg
    @ classmethod
    def INPUT_TYPES(cls):
        DEF_TOML_DIR = os.path.join(PRIMERE_ROOT, 'Toml')
        cls.default_neg = cls.get_default_neg(os.path.join(DEF_TOML_DIR, "default_neg.toml"))

        return {
            "required": {
                "clip": ("CLIP", ),
                "is_sdxl": ("INT", {"default": 0, "forceInput": True}),
                "positive_prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
                "negative_strength": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.01}),
                "use_additional": ("BOOLEAN", {"default": False}),
                "additional": (list(cls.default_neg.keys()),),
                "additional_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "adv_encode": ("BOOLEAN", {"default": False}),
                "token_normalization": (["none", "mean", "length", "length+mean"],),
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),
                # "affect_pooled": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "opt_pos_prompt": ("STRING", {"forceInput": True}),
                "opt_pos_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "opt_neg_prompt": ("STRING", {"forceInput": True}),
                "opt_neg_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),

                "style_pos_prompt": ("STRING", {"forceInput": True}),
                "style_pos_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "style_neg_prompt": ("STRING", {"forceInput": True}),
                "style_neg_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),

                "sdxl_positive_l": ("STRING", {"forceInput": True}),
                "sdxl_negative_l": ("STRING", {"forceInput": True}),
                "sdxl_l_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION, "forceInput": True}),
                "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION, "forceInput": True}),
            }
        }

    def clip_encode(self, clip, negative_strength, additional_strength, opt_pos_strength, opt_neg_strength, style_pos_strength, style_neg_strength, additional, adv_encode, token_normalization, weight_interpretation, sdxl_l_strength, width = 1024, height = 1024, positive_prompt = "", negative_prompt = "", opt_pos_prompt = "", opt_neg_prompt = "", style_neg_prompt = "", style_pos_prompt = "", sdxl_positive_l = "", sdxl_negative_l = "", use_additional = False, is_sdxl = 0):
        additional_positive = None
        additional_negative = None
        if use_additional == True:
            additional_positive = self.default_neg[additional]['positive'].strip(' ,;')
            additional_negative = self.default_neg[additional]['negative'].strip(' ,;')

        additional_positive = f'({additional_positive}:{additional_strength:.2f})' if additional_positive is not None and additional_positive != '' else ''
        additional_negative = f'({additional_negative}:{additional_strength:.2f})' if additional_negative is not None and additional_negative != '' else ''

        negative_prompt = f'({negative_prompt}:{negative_strength:.2f})' if negative_prompt is not None and negative_prompt.strip(' ,;') != '' else ''
        opt_pos_prompt = f'({opt_pos_prompt}:{opt_pos_strength:.2f})' if opt_pos_prompt is not None and opt_pos_prompt.strip(' ,;') != '' else ''
        opt_neg_prompt = f'({opt_neg_prompt}:{opt_neg_strength:.2f})' if opt_neg_prompt is not None and opt_neg_prompt.strip(' ,;') != '' else ''
        style_pos_prompt = f'({style_pos_prompt}:{style_pos_strength:.2f})' if style_pos_prompt is not None and style_pos_prompt.strip(' ,;') != '' else ''
        style_neg_prompt = f'({style_neg_prompt}:{style_neg_strength:.2f})' if style_neg_prompt is not None and style_neg_prompt.strip(' ,;') != '' else ''
        sdxl_positive_l = f'({sdxl_positive_l}:{sdxl_l_strength:.2f})' if sdxl_positive_l is not None and sdxl_positive_l.strip(' ,;') != '' else ''
        sdxl_negative_l = f'({sdxl_negative_l}:{sdxl_l_strength:.2f})' if sdxl_negative_l is not None and sdxl_negative_l.strip(' ,;') != '' else ''

        positive_text = f'{positive_prompt}, {opt_pos_prompt}, {style_pos_prompt}, {additional_positive}'.strip(' ,;').replace(", , ", ", ").replace(", , ", ", ")
        negative_text = f'{negative_prompt}, {opt_neg_prompt}, {style_neg_prompt}, {additional_negative}'.strip(' ,;').replace(", , ", ", ").replace(", , ", ", ")

        if (adv_encode == True):
            if (is_sdxl == 0):
                embeddings_final_pos, pooled_pos = advanced_encode(clip, positive_text, token_normalization, weight_interpretation, w_max = 1.0, apply_to_pooled = True)
                embeddings_final_neg, pooled_neg = advanced_encode(clip, negative_text, token_normalization, weight_interpretation, w_max = 1.0, apply_to_pooled = True)
                return ([[embeddings_final_pos, {"pooled_output": pooled_pos}]], [[embeddings_final_neg, {"pooled_output": pooled_neg}]], positive_text, negative_text)
            else:
                # embeddings_final_pos, pooled_pos = advanced_encode_XL(clip, sdxl_positive_l, positive_text, token_normalization, weight_interpretation, w_max = 1.0, clip_balance = sdxl_balance_l, apply_to_pooled = True)
                # embeddings_final_neg, pooled_neg = advanced_encode_XL(clip, sdxl_negative_l, negative_text, token_normalization, weight_interpretation, w_max = 1.0, clip_balance = sdxl_balance_l, apply_to_pooled = True)
                # return ([[embeddings_final_pos, {"pooled_output": pooled_pos}]],[[embeddings_final_neg, {"pooled_output": pooled_neg}]],)

                tokens_p = clip.tokenize(positive_text)
                tokens_p["l"] = clip.tokenize(sdxl_positive_l)["l"]

                if len(tokens_p["l"]) != len(tokens_p["g"]):
                    empty = clip.tokenize("")
                    while len(tokens_p["l"]) < len(tokens_p["g"]):
                        tokens_p["l"] += empty["l"]
                    while len(tokens_p["l"]) > len(tokens_p["g"]):
                        tokens_p["g"] += empty["g"]

                tokens_n = clip.tokenize(negative_text)
                tokens_n["l"] = clip.tokenize(sdxl_negative_l)["l"]

                if len(tokens_n["l"]) != len(tokens_n["g"]):
                    empty = clip.tokenize("")
                    while len(tokens_n["l"]) < len(tokens_n["g"]):
                        tokens_n["l"] += empty["l"]
                    while len(tokens_n["l"]) > len(tokens_n["g"]):
                        tokens_n["g"] += empty["g"]

                cond_p, pooled_p = clip.encode_from_tokens(tokens_p, return_pooled = True)
                cond_n, pooled_n = clip.encode_from_tokens(tokens_n, return_pooled = True)
                return ([[cond_p, {"pooled_output": pooled_p, "width": width, "height": height, "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]], [[cond_n, {"pooled_output": pooled_n, "width": width, "height": height, "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]], positive_text, negative_text)

        else:
            tokens = clip.tokenize(positive_text)
            cond_pos, pooled_pos = clip.encode_from_tokens(tokens, return_pooled = True)

            tokens = clip.tokenize(negative_text)
            cond_neg, pooled_neg = clip.encode_from_tokens(tokens, return_pooled = True)
            return ([[cond_pos, {"pooled_output": pooled_pos}]], [[cond_neg, {"pooled_output": pooled_neg}]], positive_text, negative_text)

class PrimereResolution:
    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("WIDTH", "HEIGHT",)
    FUNCTION = "calculate_dimensions"
    CATEGORY = TREE_DASHBOARD

    @staticmethod
    def get_ratios(toml_path: str):
        with open(toml_path, "rb") as f:
            image_ratios = tomli.load(f)
        return image_ratios

    @ classmethod
    def INPUT_TYPES(cls):
        DEF_TOML_DIR = os.path.join(PRIMERE_ROOT, 'Toml')
        cls.sd_ratios = cls.get_ratios(os.path.join(DEF_TOML_DIR, "resolution_ratios.toml"))

        namelist = {}
        for sd_ratio_key in cls.sd_ratios:
            rationName = cls.sd_ratios[sd_ratio_key]['name']
            namelist[rationName] = sd_ratio_key

        cls.ratioNames = namelist

        return {
            "required": {
                "ratio": (list(namelist.keys()),),
                "orientation": (["Horizontal", "Vertical"], {"default": "Horizontal"}),
                "default_sd": (["SD 1.x", "SD 2.x"], {"default": "SD 1.x"}),
                "is_sdxl": ("INT", {"default": 0, "forceInput": True}),
                "calculate_by_custom": ("BOOLEAN", {"default": False}),
                "custom_side_a": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 100.0, "step": 0.1}),
                "custom_side_b": ("FLOAT", {"default": 2.8, "min": 1.0, "max": 100.0, "step": 0.1}),
            },
        }

    def calculate_dimensions(self, ratio: str, orientation: str, is_sdxl: int, default_sd: str, calculate_by_custom: bool, custom_side_a: float, custom_side_b: float):
        SD_1 = 512
        SD_2 = 768
        SDXL_1 = 1024
        DEFAULT_RES = SD_1

        if (default_sd == 'SD 2.x'):
            DEFAULT_RES = SD_2

        if (is_sdxl == 1):
            DEFAULT_RES = SDXL_1
        def calculate(ratio_1: float, ratio_2: float, side: int):
            FullPixels = side ** 2
            result_x = FullPixels / ratio_2
            result_y = result_x / ratio_1
            side_base = round(math.sqrt(result_y))
            side_a = round(ratio_1 * side_base)
            side_b = round(FullPixels / side_a)
            return sorted([side_a, side_b], reverse=True)

        if (calculate_by_custom == True and isinstance(custom_side_a, (int, float)) and isinstance(custom_side_b, (int, float)) and custom_side_a >= 1 and custom_side_b >= 1):
            ratio_x = custom_side_a
            ratio_y = custom_side_b
        else:
            RatioLabel = self.ratioNames[ratio]
            ratio_x = self.sd_ratios[RatioLabel]['side_x']
            ratio_y = self.sd_ratios[RatioLabel]['side_y']

        dimensions = calculate(ratio_x, ratio_y, DEFAULT_RES)
        if (orientation == 'Vertical'):
            dimensions = sorted(dimensions)

        dimension_x = dimensions[0]
        dimension_y = dimensions[1]
        return (dimension_x, dimension_y,)

class PrimereStepsCfg:
  RETURN_TYPES = ("INT", "FLOAT")
  RETURN_NAMES = ("Steps", "CFG")
  FUNCTION = "steps_cfg"
  CATEGORY = TREE_DASHBOARD

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "steps": ("INT", {"default": 12, "min": 1, "max": 1000, "step": 1}),
        "cfg": ("FLOAT", {"default": 7, "min": 0.1, "max": 100, "step": 0.01}),
      },
    }

  def steps_cfg(self, steps = 12, cfg = 7):
    return (steps, cfg,)