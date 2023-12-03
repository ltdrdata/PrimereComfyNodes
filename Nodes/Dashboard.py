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
from .modules.adv_encode import advanced_encode, advanced_encode_XL
from nodes import MAX_RESOLUTION
from custom_nodes.ComfyUI_Primere_Nodes.components import utility
from pathlib import Path
import re
import requests

class PrimereSamplers:
    CATEGORY = TREE_DASHBOARD
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS)
    RETURN_NAMES = ("SAMPLER_NAME", "SCHEDULER_NAME")
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
    RETURN_NAMES = ("VAE_NAME",)
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
    RETURN_TYPES = ("CHECKPOINT_NAME", "STRING")
    RETURN_NAMES = ("MODEL_NAME", "MODEL_VERSION")
    FUNCTION = "load_ckpt_list"
    CATEGORY = TREE_DASHBOARD

    def __init__(self):
        self.chkp_loader = nodes.CheckpointLoaderSimple()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_model": (folder_paths.get_filename_list("checkpoints"),)
            },
        }

    def load_ckpt_list(self, base_model,):
        LOADED_CHECKPOINT = self.chkp_loader.load_checkpoint(base_model)
        model_version = utility.getCheckpointVersion(LOADED_CHECKPOINT[0])

        return (base_model, model_version,)

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

class PrimereLCMSelector:
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "FLOAT", "INT")
    RETURN_NAMES = ("SAMPLER_NAME", "SCHEDULER_NAME", "STEPS", "CFG", "IS_LCM")
    FUNCTION = "select_lcm_mode"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "use_lcm": ("BOOLEAN", {"default": False}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True, "default": "euler"}),
                "scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True, "default": "normal"}),
                "lcm_sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True, "default": "lcm"}),
                "lcm_scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True, "default": "sgm_uniform"}),
                "cfg_scale": ('FLOAT', {"forceInput": True, "default": 7}),
                "steps": ('INT', {"forceInput": True, "default": 12}),
                "lcm_cfg_scale": ('FLOAT', {"forceInput": True, "default": 1.2}),
                "lcm_steps": ('INT', {"forceInput": True, "default": 6}),
            },
        }

    def select_lcm_mode(self, use_lcm = False, sampler_name = 'euler', scheduler_name = 'normal', lcm_sampler_name = 'lcm', lcm_scheduler_name = 'sgm_uniform', cfg_scale = 7, steps = 12, lcm_cfg_scale = 1.2, lcm_steps = 6):
        lcm_mode = 0
        if use_lcm == True:
            sampler_name = lcm_sampler_name
            scheduler_name = lcm_scheduler_name
            steps = lcm_steps
            cfg_scale = lcm_cfg_scale
            lcm_mode = 1

        return (sampler_name, scheduler_name, steps, cfg_scale, lcm_mode,)


class PrimereCKPTLoader:
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING",)
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "MODEL_VERSION")
    FUNCTION = "load_primere_ckpt"
    CATEGORY = TREE_DASHBOARD

    def __init__(self):
        self.chkp_loader = nodes.CheckpointLoaderSimple()
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": ("CHECKPOINT_NAME",),
                "is_lcm": ("INT", {"default": 0, "forceInput": True}),
                "strength_lcm_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_lcm_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
        }

    def load_primere_ckpt(self, ckpt_name, is_lcm, strength_lcm_model, strength_lcm_clip):
        LOADED_CHECKPOINT = self.chkp_loader.load_checkpoint(ckpt_name)
        OUTPUT_MODEL = LOADED_CHECKPOINT[0]
        OUTPUT_CLIP = LOADED_CHECKPOINT[1]
        MODEL_VERSION = utility.getCheckpointVersion(OUTPUT_MODEL)

        def lcm(self, model, zsnr=False):
            m = model.clone()

            sampling_base = comfy.model_sampling.ModelSamplingDiscrete
            sampling_type = utility.LCM
            sampling_base = utility.ModelSamplingDiscreteLCM

            class ModelSamplingAdvanced(sampling_base, sampling_type):
                pass

            model_sampling = ModelSamplingAdvanced()
            if zsnr:
                model_sampling.set_sigmas(utility.rescale_zero_terminal_snr_sigmas(model_sampling.sigmas))

            m.add_object_patch("model_sampling", model_sampling)
            return m

        is_sdxl = 0
        match MODEL_VERSION:
            case 'SDXL_2048':
                is_sdxl = 1

        if is_lcm == 1:
            SDXL_LORA = 'https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors?download=true'
            SD_LORA = 'https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors?download=true'
            DOWNLOADED_SD_LORA = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'lcm_lora_sd.safetensors')
            DOWNLOADED_SDXL_LORA = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'lcm_lora_sdxl.safetensors')

            if os.path.exists(DOWNLOADED_SD_LORA) == False:
                print('Downloading SD LCM LORA....')
                reqsdlcm = requests.get(SD_LORA, allow_redirects=True)
                if reqsdlcm.status_code == 200 and reqsdlcm.ok == True:
                    open(DOWNLOADED_SD_LORA, 'wb').write(reqsdlcm.content)
                else:
                    print('ERROR: Cannot dowload SD LCM Lora')

            if os.path.exists(DOWNLOADED_SDXL_LORA) == False:
                print('Downloading SDXL LCM LORA....')
                reqsdxllcm = requests.get(SDXL_LORA, allow_redirects=True)
                if reqsdxllcm.status_code == 200 and reqsdxllcm.ok == True:
                    open(DOWNLOADED_SDXL_LORA, 'wb').write(reqsdxllcm.content)
                else:
                    print('ERROR: Cannot dowload SDXL LCM Lora')

            if is_sdxl == 0:
                LORA_PATH = DOWNLOADED_SD_LORA
            else:
                LORA_PATH = DOWNLOADED_SDXL_LORA

            if os.path.exists(LORA_PATH) == True:
                if strength_lcm_model > 0 or strength_lcm_clip > 0:
                    print('LCM mode on')
                    lora = None

                    if self.loaded_lora is not None:
                        if self.loaded_lora[0] == LORA_PATH:
                            lora = self.loaded_lora[1]
                        else:
                            temp = self.loaded_lora
                            self.loaded_lora = None
                            del temp

                    if lora is None:
                        lora = comfy.utils.load_torch_file(LORA_PATH, safe_load=True)
                        self.loaded_lora = (LORA_PATH, lora)

                    MODEL_LORA, CLIP_LORA = comfy.sd.load_lora_for_models(OUTPUT_MODEL, OUTPUT_CLIP, lora, strength_lcm_model, strength_lcm_clip)

                    OUTPUT_MODEL = lcm(self, MODEL_LORA, False)
                    OUTPUT_CLIP = CLIP_LORA

        return (OUTPUT_MODEL,) + (OUTPUT_CLIP,) + (LOADED_CHECKPOINT[2],) + (MODEL_VERSION,)

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

class PrimerePromptSwitch:
    any_typ = AnyType("*")

    RETURN_TYPES = (any_typ, any_typ, any_typ, any_typ, any_typ, "INT")
    RETURN_NAMES = ("PROMPT+", "PROMPT-", "SUBPATH", "MODEL", "ORIENTATION", "SELECTED_INDEX")
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
                "subpath_1": (any_typ,),
                "model_1": (any_typ,),
                "orientation_1": (any_typ,),
            },
        }

    def promptswitch(self, *args, **kwargs):
        selected_index = int(kwargs['select'])
        input_namep = f"prompt_pos_{selected_index}"
        input_namen = f"prompt_neg_{selected_index}"
        input_subpath = f"subpath_{selected_index}"
        input_model = f"model_{selected_index}"
        input_orientation = f"orientation_{selected_index}"

        if input_subpath not in kwargs:
            kwargs[input_subpath] = None

        if input_model not in kwargs:
            kwargs[input_model] = None

        if input_orientation not in kwargs:
            kwargs[input_orientation] = None

        if input_namep in kwargs:
            return (kwargs[input_namep], kwargs[input_namen], kwargs[input_subpath], kwargs[input_model], kwargs[input_orientation], selected_index)
        else:
            print(f"PrimerePromptSwitch: invalid select index (ignored)")
            return (None, None, None, None, None, selected_index)

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
    RETURN_NAMES = ("LATENTS", "PREVIEWS")
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
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("COND+", "COND-", "PROMPT+", "PROMPT-", "PROMPT L+", "PROMPT L-")
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
        cls.default_pos = cls.get_default_neg(os.path.join(DEF_TOML_DIR, "default_pos.toml"))

        return {
            "required": {
                "clip": ("CLIP", ),
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
                "positive_prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
                "negative_strength": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.01}),
                "use_int_style": ("BOOLEAN", {"default": False}),
                "int_style_pos": (['None'] + sorted(list(cls.default_pos.keys())),),
                "int_style_pos_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "int_style_neg": (['None'] + sorted(list(cls.default_neg.keys())),),
                "int_style_neg_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
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
                "copy_prompt_to_l": ("BOOLEAN", {"default": True}),
                "sdxl_l_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION, "forceInput": True}),
                "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION, "forceInput": True}),
            }
        }

    def clip_encode(self, clip, negative_strength, int_style_pos_strength, int_style_neg_strength, opt_pos_strength, opt_neg_strength, style_pos_strength, style_neg_strength, int_style_pos, int_style_neg, adv_encode, token_normalization, weight_interpretation, sdxl_l_strength, copy_prompt_to_l = True, width = 1024, height = 1024, positive_prompt = "", negative_prompt = "", opt_pos_prompt = "", opt_neg_prompt = "", style_neg_prompt = "", style_pos_prompt = "", sdxl_positive_l = "", sdxl_negative_l = "", use_int_style = False, model_version = "BaseModel_1024"):
        is_sdxl = 0
        match model_version:
            case 'SDXL_2048':
                is_sdxl = 1

        additional_positive = int_style_pos
        additional_negative = int_style_neg
        if int_style_pos == 'None' or use_int_style == False:
            additional_positive = None
        if int_style_neg == 'None' or use_int_style == False:
            additional_negative = None

        if use_int_style == True:
            if int_style_pos != 'None':
                additional_positive = self.default_pos[int_style_pos]['positive'].strip(' ,;')
            if int_style_neg != 'None':
                additional_negative = self.default_neg[int_style_neg]['negative'].strip(' ,;')

        additional_positive = f'({additional_positive}:{int_style_pos_strength:.2f})' if additional_positive is not None and additional_positive != '' else ''
        additional_negative = f'({additional_negative}:{int_style_neg_strength:.2f})' if additional_negative is not None and additional_negative != '' else ''

        negative_prompt = f'({negative_prompt}:{negative_strength:.2f})' if negative_prompt is not None and negative_prompt.strip(' ,;') != '' else ''
        if copy_prompt_to_l == True:
            sdxl_positive_l = positive_prompt
            sdxl_negative_l = negative_prompt

        opt_pos_prompt = f'({opt_pos_prompt}:{opt_pos_strength:.2f})' if opt_pos_prompt is not None and opt_pos_prompt.strip(' ,;') != '' else ''
        opt_neg_prompt = f'({opt_neg_prompt}:{opt_neg_strength:.2f})' if opt_neg_prompt is not None and opt_neg_prompt.strip(' ,;') != '' else ''
        style_pos_prompt = f'({style_pos_prompt}:{style_pos_strength:.2f})' if style_pos_prompt is not None and style_pos_prompt.strip(' ,;') != '' else ''
        style_neg_prompt = f'({style_neg_prompt}:{style_neg_strength:.2f})' if style_neg_prompt is not None and style_neg_prompt.strip(' ,;') != '' else ''
        sdxl_positive_l = f'({sdxl_positive_l}:{sdxl_l_strength:.2f})'.replace(":1.00", "") if sdxl_positive_l is not None and sdxl_positive_l.strip(' ,;') != '' else ''
        sdxl_negative_l = f'({sdxl_negative_l}:{sdxl_l_strength:.2f})'.replace(":1.00", "") if sdxl_negative_l is not None and sdxl_negative_l.strip(' ,;') != '' else ''

        positive_text = f'{positive_prompt}, {opt_pos_prompt}, {style_pos_prompt}, {additional_positive}'.strip(' ,;').replace(", , ", ", ").replace(", , ", ", ").replace(":1.00", "")
        negative_text = f'{negative_prompt}, {opt_neg_prompt}, {style_neg_prompt}, {additional_negative}'.strip(' ,;').replace(", , ", ", ").replace(", , ", ", ").replace(":1.00", "")

        if (model_version == 'BaseModel_1024'):
            adv_encode = False

        if (adv_encode == True):
            if (is_sdxl == 0):
                embeddings_final_pos, pooled_pos = advanced_encode(clip, positive_text, token_normalization, weight_interpretation, w_max = 1.0, apply_to_pooled = True)
                embeddings_final_neg, pooled_neg = advanced_encode(clip, negative_text, token_normalization, weight_interpretation, w_max = 1.0, apply_to_pooled = True)
                return ([[embeddings_final_pos, {"pooled_output": pooled_pos}]], [[embeddings_final_neg, {"pooled_output": pooled_neg}]], positive_text, negative_text, "", "")
            else:
                # embeddings_final_pos, pooled_pos = advanced_encode_XL(clip, sdxl_positive_l, positive_text, token_normalization, weight_interpretation, w_max = 1.0, clip_balance = sdxl_balance_l, apply_to_pooled = True)
                # embeddings_final_neg, pooled_neg = advanced_encode_XL(clip, sdxl_negative_l, negative_text, token_normalization, weight_interpretation, w_max = 1.0, clip_balance = sdxl_balance_l, apply_to_pooled = True)
                # return ([[embeddings_final_pos, {"pooled_output": pooled_pos}]],[[embeddings_final_neg, {"pooled_output": pooled_neg}]], positive_text, negative_text, sdxl_positive_l, sdxl_negative_l)

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
                return ([[cond_p, {"pooled_output": pooled_p, "width": width, "height": height, "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]], [[cond_n, {"pooled_output": pooled_n, "width": width, "height": height, "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]], positive_text, negative_text, sdxl_positive_l, sdxl_negative_l)

        else:
            tokens = clip.tokenize(positive_text)
            cond_pos, pooled_pos = clip.encode_from_tokens(tokens, return_pooled = True)

            tokens = clip.tokenize(negative_text)
            cond_neg, pooled_neg = clip.encode_from_tokens(tokens, return_pooled = True)
            return ([[cond_pos, {"pooled_output": pooled_pos}]], [[cond_neg, {"pooled_output": pooled_neg}]], positive_text, negative_text, "", "")

class PrimereResolution:
    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("WIDTH", "HEIGHT",)
    FUNCTION = "calculate_imagesize"
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
                # "force_768_SD1x": ("BOOLEAN", {"default": True}),
                "basemodel_res": ([512, 768, 1024, 1280], {"default": 768}),
                "rnd_orientation": ("BOOLEAN", {"default": False}),
                "orientation": (["Horizontal", "Vertical"], {"default": "Horizontal"}),
                "round_to_standard": ("BOOLEAN", {"default": False}),
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
                "seed": ("INT", {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615, "forceInput": True}),
                "calculate_by_custom": ("BOOLEAN", {"default": False}),
                "custom_side_a": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 100.0, "step": 0.1}),
                "custom_side_b": ("FLOAT", {"default": 2.8, "min": 1.0, "max": 100.0, "step": 0.1}),
            },
        }

    def calculate_imagesize(self, ratio: str, basemodel_res: int, rnd_orientation: bool, orientation: str, round_to_standard: bool, model_version: str, seed: int, calculate_by_custom: bool, custom_side_a: float, custom_side_b: float):
        if rnd_orientation == True:
            if (seed % 2) == 0:
                orientation = "Horizontal"
            else:
                orientation = "Vertical"

        # if force_768_SD1x == True and  model_version == 'BaseModel_768':
        #    model_version = 'BaseModel_1024'
        if model_version == 'BaseModel_768':
            match basemodel_res:
                case 512:
                    model_version = 'BaseModel_768'
                case 768:
                    model_version = 'BaseModel_1024'
                case 1024:
                    model_version = 'BaseModel_mod_1024'
                case 1280:
                    model_version = 'BaseModel_mod_1280'

        dimensions = utility.calculate_dimensions(self, ratio, orientation, round_to_standard, model_version, calculate_by_custom, custom_side_a, custom_side_b)
        dimension_x = dimensions[0]
        dimension_y = dimensions[1]
        return (dimension_x, dimension_y,)

class PrimereResolutionMultiplier:
    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = ("WIDTH", "HEIGHT", "UPSCALE_RATIO")
    FUNCTION = "multiply_imagesize"
    CATEGORY = TREE_DASHBOARD

    @ classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ('INT', {"forceInput": True, "default": 512}),
                "height": ('INT', {"forceInput": True, "default": 512}),
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
                "multiply_sd": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                "multiply_sdxl": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.1}),
            },
        }

    def multiply_imagesize(self, width: int, height: int, multiply_sd: float, multiply_sdxl: float, model_version: str):
        is_sdxl = 0
        match model_version:
            case 'SDXL_2048':
                is_sdxl = 1

        if (is_sdxl == 1):
            dimension_x = round(width * multiply_sdxl)
            dimension_y = round(height * multiply_sdxl)
            ratio = round(multiply_sdxl, 2)
        else:
            dimension_x = round(width * multiply_sd)
            dimension_y = round(height * multiply_sd)
            ratio = round(multiply_sd, 2)

        return (dimension_x, dimension_y, ratio)

class PrimereStepsCfg:
  RETURN_TYPES = ("INT", "FLOAT")
  RETURN_NAMES = ("STEPS", "CFG")
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

class PrimereClearPrompt:
  RETURN_TYPES = ("STRING", "STRING")
  RETURN_NAMES = ("PROMPT+", "PROMPT-")
  FUNCTION = "clean_prompt"
  CATEGORY = TREE_DASHBOARD

  @classmethod
  def INPUT_TYPES(cls):
      return {
          "required": {
              "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
              "positive_prompt": ("STRING", {"forceInput": True}),
              "negative_prompt": ("STRING", {"forceInput": True}),
              "remove_only_if_sdxl": ("BOOLEAN", {"default": False}),
              "remove_comfy_embedding": ("BOOLEAN", {"default": False}),
              "remove_a1111_embedding": ("BOOLEAN", {"default": False}),
              "remove_lora": ("BOOLEAN", {"default": False}),
              "remove_hypernetwork": ("BOOLEAN", {"default": False}),
          },
      }

  def clean_prompt(self, positive_prompt, negative_prompt, remove_comfy_embedding, remove_a1111_embedding, remove_lora, remove_hypernetwork, remove_only_if_sdxl, model_version = 'BaseModel_1024'):
      NETWORK_START = []

      is_sdxl = 0
      match model_version:
          case 'SDXL_2048':
              is_sdxl = 1

      if remove_only_if_sdxl == True and is_sdxl == 0:
          return (positive_prompt, negative_prompt,)

      if remove_comfy_embedding == True:
          NETWORK_START.append('embedding:')

      if remove_lora == True:
          NETWORK_START.append('<lora:')

      if remove_hypernetwork == True:
          NETWORK_START.append('<hypernet:')

      if remove_a1111_embedding == True:
          positive_prompt = positive_prompt.replace('embedding:', '')
          negative_prompt = negative_prompt.replace('embedding:', '')
          EMBEDDINGS = folder_paths.get_filename_list("embeddings")
          for embeddings_path in EMBEDDINGS:
              path = Path(embeddings_path)
              embedding_name = path.stem
              positive_prompt = re.sub("(\(" + embedding_name + ":\d+\.\d+\))|(\(" + embedding_name + ":\d+\))|(" + embedding_name + ":\d+\.\d+)|(" + embedding_name + ":\d+)|(" + embedding_name + ":)|(\(" + embedding_name + "\))|(" + embedding_name + ")", "", positive_prompt)
              negative_prompt = re.sub("(\(" + embedding_name + ":\d+\.\d+\))|(\(" + embedding_name + ":\d+\))|(" + embedding_name + ":\d+\.\d+)|(" + embedding_name + ":\d+)|(" + embedding_name + ":)|(\(" + embedding_name + "\))|(" + embedding_name + ")", "", negative_prompt)
              positive_prompt = re.sub(r'(, )\1+', r', ', positive_prompt).strip(', ').replace(' ,', ',')
              negative_prompt = re.sub(r'(, )\1+', r', ', negative_prompt).strip(', ').replace(' ,', ',')

      if len(NETWORK_START) > 0:
        NETWORK_END = ['\n', '>', ' ', ',', '}', ')', '|'] + NETWORK_START
        positive_prompt = utility.clear_prompt(NETWORK_START, NETWORK_END, positive_prompt)
        negative_prompt = utility.clear_prompt(NETWORK_START, NETWORK_END, negative_prompt)

      return (positive_prompt, negative_prompt,)