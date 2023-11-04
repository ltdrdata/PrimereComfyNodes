from custom_nodes.ComfyUI_Primere_Nodes.components.tree import TREE_INPUTS
from custom_nodes.ComfyUI_Primere_Nodes.components.tree import PRIMERE_ROOT
import os
import re
from dynamicprompts.parser.parse import ParserConfig
from dynamicprompts.wildcards.wildcard_manager import WildcardManager
from dynamicprompts.generators import RandomPromptGenerator
import chardet
import pandas
import comfy.samplers
import folder_paths
import hashlib
from .modules.image_meta_reader import ImageExifReader
from .modules import exif_data_checker

class PrimereDoublePrompt:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "get_prompt"
    CATEGORY = TREE_INPUTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def get_prompt(self, positive_prompt, negative_prompt):
        return positive_prompt, negative_prompt,


class PrimereStyleLoader:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "load_csv"
    CATEGORY = TREE_INPUTS

    @staticmethod
    def load_styles_csv(styles_path: str):
        fileTest = open(styles_path, 'rb').readline()
        result = chardet.detect(fileTest)
        ENCODING = result['encoding']
        if ENCODING == 'ascii':
            ENCODING = 'UTF-8'

        with open(styles_path, "r", newline = '', encoding = ENCODING) as csv_file:
            try:
                return pandas.read_csv(csv_file)
            except pandas.errors.ParserError as e:
                errorstring = repr(e)
                matchre = re.compile('Expected (\d+) fields in line (\d+), saw (\d+)')
                (expected, line, saw) = map(int, matchre.search(errorstring).groups())
                print(f'Error at line {line}. Fields added : {saw - expected}.')

    @classmethod
    def INPUT_TYPES(cls):
        STYLE_DIR = os.path.join(PRIMERE_ROOT, 'stylecsv')
        cls.styles_csv = cls.load_styles_csv(os.path.join(STYLE_DIR, "styles.csv"))
        return {
            "required": {
                "styles": (list(cls.styles_csv['name']),),
            },
        }

    def load_csv(self, styles):
        try:
            positive_prompt = self.styles_csv[self.styles_csv['name'] == styles]['prompt'].values[0]
        except Exception:
            positive_prompt = ''

        try:
            negative_prompt = self.styles_csv[self.styles_csv['name'] == styles]['negative_prompt'].values[0]
        except Exception:
            negative_prompt = ''

        pos_type = type(positive_prompt).__name__
        neg_type = type(negative_prompt).__name__
        if (pos_type != 'str'):
            positive_prompt = ''
        if (neg_type != 'str'):
            negative_prompt = ''

        return (positive_prompt, negative_prompt)


class PrimereDynParser:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "dyndecoder"
    CATEGORY = TREE_INPUTS

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dyn_prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "seed": ("INT", {"default": 0, "min": -18446744073709551615, "max": 18446744073709551615, "forceInput": True}),
            }
        }

    def __init__(self):
        wildcard_dir = os.path.join(PRIMERE_ROOT, 'wildcards')
        self._wildcard_manager = WildcardManager(wildcard_dir)
        self._parser_config = ParserConfig(
            variant_start = "{",
            variant_end = "}",
            wildcard_wrap = "__"
        )

    def dyndecoder(self, dyn_prompt, seed):
        prompt_generator = RandomPromptGenerator(
            self._wildcard_manager,
            seed = seed,
            parser_config = self._parser_config,
            unlink_seed_from_prompt = False,
            ignore_whitespace = False
        )

        dyn_type = type(dyn_prompt).__name__
        if (dyn_type != 'str'):
            dyn_prompt = ''

        try:
            all_prompts = prompt_generator.generate(dyn_prompt, 1) or [""]
        except Exception:
            all_prompts = [""]

        prompt = all_prompts[0]
        return (prompt, )

class PrimereVAESelector:
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("VAE",)
    FUNCTION = "primere_vae_selector"
    CATEGORY = TREE_INPUTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_sd": ("VAE",),
                "vae_sdxl": ("VAE",),
                "is_sdxl": ("INT", {"default": 0, "forceInput": True}),
            }
        }

    def primere_vae_selector(self, vae_sd, vae_sdxl, is_sdxl = 0):
        if int(round(is_sdxl)) == 1:
            return (vae_sdxl, )
        else:
            return (vae_sd, )

class PrimereMetaRead:
    CATEGORY = TREE_INPUTS
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "CHECKPOINT_NAME", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "INT", "INT", "FLOAT", "INT", "VAE_NAME", "TUPLE")
    RETURN_NAMES = ("prompt+", "prompt-", "prompt L+", "prompt L-", "refiner+", "refiner-", "model_name", "sampler_name", "scheduler_name", "seed", "width", "height", "cfg", "steps", "vae", "metadata")
    FUNCTION = "load_image_meta"

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {
            "required": {
                "is_sdxl": ("INT", {"default": 0, "forceInput": True}),
                "use_exif": ("BOOLEAN", {"default": True}),
                "use_model": ("BOOLEAN", {"default": True}),
                "model_hash_check": ("BOOLEAN", {"default": False}),
                "use_sampler": ("BOOLEAN", {"default": True}),
                "use_seed": ("BOOLEAN", {"default": True}),
                "use_size": ("BOOLEAN", {"default": True}),
                "use_cfg_scale": ("BOOLEAN", {"default": True}),
                "use_steps": ("BOOLEAN", {"default": True}),
                "use_vae": ("BOOLEAN", {"default": True}),
                "image": (sorted(files),),
            },
            "optional": {
                "positive": ('STRING', {"forceInput": True, "default": ""}),
                "negative": ('STRING', {"forceInput": True, "default": ""}),
                "positive_l": ('STRING', {"forceInput": True, "default": ""}),
                "negative_l": ('STRING', {"forceInput": True, "default": ""}),
                "positive_r": ('STRING', {"forceInput": True, "default": ""}),
                "negative_r": ('STRING', {"forceInput": True, "default": ""}),
                "model_name": ('CHECKPOINT_NAME', {"forceInput": True, "default": ""}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True, "default": "euler"}),
                "scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True, "default": "normal"}),
                "seed": ('INT', {"forceInput": True, "default": 1}),
                "width": ('INT', {"forceInput": True, "default": 512}),
                "height": ('INT', {"forceInput": True, "default": 512}),
                "cfg_scale": ('FLOAT', {"forceInput": True, "default": 7}),
                "steps": ('INT', {"forceInput": True, "default": 12}),
                "vae_name_sd": ('VAE_NAME', {"forceInput": True, "default": ""}),
                "vae_name_sdxl": ('VAE_NAME', {"forceInput": True, "default": ""}),
            },
        }

    def load_image_meta(self, is_sdxl, use_exif, use_model, model_hash_check, use_sampler, use_seed, use_size, use_cfg_scale, use_steps, use_vae, image,
                        positive="", negative="", positive_l="", negative_l="", positive_r="", negative_r="",
                        model_hash="", model_name="", sampler_name="euler", scheduler_name="normal", seed=1, width=512, height=512, cfg_scale=7, steps=12, vae_name_sd="", vae_name_sdxl=""):

        data_json = {}
        data_json['positive'] = positive
        data_json['negative'] = negative
        data_json['positive_l'] = positive_l
        data_json['negative_l'] = negative_l
        data_json['positive_r'] = positive_r
        data_json['negative_r'] = negative_r
        data_json['model_hash'] = model_hash
        data_json['model_name'] = model_name
        data_json['sampler_name'] = sampler_name
        data_json['scheduler_name'] = scheduler_name
        data_json['seed'] = seed
        data_json['width'] = width
        data_json['height'] = height
        data_json['cfg_scale'] = cfg_scale
        data_json['steps'] = steps
        data_json['is_sdxl'] = is_sdxl

        if (is_sdxl == True):
            vae_name = vae_name_sdxl
            data_json['vae_name'] = vae_name_sdxl
        else:
            vae_name = vae_name_sd
            data_json['vae_name'] = vae_name_sd


        if use_exif:
            image_path = folder_paths.get_annotated_filepath(image)
            if os.path.isfile(image_path):
                readerResult = ImageExifReader(image_path)
                if (type(readerResult.parser).__name__ == 'dict'):
                    print('Reader tool return empty, using node input')
                    return (positive, negative, positive_l, negative_l, positive_r, negative_r, model_name, sampler_name, scheduler_name, seed, width, height, cfg_scale, steps, vae_name_sd, data_json)

                reader = readerResult.parser

                if 'positive' in reader.parameter:
                    data_json['positive'] = reader.parameter["positive"]
                else:
                    data_json['positive'] = ""

                if 'negative' in reader.parameter:
                    data_json['negative'] = reader.parameter["negative"]
                else:
                    data_json['negative'] = ""

                if (readerResult.tool == ''):
                    print('Reader tool return empty, using node input')
                    return (positive, negative, positive_l, negative_l, positive_r, negative_r, model_name, sampler_name, scheduler_name, seed, width, height, cfg_scale, steps, vae_name_sd, data_json)

                try:
                    if use_model == True:
                        if 'model_hash' in reader.parameter:
                            data_json['model_hash'] = reader.parameter["model_hash"]
                        else:
                            checkpointpaths = folder_paths.get_folder_paths("checkpoints")[0]
                            model_full_path = checkpointpaths + os.sep + model_name
                            if os.path.isfile(model_full_path):
                                data_json['model_hash'] = exif_data_checker.get_model_hash(model_full_path)
                            else:
                                data_json['model_hash'] = 'no_hash_data'

                        if 'model' in reader.parameter:
                            model_name_exif = reader.parameter["model"]
                            data_json['model_name'] = exif_data_checker.check_model_from_exif(data_json['model_hash'], model_name_exif, model_name, model_hash_check)
                        else:
                            data_json['model_name'] = folder_paths.get_filename_list("checkpoints")[0]

                    if use_sampler == True:
                        if 'sampler' in reader.parameter:
                            sampler_name_exif = reader.parameter["sampler"]
                            samplers = exif_data_checker.check_sampler_from_exif(sampler_name_exif.lower(), sampler_name, scheduler_name)
                            data_json['sampler_name'] = samplers['sampler']
                            data_json['scheduler_name'] = samplers['scheduler']

                    if use_seed == True:
                        if 'seed' in reader.parameter:
                            data_json['seed'] = reader.parameter["seed"]

                    if use_cfg_scale == True:
                        if 'cfg_scale' in reader.parameter:
                            data_json['cfg_scale'] = reader.parameter["cfg_scale"]

                    if use_steps == True:
                        if 'steps' in reader.parameter:
                            data_json['steps'] = reader.parameter["steps"]

                    if use_vae == True:
                        if 'vae' in reader.parameter:
                            vae_name_exif = reader.parameter["vae"]
                            print(vae_name_exif)
                            vae = exif_data_checker.check_vae_exif(vae_name_exif.lower(), vae_name)
                            data_json['vae_name'] = vae

                    if use_size == True:
                        if 'size_string' in reader.parameter:
                            data_json['width'] = reader.parameter["width"]
                            data_json['height'] = reader.parameter["height"]

                    return (data_json['positive'], data_json['negative'], data_json['positive_l'], data_json['negative_l'], data_json['positive_r'], data_json['negative_r'], data_json['model_name'], data_json['sampler_name'], data_json['scheduler_name'], data_json['seed'], data_json['width'], data_json['height'], data_json['cfg_scale'], data_json['steps'], data_json['vae_name'], data_json)

                except ValueError as VE:
                    print(VE)
                    return (data_json['positive'], data_json['negative'], data_json['positive_l'], data_json['negative_l'], data_json['positive_r'], data_json['negative_r'], data_json['model_name'], data_json['sampler_name'], data_json['scheduler_name'], data_json['seed'], data_json['width'], data_json['height'], data_json['cfg_scale'], data_json['steps'], data_json['vae_name'], data_json)

            else:
                print('No source image loaded')
                return (data_json['positive'], data_json['negative'], data_json['positive_l'], data_json['negative_l'], data_json['positive_r'], data_json['negative_r'], data_json['model_name'], data_json['sampler_name'], data_json['scheduler_name'], data_json['seed'], data_json['width'], data_json['height'], data_json['cfg_scale'], data_json['steps'], data_json['vae_name'], data_json)

        else:
            print('Exif reader off')
            return (data_json['positive'], data_json['negative'], data_json['positive_l'], data_json['negative_l'], data_json['positive_r'], data_json['negative_r'], data_json['model_name'], data_json['sampler_name'], data_json['scheduler_name'], data_json['seed'], data_json['width'], data_json['height'], data_json['cfg_scale'], data_json['steps'], data_json['vae_name'], data_json)

    @classmethod
    def IS_CHANGED(cls, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()