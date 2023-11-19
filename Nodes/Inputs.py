from custom_nodes.ComfyUI_Primere_Nodes.components.tree import TREE_INPUTS
from custom_nodes.ComfyUI_Primere_Nodes.components.tree import PRIMERE_ROOT
import os
import re
from dynamicprompts.parser.parse import ParserConfig
from dynamicprompts.wildcards.wildcard_manager import WildcardManager
import chardet
import pandas
import comfy.samplers
import folder_paths
import hashlib
from .modules.image_meta_reader import ImageExifReader
from .modules import exif_data_checker
import nodes
from custom_nodes.ComfyUI_Primere_Nodes.components import utility
from pathlib import Path
import random

class PrimereDoublePrompt:
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("PROMPT+", "PROMPT-", "SUBPATH", "MODEL", "ORIENTATION")
    FUNCTION = "get_prompt"
    CATEGORY = TREE_INPUTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "subpath": ("STRING", {"default": "", "multiline": False}),
                "model": (["None"] + folder_paths.get_filename_list("checkpoints"), {"default": "None"}),
                "orientation": (["None", "Random", "Horizontal", "Vertical"], {"default": "None"}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "id": "UNIQUE_ID",
            },
        }

    def get_prompt(self, positive_prompt, negative_prompt, extra_pnginfo, id, subpath="", model="", orientation=""):
        def debug_state(self, extra_pnginfo, id):
            workflow = extra_pnginfo["workflow"]
            for node in workflow["nodes"]:
                node_id = str(node["id"])
                name = node["type"]
                if node_id == id and name == 'PrimerePrompt':
                    if "Debug" in name or "Show" in name or "Function" in name or "Evaluate" in name:
                        continue

                    return node['widgets_values']

        rawResult = debug_state(self, extra_pnginfo, id)
        if not rawResult:
            rawResult = (positive_prompt, negative_prompt)

        if len(subpath.strip()) < 1 or subpath.strip() == 'None':
            subpath = None
        if len(model.strip()) < 1 or model.strip() == 'None':
            model = None
        if len(orientation.strip()) < 1 or orientation.strip() == 'None':
            orientation = None

        if orientation == 'Random':
            orientations = ["Horizontal", "Vertical"]
            orientation = random.choice(orientations)

        return (rawResult[0].replace('\n', ' '), rawResult[1].replace('\n', ' '), subpath, model, orientation)

class PrimereStyleLoader:
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("PROMPT+", "PROMPT-", "SUBPATH", "MODEL", "ORIENTATION")
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
                "styles": (sorted(list(cls.styles_csv['name'])),),
                "use_subpath": ("BOOLEAN", {"default": False}),
                "use_model": ("BOOLEAN", {"default": False}),
                "use_orientation": ("BOOLEAN", {"default": False}),
            },
        }

    def load_csv(self, styles, use_subpath, use_model, use_orientation):
        try:
            positive_prompt = self.styles_csv[self.styles_csv['name'] == styles]['prompt'].values[0]
        except Exception:
            positive_prompt = ''

        try:
            negative_prompt = self.styles_csv[self.styles_csv['name'] == styles]['negative_prompt'].values[0]
        except Exception:
            negative_prompt = ''

        try:
            prefered_subpath = self.styles_csv[self.styles_csv['name'] == styles]['prefered_subpath'].values[0]
        except Exception:
            prefered_subpath = ''

        try:
            prefered_model = self.styles_csv[self.styles_csv['name'] == styles]['prefered_model'].values[0]
        except Exception:
            prefered_model = ''

        try:
            prefered_orientation = self.styles_csv[self.styles_csv['name'] == styles]['prefered_orientation'].values[0]
        except Exception:
            prefered_orientation = ''

        pos_type = type(positive_prompt).__name__
        neg_type = type(negative_prompt).__name__
        subp_type = type(prefered_subpath).__name__
        model_type = type(prefered_model).__name__
        orientation_type = type(prefered_orientation).__name__
        if (pos_type != 'str'):
            positive_prompt = ''
        if (neg_type != 'str'):
            negative_prompt = ''
        if (subp_type != 'str'):
            prefered_subpath = ''
        if (model_type != 'str'):
            prefered_model = ''
        if (orientation_type != 'str'):
            prefered_orientation = ''

        if len(prefered_subpath.strip()) < 1:
            prefered_subpath = None
        if len(prefered_model.strip()) < 1:
            prefered_model = None
        if len(prefered_orientation.strip()) < 1:
            prefered_orientation = None

        if use_subpath == False:
            prefered_subpath = None
        if use_model == False:
            prefered_model = None
        if use_orientation == False:
            prefered_orientation = None

        return (positive_prompt, negative_prompt, prefered_subpath, prefered_model, prefered_orientation)

class PrimereDynParser:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PROMPT",)
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
        prompt = utility.DynPromptDecoder(self, dyn_prompt, seed)
        return (prompt, )

class PrimereEmbeddingHandler:
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("PROMPT+", "PROMPT-",)
    FUNCTION = "embedding_handler"
    CATEGORY = TREE_INPUTS
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "negative_prompt": ("STRING", {"multiline": True, "forceInput": True}),
            }
        }

    def embedding_handler(self, positive_prompt, negative_prompt):
        return (self.EmbeddingConverter(positive_prompt), self.EmbeddingConverter(negative_prompt),)

    def EmbeddingConverter(self, text):
        EMBEDDINGS = folder_paths.get_filename_list("embeddings")
        text = text.replace('embedding:', '')
        for embeddings_path in EMBEDDINGS:
            path = Path(embeddings_path)
            embedding_name = path.stem
            text = text.replace(embedding_name, 'embedding:' + embedding_name)

        return text

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
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
            }
        }

    def primere_vae_selector(self, vae_sd, vae_sdxl, model_version = "BaseModel_1024"):
        if model_version == 'SDXL_2048':
            return (vae_sdxl, )
        else:
            return (vae_sd, )

class PrimereMetaRead:
    CATEGORY = TREE_INPUTS
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "CHECKPOINT_NAME", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "INT", "INT", "FLOAT", "INT", "VAE_NAME", "VAE", "TUPLE")
    RETURN_NAMES = ("PROMPT+", "PROMPT-", "PROMPT L+", "PROMPT L-", "REFINER+", "REFINER-", "MODEL_NAME", "SAMPLER_NAME", "SCHEDULER_NAME", "SEED", "WIDTH", "HEIGHT", "CFG", "STEPS", "VAE_NAME", "VAE", "METADATA")
    FUNCTION = "load_image_meta"

    def __init__(self):
        self.chkp_loader = nodes.CheckpointLoaderSimple()
        self.vae_loader = nodes.VAELoader()
        wildcard_dir = os.path.join(PRIMERE_ROOT, 'wildcards')
        self._wildcard_manager = WildcardManager(wildcard_dir)
        self._parser_config = ParserConfig(
            variant_start = "{",
            variant_end = "}",
            wildcard_wrap = "__"
        )

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {
            "required": {
                "use_exif": ("BOOLEAN", {"default": True}),
                "use_decoded_dyn": ("BOOLEAN", {"default": False}),
                "use_model": ("BOOLEAN", {"default": True}),
                "model_hash_check": ("BOOLEAN", {"default": False}),
                "use_sampler": ("BOOLEAN", {"default": True}),
                "use_seed": ("BOOLEAN", {"default": True}),
                "use_size": ("BOOLEAN", {"default": True}),
                "recount_size": ("BOOLEAN", {"default": False}),
                "use_cfg_scale": ("BOOLEAN", {"default": True}),
                "use_steps": ("BOOLEAN", {"default": True}),
                "use_exif_vae": ("BOOLEAN", {"default": True}),
                "force_model_vae": ("BOOLEAN", {"default": False}),
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
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True, "default": "euler"}),
                "scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True, "default": "normal"}),
                "seed": ('INT', {"forceInput": True, "default": 1}),
                "width": ('INT', {"forceInput": True, "default": 512}),
                "height": ('INT', {"forceInput": True, "default": 512}),
                "cfg_scale": ('FLOAT', {"forceInput": True, "default": 7}),
                "steps": ('INT', {"forceInput": True, "default": 12}),
                "vae_name_sd": ('VAE_NAME', {"forceInput": True, "default": ""}),
                "vae_name_sdxl": ('VAE_NAME', {"forceInput": True, "default": ""}),
                "is_lcm": ("INT", {"default": 0, "forceInput": True}),
                "prefered_model": ("STRING", {"default": "", "forceInput": True}),
                "prefered_orientation": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    def load_image_meta(self, use_exif, use_decoded_dyn, use_model, model_hash_check, use_sampler, use_seed, use_size, recount_size, use_cfg_scale, use_steps, use_exif_vae, force_model_vae, image,
                        positive="", negative="", positive_l="", negative_l="", positive_r="", negative_r="",
                        model_hash="", model_name="", model_version="BaseModel_1024", sampler_name="euler", scheduler_name="normal", seed=1, width=512, height=512, cfg_scale=7, steps=12, vae_name_sd="", vae_name_sdxl="", is_lcm=0, prefered_model="", prefered_orientation=""):

        if prefered_orientation == 'Random':
            if (seed % 2) == 0:
                prefered_orientation = "Horizontal"
            else:
                prefered_orientation = "Vertical"

        data_json = {}
        data_json['positive'] = positive.replace('ADDROW ', '').replace('ADDCOL ', '').replace('ADDCOMM ', '').replace('\n', ' ')
        data_json['negative'] = negative.replace('\n', ' ')
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
        data_json['model_version'] = model_version
        data_json['is_lcm'] = is_lcm
        data_json['vae_name'] = vae_name_sd
        data_json['force_model_vae'] = force_model_vae
        data_json['prefered_model'] = prefered_model
        data_json['prefered_orientation'] = prefered_orientation
        LOADED_CHECKPOINT = None

        is_sdxl = 0
        match model_version:
            case 'SDXL_2048':
                is_sdxl = 1
        data_json['is_sdxl'] = is_sdxl

        if (is_sdxl == 1):
            data_json['vae_name'] = vae_name_sdxl
        else:
            data_json['vae_name'] = vae_name_sd

        if (data_json['vae_name'] == ""):
            data_json['vae_name'] = folder_paths.get_filename_list("vae")[0]

        if use_exif:
            image_path = folder_paths.get_annotated_filepath(image)
            if os.path.isfile(image_path):
                readerResult = ImageExifReader(image_path)

                if (type(readerResult.parser).__name__ == 'dict'):
                    print('Reader tool return empty, using node input')
                    if (force_model_vae == True):
                        LOADED_CHECKPOINT = self.chkp_loader.load_checkpoint(model_name)
                        realvae = LOADED_CHECKPOINT[2]
                    else:
                        realvae = self.vae_loader.load_vae(data_json['vae_name'])[0]

                    return (positive, negative, positive_l, negative_l, positive_r, negative_r, model_name, sampler_name, scheduler_name, seed, width, height, cfg_scale, steps, data_json['vae_name'], realvae, data_json)

                reader = readerResult.parser

                if 'positive' in reader.parameter:
                    data_json['positive'] = reader.parameter["positive"].replace('ADDROW ', '').replace('ADDCOL ', '').replace('ADDCOMM ', '').replace('\n', ' ')
                else:
                    data_json['positive'] = ""

                if 'negative' in reader.parameter:
                    data_json['negative'] = reader.parameter["negative"].replace('\n', ' ')
                else:
                    data_json['negative'] = ""

                data_json['dynamic_positive'] = utility.DynPromptDecoder(self, data_json['positive'], seed)
                data_json['dynamic_negative'] = utility.DynPromptDecoder(self, data_json['negative'], seed)

                if (readerResult.tool == ''):
                    print('Reader tool return empty, using node input')
                    if (force_model_vae == True):
                        LOADED_CHECKPOINT = self.chkp_loader.load_checkpoint(model_name)
                        realvae = LOADED_CHECKPOINT[2]
                    else:
                        realvae = self.vae_loader.load_vae(data_json['vae_name'])[0]

                    return (positive, negative, positive_l, negative_l, positive_r, negative_r, model_name, sampler_name, scheduler_name, seed, width, height, cfg_scale, steps, data_json['vae_name'], realvae, data_json)

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

                        if 'model_name' in reader.parameter:
                            model_name_exif = reader.parameter["model_name"]
                            data_json['model_name'] = exif_data_checker.check_model_from_exif(data_json['model_hash'], model_name_exif, model_name, model_hash_check)
                        else:
                            data_json['model_name'] = folder_paths.get_filename_list("checkpoints")[0]

                    if (data_json['model_name'] != model_name):
                        is_sdxl = 0
                        LOADED_CHECKPOINT = self.chkp_loader.load_checkpoint(data_json['model_name'])
                        model_version = utility.getCheckpointVersion(LOADED_CHECKPOINT[0])
                        data_json['model_version'] = model_version
                        match model_version:
                            case 'SDXL_2048':
                                is_sdxl = 1

                        data_json['is_sdxl'] = is_sdxl

                    if use_sampler == True and data_json['is_lcm'] == 0 and (reader.parameter["cfg_scale"] >= 3 and reader.parameter["steps"] >= 9):
                        if 'sampler' in reader.parameter:
                            sampler_name_exif = reader.parameter["sampler"]
                            samplers = exif_data_checker.check_sampler_from_exif(sampler_name_exif.lower(), sampler_name, scheduler_name)
                            data_json['sampler_name'] = samplers['sampler']
                            data_json['scheduler_name'] = samplers['scheduler']
                        elif ('sampler_name' in reader.parameter and 'scheduler_name' in reader.parameter):
                            data_json['sampler_name'] = reader.parameter["sampler_name"]
                            data_json['scheduler_name'] = reader.parameter["scheduler_name"]

                    if use_seed == True:
                        if 'seed' in reader.parameter:
                            data_json['seed'] = reader.parameter["seed"]

                    if use_cfg_scale == True and data_json['is_lcm'] == 0 and reader.parameter["cfg_scale"] >= 3:
                        if 'cfg_scale' in reader.parameter:
                            data_json['cfg_scale'] = reader.parameter["cfg_scale"]

                    if use_steps == True and data_json['is_lcm'] == 0 and reader.parameter["steps"] >= 9:
                        if 'steps' in reader.parameter:
                            data_json['steps'] = reader.parameter["steps"]

                    if (is_sdxl == 1):
                        data_json['vae_name'] = vae_name_sdxl
                    else:
                        data_json['vae_name'] = vae_name_sd

                    if (data_json['vae_name'] == ""):
                        data_json['vae_name'] = folder_paths.get_filename_list("vae")[0]

                    if force_model_vae == True:
                        if LOADED_CHECKPOINT is not None:
                            realvae = LOADED_CHECKPOINT[2]
                        else:
                            realvae = self.chkp_loader.load_checkpoint(data_json['model_name'])[2]
                    else:
                        if use_exif_vae == True:
                            if 'vae' in reader.parameter:
                                vae_name_exif = reader.parameter["vae"]
                                vae = exif_data_checker.check_vae_exif(vae_name_exif.lower(), data_json['vae_name'])
                                data_json['vae_name'] = vae

                        realvae = self.vae_loader.load_vae(data_json['vae_name'])[0]

                    if use_size == True:
                        if 'size_string' in reader.parameter or ('width' in reader.parameter and 'height' in reader.parameter):
                            data_json['width'] = reader.parameter["width"]
                            data_json['height'] = reader.parameter["height"]
                        if recount_size == True:
                            if (data_json['width'] > data_json['height']):
                                orientation = 'Horizontal'
                            else:
                                orientation = 'Vertical'

                            image_sides = sorted([data_json['width'], data_json['height']])
                            custom_side_b = round((image_sides[1] / image_sides[0]), 4)
                            dimensions = utility.calculate_dimensions(self, "Square [1:1]", orientation, 1, model_version, True, 1, custom_side_b)
                            data_json['width'] = dimensions[0]
                            data_json['height'] = dimensions[1]

                    if use_decoded_dyn == True:
                        if 'dynamic_positive' in reader.parameter:
                            data_json['positive'] = reader.parameter['dynamic_positive']
                        if 'dynamic_negative' in reader.parameter:
                            data_json['negative'] = reader.parameter['dynamic_negative']

                    return (data_json['positive'], data_json['negative'], data_json['positive_l'], data_json['negative_l'], data_json['positive_r'], data_json['negative_r'], data_json['model_name'], data_json['sampler_name'], data_json['scheduler_name'], data_json['seed'], data_json['width'], data_json['height'], data_json['cfg_scale'], data_json['steps'], data_json['vae_name'], realvae, data_json)

                except ValueError as VE:
                    print(VE)
                    if (force_model_vae == True):
                        if LOADED_CHECKPOINT is not None:
                            realvae = LOADED_CHECKPOINT[2]
                        else:
                            realvae = self.chkp_loader.load_checkpoint(data_json['model_name'])[2]
                    else:
                        realvae = self.vae_loader.load_vae(data_json['vae_name'])[0]

                    return (data_json['positive'], data_json['negative'], data_json['positive_l'], data_json['negative_l'], data_json['positive_r'], data_json['negative_r'], data_json['model_name'], data_json['sampler_name'], data_json['scheduler_name'], data_json['seed'], data_json['width'], data_json['height'], data_json['cfg_scale'], data_json['steps'], data_json['vae_name'], realvae, data_json)

            else:
                print('No source image loaded')
                if (force_model_vae == True):
                    if LOADED_CHECKPOINT is not None:
                        realvae = LOADED_CHECKPOINT[2]
                    else:
                        realvae = self.chkp_loader.load_checkpoint(data_json['model_name'])[2]
                else:
                    realvae = self.vae_loader.load_vae(data_json['vae_name'])[0]

                return (data_json['positive'], data_json['negative'], data_json['positive_l'], data_json['negative_l'], data_json['positive_r'], data_json['negative_r'], data_json['model_name'], data_json['sampler_name'], data_json['scheduler_name'], data_json['seed'], data_json['width'], data_json['height'], data_json['cfg_scale'], data_json['steps'], data_json['vae_name'], realvae, data_json)

        else:
            print('Exif reader off')
            if prefered_model is not None and len(prefered_model.strip()) > 0:
                data_json['model_name'] = exif_data_checker.check_model_from_exif("no_hash_data", prefered_model, prefered_model, False)

                is_sdxl = 0
                LOADED_CHECKPOINT = self.chkp_loader.load_checkpoint(data_json['model_name'])
                model_version = utility.getCheckpointVersion(LOADED_CHECKPOINT[0])
                data_json['model_version'] = model_version
                match model_version:
                    case 'SDXL_2048':
                        is_sdxl = 1
                data_json['is_sdxl'] = is_sdxl

            if (force_model_vae == True):
                if LOADED_CHECKPOINT is not None:
                    realvae = LOADED_CHECKPOINT[2]
                else:
                    realvae = self.chkp_loader.load_checkpoint(data_json['model_name'])[2]
            else:
                if (is_sdxl == 1):
                    data_json['vae_name'] = vae_name_sdxl
                else:
                    data_json['vae_name'] = vae_name_sd
                realvae = self.vae_loader.load_vae(data_json['vae_name'])[0]

            data_json['dynamic_positive'] = utility.DynPromptDecoder(self, data_json['positive'], seed)
            data_json['dynamic_negative'] = utility.DynPromptDecoder(self, data_json['negative'], seed)

            if prefered_orientation is not None and len(prefered_orientation.strip()) > 0:
                image_sides = sorted([data_json['width'], data_json['height']])
                custom_side_b = round((image_sides[1] / image_sides[0]), 4)
                dimensions = utility.calculate_dimensions(self, "Square [1:1]", prefered_orientation, 1, model_version, True, 1, custom_side_b)
                data_json['width'] = dimensions[0]
                data_json['height'] = dimensions[1]

                width = dimensions[0]
                height = dimensions[1]
                if prefered_orientation == 'Vertical' and (data_json['width'] > data_json['height']):
                    data_json['width'] = height
                    data_json['height'] = width
                if prefered_orientation == 'Horizontal' and (data_json['height'] > data_json['width']):
                    data_json['width'] = height
                    data_json['height'] = width

            return (data_json['positive'], data_json['negative'], data_json['positive_l'], data_json['negative_l'], data_json['positive_r'], data_json['negative_r'], data_json['model_name'], data_json['sampler_name'], data_json['scheduler_name'], data_json['seed'], data_json['width'], data_json['height'], data_json['cfg_scale'], data_json['steps'], data_json['vae_name'], realvae, data_json)

    @classmethod
    def IS_CHANGED(cls, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()