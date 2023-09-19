import custom_nodes.ComfyUI_Primere_Nodes.components.fields as field
from custom_nodes.ComfyUI_Primere_Nodes.components.tree import TREE_IO

import folder_paths as comfy_paths
import os
import re
import json
import time
import socket
import numpy as np
import pyexiv2
from PIL.PngImagePlugin import PngInfo
from .sd_prompt_reader.image_data_reader import ImageDataReader
from PIL import Image, ImageOps
import hashlib
import difflib

## 3 input summarizer --------------
class ThreeSumNode:
    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Value_A": field.FLOAT,
                "Value_B": field.FLOAT,
                "Value_C": field.FLOAT,
            },
        }

    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "sum"
    CATEGORY = TREE_IO

    def sum(self, Value_A, Value_B, Value_C):
        total = float(Value_A + Value_B + Value_C)
        totalint = int(Value_A + Value_B + Value_C)
        return (total, totalint)


## Image and meta saver  --------------
# ! SYSTEM HOOKS
ALLOWED_EXT = ('.jpeg', '.jpg', '.png', '.tiff', '.gif', '.bmp', '.webp')
NODE_FILE = os.path.abspath(__file__)
NODE_ROOT = os.path.dirname(NODE_FILE)

class PrimereMetaSave:
    def __init__(self):
        self.output_dir = comfy_paths.output_directory
        self.type = 'output'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_path": ("STRING", {"default": '[time(%Y-%m-%d)]', "multiline": False}),
                "subpath": (
                ["None", "Dev", "Test", "Production", "Project", "Portfolio", "Fun"], {"default": "Project"}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "filename_delimiter": ("STRING", {"default": "_"}),
                "filename_number_padding": ("INT", {"default": 2, "min": 1, "max": 9, "step": 1}),
                "filename_number_start": ("BOOLEAN", {"default":False}),
                "extension": (['png', 'jpeg', 'jpg', 'gif', 'tiff', 'webp'], {"default": "jpg"}),
                "png_embed_workflow": ("BOOLEAN", {"default":False}),
                "image_embed_exif": ("BOOLEAN", {"default":False}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "overwrite_mode": (["false", "prefix_as_filename"],),
            },
            "optional": {
                "positive_g": ('STRING', {"forceInput": True}),
                "negative_g": ('STRING', {"forceInput": True}),
                "positive_l": ('STRING', {"forceInput": True}),
                "negative_l": ('STRING', {"forceInput": True}),
                "positive_refiner": ('STRING', {"forceInput": True}),
                "negative_refiner": ('STRING', {"forceInput": True}),
                "model_name": ('STRING', {"forceInput": True}),
                "sampler_name": ('STRING', {"forceInput": True}),
                "seed": ('INT', {"forceInput": True}),
                "original_width": ('INT', {"forceInput": True}),
                "original_height": ('INT', {"forceInput": True}),
                "cfg_scale": ('FLOAT', {"forceInput": True}),
                "steps": ('INT', {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images_meta"
    OUTPUT_NODE = True
    CATEGORY = TREE_IO

    def save_images_meta(self, images, positive_g='', negative_g='', positive_l='', negative_l='', positive_refiner='',
                         negative_refiner='', seed=0, model_hash="", model_name='', sampler_name='', original_width=0,
                         original_height=0, steps=0, cfg_scale=0,
                         output_path='', subpath='', filename_prefix="ComfyUI", filename_delimiter='_',
                         extension='png', quality=95, prompt=None, extra_pnginfo=None,
                         overwrite_mode='false', filename_number_padding=2, filename_number_start=False,
                         png_embed_workflow=False, image_embed_exif=False):

        delimiter = filename_delimiter
        number_padding = filename_number_padding

        # Define token system
        tokens = TextTokens()

        original_output = self.output_dir
        # Parse prefix tokens
        filename_prefix = tokens.parseTokens(filename_prefix)

        # Setup output path
        if output_path in [None, '', "none", "."]:
            output_path = self.output_dir
        else:
            output_path = tokens.parseTokens(output_path)
        if not os.path.isabs(output_path):
            output_path = os.path.join(self.output_dir, output_path)
        base_output = os.path.basename(output_path)
        if output_path.endswith("ComfyUI/output") or output_path.endswith("ComfyUI\output"):
            base_output = ""

        if subpath != 'None': output_path = output_path + os.sep + subpath

        # Check output destination
        if output_path.strip() != '':
            if not os.path.isabs(output_path):
                output_path = os.path.join(comfy_paths.output_directory, output_path)
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)

        # Find existing counter values
        if filename_number_start == 'true':
            pattern = f"(\\d{{{filename_number_padding}}}){re.escape(delimiter)}{re.escape(filename_prefix)}"
        else:
            pattern = f"{re.escape(filename_prefix)}{re.escape(delimiter)}(\\d{{{filename_number_padding}}})"
        existing_counters = [
            int(re.search(pattern, filename).group(1))
            for filename in os.listdir(output_path)
            if re.match(pattern, os.path.basename(filename))
        ]
        existing_counters.sort(reverse=True)

        # Set initial counter value
        if existing_counters:
            counter = existing_counters[0] + 1
        else:
            counter = 1

        # Set initial counter value
        if existing_counters:
            counter = existing_counters[0] + 1
        else:
            counter = 1

        # Set Extension
        file_extension = '.' + extension
        if file_extension not in ALLOWED_EXT:
            print(f"The extension `{extension}` is not valid. The valid formats are: {', '.join(sorted(ALLOWED_EXT))}")
            file_extension = "jpg"

        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            metadata = PngInfo()
            if png_embed_workflow == 'true':
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            if overwrite_mode == 'prefix_as_filename':
                file = f"{filename_prefix}{file_extension}"
            else:
                if filename_number_start == 'true':
                    file = f"{counter:0{number_padding}}{delimiter}{filename_prefix}{file_extension}"
                else:
                    file = f"{filename_prefix}{delimiter}{counter:0{number_padding}}{file_extension}"
                if os.path.exists(os.path.join(output_path, file)):
                    counter += 1
            try:
                output_file = os.path.abspath(os.path.join(output_path, file))
                '''
                if extension == 'png': img.save(output_file, pnginfo=metadata, optimize=True)
                elif extension == 'webp': img.save(output_file, quality=quality)
                elif extension == 'jpeg': img.save(output_file, quality=quality, optimize=True)
                elif extension == 'jpg': img.save(output_file, quality=quality, optimize=True)
                elif extension == 'tiff': img.save(output_file, quality=quality, optimize=True)
                elif extension == 'webp': img.save(output_file, quality=quality, lossless=lossless_webp, exif=metadata)
                '''

                model_hash = 'fakemodelhash1122'

                exif_metadata_A11 = f"""{positive_g}
Negative prompt: {negative_g}
Steps: {str(steps)}, Sampler: {sampler_name}, CFG scale: {str(cfg_scale)}, Seed: {str(seed)}, Size: {str(original_width)}x{str(original_height)}, Model hash: {model_hash}, Model: {model_name}"""

                exif_metadata_json = {}
                exif_metadata_json['positive_g'] = positive_g
                exif_metadata_json['negative_g'] = negative_g
                exif_metadata_json['positive_l'] = positive_l
                exif_metadata_json['negative_l'] = negative_l
                exif_metadata_json['positive_refiner'] = positive_refiner
                exif_metadata_json['negative_refiner'] = negative_refiner
                exif_metadata_json['seed'] = str(seed)
                exif_metadata_json['model_hash'] = model_hash
                exif_metadata_json['model_name'] = model_name
                exif_metadata_json['sampler_name'] = sampler_name
                exif_metadata_json['original_width'] = str(original_width)
                exif_metadata_json['original_height'] = str(original_height)
                exif_metadata_json['steps'] = str(steps)
                exif_metadata_json['cfg_scale'] = str(cfg_scale)

                # print(f"Metadata input: {exif_metadata}")
                # print(f"A11 Metadata input: {exif_metadata_A11}")
                # print(f"PNG Metadata input: {metadata}")

                if extension == 'png':
                    img.save(output_file, pnginfo=metadata, optimize=True)
                elif extension == 'webp':
                    img.save(output_file, quality=quality, exif=metadata)
                else:
                    img.save(output_file, quality=quality, optimize=True)
                    # exif_dict = piexif.load(output_file)
                    # exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(exif_metadata_A11, encoding="unicode")
                    # exif_dict["Exif"][piexif.IFD0.ImageDescription] = piexif.helper.ImageDescription.dump(json.dumps(userdata), encoding="ascii")
                    # piexif.insert(
                    #     piexif.dump(exif_dict),
                    #     output_file
                    # )
                    if image_embed_exif == True:
                        metadata = pyexiv2.Image(output_file)
                        metadata.modify_exif({'Exif.Photo.UserComment': 'charset=Unicode ' + exif_metadata_A11})
                        metadata.modify_exif({'Exif.Image.ImageDescription': json.dumps(exif_metadata_json)})
                        print(f"Image file saved to with exif: {output_file}")
                    else:
                        if extension == 'webp':
                            img.save(output_file, quality=quality, exif=metadata)
                        else:
                            img.save(output_file, quality=quality, optimize=True)
                            print(f"Image file saved to without exif: {output_file}")

            except OSError as e:
                print(f'Unable to save file to: {output_file}')
                print(e)
            except Exception as e:
                print('Unable to save file due to the to the following error:')
                print(e)

            if overwrite_mode == 'false':
                counter += 1

        filtered_paths = []

        if filtered_paths:
            for image_path in filtered_paths:
                subfolder = self.get_subfolder_path(image_path, self.output_dir)
                image_data = {
                    "filename": os.path.basename(image_path),
                    "subfolder": subfolder,
                    "type": self.type
                }
                results.append(image_data)

        return {"ui": {"images": []}}

    def get_subfolder_path(self, image_path, output_path):
        output_parts = output_path.strip(os.sep).split(os.sep)
        image_parts = image_path.strip(os.sep).split(os.sep)
        common_parts = os.path.commonprefix([output_parts, image_parts])
        subfolder_parts = image_parts[len(common_parts):]
        subfolder_path = os.sep.join(subfolder_parts[:-1])
        return subfolder_path

class TextTokens:
    def __init__(self):

        self.tokens = {
            '[time]': str(time.time()).replace('.', '_'),
            '[hostname]': socket.gethostname(),
        }

        if '.' in self.tokens['[time]']: self.tokens['[time]'] = self.tokens['[time]'].split('.')[0]

        try:
            self.tokens['[user]'] = (os.getlogin() if os.getlogin() else 'null')
        except Exception:
            self.tokens['[user]'] = 'null'

    def format_time(self, format_code):
        return time.strftime(format_code, time.localtime(time.time()))

    def parseTokens(self, text):
        tokens = self.tokens.copy()

        # Update time
        tokens['[time]'] = str(time.time())
        if '.' in tokens['[time]']:
            tokens['[time]'] = tokens['[time]'].split('.')[0]

        for token, value in tokens.items():
            if token.startswith('[time('):
                continue
            text = text.replace(token, value)

        def replace_custom_time(match):
            format_code = match.group(1)
            return self.format_time(format_code)

        text = re.sub(r'\[time\((.*?)\)\]', replace_custom_time, text)

        return text

class PrimereMetaRead:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = comfy_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "use_exif": ("BOOLEAN", {"default": False}),
                "use_model": ("BOOLEAN", {"default": False}),
                "model_hash_check": ("BOOLEAN", {"default": False}),
                "use_sampler": ("BOOLEAN", {"default": False}),
                "use_seed": ("BOOLEAN", {"default": False}),
                "use_size": ("BOOLEAN", {"default": False}),
                "use_cfg_scale": ("BOOLEAN", {"default": False}),
                "use_steps": ("BOOLEAN", {"default": False}),
                "image": (sorted(files),),
            },
            "optional": {
                "positive_g": ('STRING', {"forceInput": True, "default": ""}),
                "negative_g": ('STRING', {"forceInput": True, "default": ""}),
                "positive_l": ('STRING', {"forceInput": True, "default": ""}),
                "negative_l": ('STRING', {"forceInput": True, "default": ""}),
                "positive_refiner": ('STRING', {"forceInput": True, "default": ""}),
                "negative_refiner": ('STRING', {"forceInput": True, "default": ""}),
                "model_name": ('STRING', {"forceInput": True, "default": ""}),
                "sampler_name": ('STRING', {"forceInput": True, "default": ""}),
                "seed": ('INT', {"forceInput": True, "default": 0}),
                "original_width": ('INT', {"forceInput": True, "default": 0}),
                "original_height": ('INT', {"forceInput": True, "default": 0}),
                "cfg_scale": ('FLOAT', {"forceInput": True, "default": 7}),
                "steps": ('INT', {"forceInput": True, "default": 12}),
            },

        }

    CATEGORY = TREE_IO
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "INT", "INT", "INT", "FLOAT", "INT")
    RETURN_NAMES = ("positive", "negative", "model_hash", "model_name", "sampler_name", "seed", "width", "height", "cfg", "steps")
    FUNCTION = "load_image"

    def load_image(self, image, use_exif, use_model, model_hash_check, use_sampler, use_seed, use_size, use_cfg_scale, use_steps,
                   positive_g="", negative_g="", positive_l="", negative_l="", positive_refiner="", negative_refiner="",
                   model_hash="", model_name="", sampler_name="DPM++ SDE Karras", seed=0, original_width=512, original_height=512, cfg_scale=7, steps=12):

        if use_exif == True:
            image_path = comfy_paths.get_annotated_filepath(image)

            def get_model_hash(filename):
                hash_sha256 = hashlib.sha256()
                blksize = 1024 * 1024

                with open(filename, "rb") as f:
                    for chunk in iter(lambda: f.read(blksize), b""):
                        hash_sha256.update(chunk)

                return hash_sha256.hexdigest()[0:10]

            def check_model_from_exif(model_hash_exif, model_name_exif, model_name, model_hash_check):
                checkpointpaths = comfy_paths.get_folder_paths("checkpoints")[0]
                allcheckpoints = comfy_paths.get_filename_list("checkpoints")
                source_model_name = model_name_exif.split('_', 1)[-1]

                cutoff_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
                is_found = []

                for trycut in cutoff_list:
                    is_found = difflib.get_close_matches(model_name_exif, allcheckpoints, cutoff=trycut)
                    if len(is_found) == 1:
                        break

                if len(is_found) != 1:
                    for trycut in cutoff_list:
                        is_found = difflib.get_close_matches(source_model_name, allcheckpoints, cutoff=trycut)
                        if len(is_found) == 1:
                            break

                if len(is_found) == 1:
                    valid_model = is_found[0]
                    model_full_path = checkpointpaths + os.sep + valid_model

                    if model_hash_check == True:
                        match_model_hash = get_model_hash(model_full_path)
                        if match_model_hash == model_hash_exif:
                            model_name = valid_model
                        else:
                            print('Model name:' + model_name_exif + ' not available by hashcheck, using system source: ' + model_name)
                    else:
                        model_name = valid_model
                else:
                    print('Model name:' + model_name_exif + ' not available by diffcheck, using system source: ' + model_name)
                return model_name

            reader = ImageDataReader(image_path)
            if (reader.tool == ''):
                print('Reader tool return empty, using node input')
                return (positive_g, negative_g, model_hash, model_name, sampler_name, seed, original_width, original_height, cfg_scale, steps)

            try:
                if use_model == True:
                    model_hash_exif = reader.parameter["model_hash"]
                    model_name_exif = reader.parameter["model"]
                    model_name = check_model_from_exif(model_hash_exif, model_name_exif, model_name, model_hash_check)

                if use_sampler ==True:
                    sampler_name = reader.parameter["sampler"]

                if use_seed == True:
                    seed = int(reader.parameter["seed"])

                if use_cfg_scale == True:
                    cfg_scale = float(reader.parameter["cfg"])

                if use_steps == True:
                    steps = int(reader.parameter["steps"])

                if use_size == True:
                    size = reader.parameter["size"]
                    sizeSplit = size.split("x")
                    original_width = int(sizeSplit[0])
                    original_height = int(sizeSplit[1])

            except ValueError as VE:
                print(VE)
                return (positive_g, negative_g, model_hash, model_name, sampler_name, seed, original_width, original_height, cfg_scale, steps)

            return (reader.positive, reader.negative, model_hash, model_name, sampler_name, seed, original_width, original_height, cfg_scale, steps)
        else:
            return (positive_g, negative_g, model_hash, model_name, sampler_name, seed, original_width, original_height, cfg_scale, steps)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = comfy_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not comfy_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True
