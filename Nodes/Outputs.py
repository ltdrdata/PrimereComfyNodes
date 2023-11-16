from custom_nodes.ComfyUI_Primere_Nodes.components.tree import TREE_OUTPUTS
from custom_nodes.ComfyUI_Primere_Nodes.components.tree import PRIMERE_ROOT
import os
import folder_paths
import re
import json
import time
import numpy as np
import pyexiv2
from PIL.PngImagePlugin import PngInfo
from PIL import Image, ImageOps
from pathlib import Path
import datetime

ALLOWED_EXT = ('.jpeg', '.jpg', '.png', '.tiff', '.gif', '.bmp', '.webp')

class PrimereMetaSave:
    NODE_FILE = os.path.abspath(__file__)
    NODE_ROOT = os.path.dirname(NODE_FILE)

    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.type = 'output'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "output_path": ("STRING", {"default": '[time(%Y-%m-%d)]', "multiline": False}),
                "subpath": (["None", "Dev", "Test", "Production", "Project", "Portfolio", "Character", "Fun", "SFW", "NSFW"], {"default": "Project"}),
                "prefered_subpath": ("STRING", {"default": "", "forceInput": True}),
                "add_modelname_to_path": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "filename_delimiter": ("STRING", {"default": "_"}),
                "add_date_to_filename": ("BOOLEAN", {"default": True}),
                "add_time_to_filename": ("BOOLEAN", {"default": True}),
                "add_seed_to_filename": ("BOOLEAN", {"default": True}),
                "add_size_to_filename": ("BOOLEAN", {"default": True}),
                "filename_number_padding": ("INT", {"default": 2, "min": 1, "max": 9, "step": 1}),
                "filename_number_start": ("BOOLEAN", {"default":False}),
                "extension": (['png', 'jpeg', 'jpg', 'gif', 'tiff', 'webp'], {"default": "jpg"}),
                "png_embed_workflow": ("BOOLEAN", {"default":False}),
                "image_embed_exif": ("BOOLEAN", {"default":False}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "overwrite_mode": (["false", "prefix_as_filename"],),
                "save_mata_to_json": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "prefered_subpath": ("STRING", {"default": "", "forceInput": True}),
                "image_metadata": ('TUPLE', {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save_images_meta"
    OUTPUT_NODE = True
    CATEGORY = TREE_OUTPUTS

    def save_images_meta(self, images, add_date_to_filename, add_time_to_filename, add_seed_to_filename, add_size_to_filename, save_mata_to_json, image_metadata=None,
                         output_path='[time(%Y-%m-%d)]', subpath='Project', add_modelname_to_path = False, filename_prefix="ComfyUI", filename_delimiter='_',
                         extension='jpg', quality=95, prompt=None, extra_pnginfo=None,
                         overwrite_mode='false', filename_number_padding=2, filename_number_start=False,
                         png_embed_workflow=False, image_embed_exif=False, prefered_subpath=""):

        delimiter = filename_delimiter
        number_padding = filename_number_padding
        tokens = TextTokens()

        original_output = self.output_dir
        filename_prefix = tokens.parseTokens(filename_prefix)
        nowdate = datetime.datetime.now()

        if add_date_to_filename:
            filename_prefix = filename_prefix + '_' + nowdate.strftime("%Y%d%m")
        if add_time_to_filename:
            filename_prefix = filename_prefix + '_' + nowdate.strftime("%H%M%S")
        if add_seed_to_filename:
            if 'seed' in image_metadata:
                filename_prefix = filename_prefix + '_' + str(image_metadata['seed'])
        if add_size_to_filename:
            if 'width' in image_metadata:
                filename_prefix = filename_prefix + '_' + str(image_metadata['width']) + 'x' + str(image_metadata['height'])

        if output_path in [None, '', "none", "."]:
            output_path = self.output_dir
        else:
            output_path = tokens.parseTokens(output_path)
        if not os.path.isabs(output_path):
            output_path = os.path.join(self.output_dir, output_path)
        base_output = os.path.basename(output_path)
        if output_path.endswith("ComfyUI/output") or output_path.endswith("ComfyUI\output"):
            base_output = ""

        if add_modelname_to_path == True:
            path = Path(output_path)
            ModelStartPath = output_path.replace(path.stem, '')
            ModelPath = Path(image_metadata['model_name'])
            if prefered_subpath is not None and len(prefered_subpath.strip()) > 0:
                subpath = prefered_subpath

            if subpath is not None and subpath != 'None' and len(subpath.strip()) > 0:
                output_path = ModelStartPath + ModelPath.stem.upper() + os.sep + subpath + os.sep + path.stem
            else:
                output_path = ModelStartPath + ModelPath.stem.upper() + os.sep + path.stem

        if output_path.strip() != '':
            if not os.path.isabs(output_path):
                output_path = os.path.join(folder_paths.output_directory, output_path)
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)

        if filename_number_start == 'true':
            pattern = f"(\\d{{{filename_number_padding}}}){re.escape(delimiter)}{re.escape(filename_prefix)}"
        else:
            pattern = f"{re.escape(filename_prefix)}{re.escape(delimiter)}(\\d{{{filename_number_padding}}})"
        existing_counters = [int(re.search(pattern, filename).group(1)) for filename in os.listdir(output_path) if re.match(pattern, os.path.basename(filename))]
        existing_counters.sort(reverse=True)

        if existing_counters:
            counter = existing_counters[0] + 1
        else:
            counter = 1

        file_extension = '.' + extension
        if file_extension not in ALLOWED_EXT:
            # print(f"The extension `{extension}` is not valid. The valid formats are: {', '.join(sorted(ALLOWED_EXT))}")
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

                exif_metadata_A11 = f"""{image_metadata['positive']}
Negative prompt: {image_metadata['negative']}
Steps: {str(image_metadata['steps'])}, Sampler: {image_metadata['sampler_name'] + ' ' + image_metadata['scheduler_name']}, CFG scale: {str(image_metadata['cfg_scale'])}, Seed: {str(image_metadata['seed'])}, Size: {str(image_metadata['width'])}x{str(image_metadata['height'])}, Model hash: {image_metadata['model_hash']}, Model: {image_metadata['model_name']}, VAE: {image_metadata['vae_name']}"""

                exif_metadata_json = image_metadata

                if extension == 'png':
                    img.save(output_file, pnginfo=metadata, optimize=True)
                elif extension == 'webp':
                    img.save(output_file, quality=quality, exif=metadata)
                else:
                    img.save(output_file, quality=quality, optimize=True)
                    if image_embed_exif == True:
                        metadata = pyexiv2.Image(output_file)
                        metadata.modify_exif({'Exif.Photo.UserComment': 'charset=Unicode ' + exif_metadata_A11})
                        metadata.modify_exif({'Exif.Image.ImageDescription': json.dumps(exif_metadata_json)})
                        print(f"Image file saved with exif: {output_file}")
                    else:
                        if extension == 'webp':
                            img.save(output_file, quality=quality, exif=metadata)
                        else:
                            img.save(output_file, quality=quality, optimize=True)
                            print(f"Image file saved without exif: {output_file}")

                if save_mata_to_json:
                    jsonfile = os.path.splitext(output_file)[0] + '.json'
                    with open(jsonfile, 'w', encoding='utf-8') as jf:
                        json.dump(exif_metadata_json, jf, ensure_ascii=False, indent=4)

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
            '[time]': str(time.time()).replace('.', '_')
        }
        if '.' in self.tokens['[time]']: self.tokens['[time]'] = self.tokens['[time]'].split('.')[0]

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

class AnyType(str):
  def __ne__(self, __value: object) -> bool:
    return False

any = AnyType("*")

class PrimereAnyOutput:
  RETURN_TYPES = ()
  FUNCTION = "show_output"
  OUTPUT_NODE = True
  CATEGORY = TREE_OUTPUTS

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "input": (any, {}),
      },
    }

  def show_output(self, input = None):
    value = 'None'
    if input is not None:
      try:
        value = json.dumps(input, indent=4)
      except Exception:
        try:
          value = str(input)
        except Exception:
          value = 'Input data exists, but could not be serialized.'

    return {"ui": {"text": (value.strip( '"'),)}}