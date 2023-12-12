import nodes
from custom_nodes.ComfyUI_Primere_Nodes.components.tree import TREE_VISUALS
from custom_nodes.ComfyUI_Primere_Nodes.components.tree import PRIMERE_ROOT
import folder_paths
from custom_nodes.ComfyUI_Primere_Nodes.components import utility
import comfy.sd
import comfy.utils
import os
import random

class PrimereVisualCKPT:
    RETURN_TYPES = ("CHECKPOINT_NAME", "STRING", "MODEL_KEYWORD")
    RETURN_NAMES = ("MODEL_NAME", "MODEL_VERSION", "MODEL_KEYWORD")
    FUNCTION = "load_ckpt_visual_list"
    CATEGORY = TREE_VISUALS

    def __init__(self):
        self.chkp_loader = nodes.CheckpointLoaderSimple()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": (folder_paths.get_filename_list("checkpoints"),),
                "show_modal": ("BOOLEAN", {"default": True}),
                "show_hidden": ("BOOLEAN", {"default": True}),
                "use_model_keyword": ("BOOLEAN", {"default": False}),
                "model_keyword_placement": (["First", "Last"], {"default": "Last"}),
                "model_keyword_selection": (["Select in order", "Random select"], {"default": "Select in order"}),
                "model_keywords_num": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1}),
                "model_keyword_weight": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.1}),
            },
        }

    def load_ckpt_visual_list(self, base_model, show_hidden, show_modal, use_model_keyword, model_keyword_placement, model_keyword_selection, model_keywords_num, model_keyword_weight):
        LOADED_CHECKPOINT = self.chkp_loader.load_checkpoint(base_model)
        model_version = utility.getCheckpointVersion(LOADED_CHECKPOINT[0])
        model_keyword = [None, None]

        if use_model_keyword == True:
            ckpt_path = folder_paths.get_full_path("checkpoints", base_model)
            ModelKvHash = utility.get_model_hash(ckpt_path)
            if ModelKvHash is not None:
                KEYWORD_PATH = os.path.join(PRIMERE_ROOT, 'front_end', 'keywords', 'model-keyword.txt')
                keywords = utility.get_model_keywords(KEYWORD_PATH, ModelKvHash, base_model)

                if keywords is not None:
                    if keywords.find('|') > 1:
                        keyword_list = keywords.split("|")
                        if (len(keyword_list) > 0):
                            keyword_qty = len(keyword_list)
                            if (model_keywords_num > keyword_qty):
                                model_keywords_num = keyword_qty
                            if model_keyword_selection == 'Select in order':
                                list_of_keyword_items = keyword_list[:model_keywords_num]
                            else:
                                list_of_keyword_items = random.sample(keyword_list, model_keywords_num)
                            keywords = ", ".join(list_of_keyword_items)

                    if (model_keyword_weight != 1):
                        keywords = '(' + keywords + ':' + str(model_keyword_weight) + ')'

                    model_keyword = [keywords, model_keyword_placement]

        return (base_model, model_version, model_keyword)

class PrimereVisualLORA:
    RETURN_TYPES = ("MODEL", "CLIP", "LORA_STACK", "MODEL_KEYWORD")
    RETURN_NAMES = ("MODEL", "CLIP", "LORA_STACK", "LORA_KEYWORD")
    FUNCTION = "visual_lora_stacker"
    CATEGORY = TREE_VISUALS
    LORASCOUNT = 6

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),

                "stack_version": (["SD", "SDXL"], {"default": "SD"}),
                "show_modal": ("BOOLEAN", {"default": True}),
                "show_hidden": ("BOOLEAN", {"default": True}),
                "use_only_model_weight": ("BOOLEAN", {"default": True}),

                "use_lora_1": ("BOOLEAN", {"default": False}),
                "lora_1": (folder_paths.get_filename_list("loras"),),
                "lora_1_model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_1_clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_lora_2": ("BOOLEAN", {"default": False}),
                "lora_2": (folder_paths.get_filename_list("loras"),),
                "lora_2_model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_2_clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_lora_3": ("BOOLEAN", {"default": False}),
                "lora_3": (folder_paths.get_filename_list("loras"),),
                "lora_3_model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_3_clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_lora_4": ("BOOLEAN", {"default": False}),
                "lora_4": (folder_paths.get_filename_list("loras"),),
                "lora_4_model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_4_clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_lora_5": ("BOOLEAN", {"default": False}),
                "lora_5": (folder_paths.get_filename_list("loras"),),
                "lora_5_model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_5_clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_lora_6": ("BOOLEAN", {"default": False}),
                "lora_6": (folder_paths.get_filename_list("loras"),),
                "lora_6_model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_6_clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_lora_keyword": ("BOOLEAN", {"default": False}),
                "lora_keyword_placement": (["First", "Last"], {"default": "Last"}),
                "lora_keyword_selection": (["Select in order", "Random select"], {"default": "Select in order"}),
                "lora_keywords_num": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1}),
                "lora_keyword_weight": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.1}),
            },
        }

    def visual_lora_stacker(self, model, clip, use_only_model_weight, use_lora_keyword, lora_keyword_placement, lora_keyword_selection, lora_keywords_num, lora_keyword_weight, stack_version = 'SD', model_version = "BaseModel_1024", **kwargs):
        model_keyword = [None, None]

        if model_version == 'SDXL_2048' and stack_version == 'SD':
            return (model, clip, [], model_keyword)

        if model_version != 'SDXL_2048' and stack_version == 'SDXL':
            return (model, clip, [], model_keyword)

        loras = [kwargs.get(f"lora_{i}") for i in range(1, self.LORASCOUNT + 1)]
        model_weight = [kwargs.get(f"lora_{i}_model_weight") for i in range(1, self.LORASCOUNT + 1)]
        if use_only_model_weight == True:
            clip_weight =[kwargs.get(f"lora_{i}_model_weight") for i in range(1, self.LORASCOUNT + 1)]
        else:
            clip_weight =[kwargs.get(f"lora_{i}_clip_weight") for i in range(1, self.LORASCOUNT + 1)]

        uses = [kwargs.get(f"use_lora_{i}") for i in range(1, self.LORASCOUNT + 1)]
        lora_stack = [(lora_name, lora_model_weight, lora_clip_weight) for lora_name, lora_model_weight, lora_clip_weight, lora_uses in zip(loras, model_weight, clip_weight, uses) if lora_uses == True]

        lora_params = list()
        if lora_stack and len(lora_stack) > 0:
            lora_params.extend(lora_stack)
        else:
            return (model, clip, lora_stack, model_keyword)

        model_lora = model
        clip_lora = clip
        list_of_keyword_items = []
        lora_keywords_num_set = lora_keywords_num

        for tup in lora_params:
            lora_name, strength_model, strength_clip = tup

            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model_lora, clip_lora, lora, strength_model, strength_clip)

            if use_lora_keyword == True:
                ModelKvHash = utility.get_model_hash(lora_path)
                if ModelKvHash is not None:
                    KEYWORD_PATH = os.path.join(PRIMERE_ROOT, 'front_end', 'keywords', 'lora-keyword.txt')
                    keywords = utility.get_model_keywords(KEYWORD_PATH, ModelKvHash, lora_name)
                    if keywords is not None and keywords != "":
                        if keywords.find('|') > 1:
                            keyword_list = [word.strip() for word in keywords.split('|')]
                            keyword_list = list(filter(None, keyword_list))
                            if (len(keyword_list) > 0):
                                lora_keywords_num = lora_keywords_num_set
                                keyword_qty = len(keyword_list)
                                if (lora_keywords_num > keyword_qty):
                                    lora_keywords_num = keyword_qty
                                if lora_keyword_selection == 'Select in order':
                                    list_of_keyword_items.extend(keyword_list[:lora_keywords_num])
                                else:
                                    list_of_keyword_items.extend(random.sample(keyword_list, lora_keywords_num))
                        else:
                            list_of_keyword_items.append(keywords)

        if len(list_of_keyword_items) > 0:
            if lora_keyword_selection != 'Select in order':
                random.shuffle(list_of_keyword_items)

            list_of_keyword_items = list(set(list_of_keyword_items))
            keywords = ", ".join(list_of_keyword_items)

            if (lora_keyword_weight != 1):
                keywords = '(' + keywords + ':' + str(lora_keyword_weight) + ')'

            model_keyword = [keywords, lora_keyword_placement]

        return (model_lora, clip_lora, lora_stack, model_keyword)