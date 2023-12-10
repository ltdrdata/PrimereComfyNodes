import nodes
from custom_nodes.ComfyUI_Primere_Nodes.components.tree import TREE_VISUALS
import folder_paths
from custom_nodes.ComfyUI_Primere_Nodes.components import utility
import comfy.sd
import comfy.utils

class PrimereVisualCKPT:
    RETURN_TYPES = ("CHECKPOINT_NAME", "STRING")
    RETURN_NAMES = ("MODEL_NAME", "MODEL_VERSION")
    FUNCTION = "load_ckpt_visual_list"
    CATEGORY = TREE_VISUALS

    def __init__(self):
        self.chkp_loader = nodes.CheckpointLoaderSimple()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_model": (folder_paths.get_filename_list("checkpoints"),),
                "show_modal": ("BOOLEAN", {"default": True}),
                "show_hidden": ("BOOLEAN", {"default": True}),
            },
        }

    def load_ckpt_visual_list(self, base_model, show_hidden, show_modal):
        LOADED_CHECKPOINT = self.chkp_loader.load_checkpoint(base_model)
        model_version = utility.getCheckpointVersion(LOADED_CHECKPOINT[0])

        return (base_model, model_version,)

class PrimereVisualLORA:
    RETURN_TYPES = ("MODEL", "CLIP", "LORA_STACK")
    RETURN_NAMES = ("MODEL", "CLIP", "LORA_STACK")
    FUNCTION = "visual_lora_stacker"
    CATEGORY = TREE_VISUALS
    LORASCOUNT = 6

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "show_modal": ("BOOLEAN", {"default": True}),
                "show_hidden": ("BOOLEAN", {"default": True}),
                "use_lora_1": ("BOOLEAN", {"default": False}),
                "lora_1": (folder_paths.get_filename_list("loras"),),
                "lora_1_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "use_lora_2": ("BOOLEAN", {"default": False}),
                "lora_2": (folder_paths.get_filename_list("loras"),),
                "lora_2_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "use_lora_3": ("BOOLEAN", {"default": False}),
                "lora_3": (folder_paths.get_filename_list("loras"),),
                "lora_3_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "use_lora_4": ("BOOLEAN", {"default": False}),
                "lora_4": (folder_paths.get_filename_list("loras"),),
                "lora_4_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "use_lora_5": ("BOOLEAN", {"default": False}),
                "lora_5": (folder_paths.get_filename_list("loras"),),
                "lora_5_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "use_lora_6": ("BOOLEAN", {"default": False}),
                "lora_6": (folder_paths.get_filename_list("loras"),),
                "lora_6_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
        }

    def visual_lora_stacker(self, model, clip, **kwargs):
        loras = [kwargs.get(f"lora_{i}") for i in range(1, self.LORASCOUNT + 1)]
        weights = [kwargs.get(f"lora_{i}_weight") for i in range(1, self.LORASCOUNT + 1)]
        uses = [kwargs.get(f"use_lora_{i}") for i in range(1, self.LORASCOUNT + 1)]
        lora_stack = [(lora_name, lora_weight, lora_weight) for lora_name, lora_weight, lora_uses in zip(loras, weights, uses) if lora_uses == True]

        lora_params = list()
        if lora_stack and len(lora_stack) > 0:
            lora_params.extend(lora_stack)
        else:
            return (model, clip, lora_stack)

        model_lora = model
        clip_lora = clip

        for tup in lora_params:
            lora_name, strength_model, strength_clip = tup

            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model_lora, clip_lora, lora, strength_model, strength_clip)

        return (model_lora, clip_lora, lora_stack,)