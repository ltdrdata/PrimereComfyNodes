import nodes
from custom_nodes.ComfyUI_Primere_Nodes.components.tree import TREE_VISUALS
import folder_paths
from custom_nodes.ComfyUI_Primere_Nodes.components import utility

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