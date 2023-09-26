import custom_nodes.ComfyUI_Primere_Nodes.Nodes.IO.ImageMeta as PriIO
from custom_nodes.ComfyUI_Primere_Nodes.components.startup import symlink_primere_frontend
import folder_paths
import os

comfy_path = os.path.dirname(folder_paths.__file__)
tk_nodes_path = os.path.join(os.path.dirname(__file__))

NODE_CLASS_MAPPINGS = {
    "PrimereSamplers": PriIO.PrimereSamplers,
    "PrimereImageMetaSaver": PriIO.PrimereMetaSave,
    "PrimereImageMetaReader": PriIO.PrimereMetaRead
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrimereSamplers": "Primere Sampler Selector",
    "PrimereImageMetaSaver": "Primere Image Meta Saver",
    "PrimereImageMetaReader": "Primere Image Meta Reader"
}

symlink_primere_frontend("javascript", "Primere", tk_nodes_path)
