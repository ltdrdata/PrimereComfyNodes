import custom_nodes.ComfyUI_Primere_Nodes.Nodes.IO.ImageMeta as PriIO

import shutil
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

def setup_js():
    js_dest_path = os.path.join(comfy_path, "web", "extensions", "primere")
    if not os.path.exists(js_dest_path):
        os.makedirs(js_dest_path)

    js_src_path = os.path.join(tk_nodes_path, "javascript", "primere_metadata.js")
    shutil.copy(js_src_path, js_dest_path)

setup_js()
