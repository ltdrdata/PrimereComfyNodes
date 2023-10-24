import os
from .utils import comfy_dir
from .utils import here

__version__ = "0.1.0"

comfy_frontend = comfy_dir/"web"/"extensions"
frontend_target = comfy_frontend/"Primere"

if frontend_target.exists():
    print(f"Primere front-end folder found at {frontend_target}")
    # if not os.path.islink(frontend_target.as_posix()):
    #    print(f"Primere front-end folder at {frontend_target} is not a symlink, if updating please delete it before")

elif comfy_frontend.exists():
    frontend_source = here/"front_end"
    src = frontend_source.as_posix()
    dst = frontend_target.as_posix()

    try:
        if os.name == "nt":
            import _winapi
            _winapi.CreateJunction(src, dst)
        else:
            os.symlink(frontend_source.as_posix(), frontend_target.as_posix())
        print(f"Primere front-end folder symlinked to {frontend_target}")

    except OSError:
        print(f"Failed to create frint-end symlink to {frontend_target}, trying to copy it")
        try:
            import shutil
            shutil.copytree(frontend_source, frontend_target)
            print(f"Successfully copied {frontend_source} to {frontend_target}")
        except Exception as e:
            print(f"Failed to symlink and copy {frontend_source} to {frontend_target}. Please copy the folder manually.")
    except Exception as e:
        print(f"Failed to create symlink to {frontend_target}. Please copy the folder manually.")
else:
    print(f"Comfy root probably not found automatically, please copy the folder {frontend_target} manually in the web/extensions folder of ComfyUI")

import custom_nodes.ComfyUI_Primere_Nodes.Nodes.Dashboard as Dashboard
import custom_nodes.ComfyUI_Primere_Nodes.Nodes.Inputs as Inputs

NODE_CLASS_MAPPINGS = {
    "PrimereSamplers": Dashboard.PrimereSamplers,
    "PrimereVAE": Dashboard.PrimereVAE,
    "PrimereCKPT": Dashboard.PrimereCKPT,
    "PrimereVAELoader": Dashboard.PrimereVAELoader,
    "PrimereCKPTLoader": Dashboard.PrimereCKPTLoader,
    "PrimerePromptSwitch": Dashboard.PrimerePromptSwitch,

    "PrimerePrompt": Inputs.PrimereDoublePrompt,
    "PrimereStyleLoader": Inputs.PrimereStyleLoader,
    # "PrimereImageMetaSaver": PriIO.PrimereMetaSave,
    # "PrimereImageMetaReader": PriIO.PrimereMetaRead
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrimereSamplers": "Primere Sampler Selector",
    "PrimereVAE": "Primere VAE Selector",
    "PrimereCKPT": "Primere CKPT Selector",
    "PrimereVAELoader": "Primere VAE Loader",
    "PrimereCKPTLoader": "Primere CKPT Loader",
    "PrimerePromptSwitch": "Primere Prompt Switch",

    "PrimerePrompt": "Primere Prompt",
    "PrimereStyleLoader": "Primere Styles",
    # "PrimereImageMetaSaver": "Primere Image Meta Saver",
    # "PrimereImageMetaReader": "Primere Image Meta Reader"
}