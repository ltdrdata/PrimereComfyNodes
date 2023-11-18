import os
from .utils import comfy_dir
from .utils import here

__version__ = "0.1.0"

comfy_frontend = comfy_dir/"web"/"extensions"
frontend_target = comfy_frontend/"Primere"

if frontend_target.exists() == False:
    # print(f"Primere front-end folder found at {frontend_target}")
    # if not os.path.islink(frontend_target.as_posix()):
    # print(f"Primere front-end folder at {frontend_target} is not a symlink, if updating please delete it before")

# elif comfy_frontend.exists():
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
# else:
#    print(f"Comfy root probably not found automatically, please copy the folder {frontend_target} manually in the web/extensions folder of ComfyUI")

import custom_nodes.ComfyUI_Primere_Nodes.Nodes.Dashboard as Dashboard
import custom_nodes.ComfyUI_Primere_Nodes.Nodes.Inputs as Inputs
import custom_nodes.ComfyUI_Primere_Nodes.Nodes.Styles as Styles
import custom_nodes.ComfyUI_Primere_Nodes.Nodes.Outputs as Outputs

NODE_CLASS_MAPPINGS = {
    "PrimereSamplers": Dashboard.PrimereSamplers,
    "PrimereVAE": Dashboard.PrimereVAE,
    "PrimereCKPT": Dashboard.PrimereCKPT,
    "PrimereVAELoader": Dashboard.PrimereVAELoader,
    "PrimereCKPTLoader": Dashboard.PrimereCKPTLoader,
    "PrimerePromptSwitch": Dashboard.PrimerePromptSwitch,
    "PrimereSeed": Dashboard.PrimereSeed,
    "PrimereLatentNoise": Dashboard.PrimereFractalLatent,
    "PrimereCLIPEncoder": Dashboard.PrimereCLIP,
    "PrimereResolution": Dashboard.PrimereResolution,
    "PrimereStepsCfg": Dashboard.PrimereStepsCfg,
    "PrimereClearPrompt": Dashboard.PrimereClearPrompt,
    "PrimereLCMSelector": Dashboard.PrimereLCMSelector,
    "PrimereResolutionMultiplier": Dashboard.PrimereResolutionMultiplier,

    "PrimerePrompt": Inputs.PrimereDoublePrompt,
    "PrimereStyleLoader": Inputs.PrimereStyleLoader,
    "PrimereDynamicParser": Inputs.PrimereDynParser,
    "PrimereVAESelector": Inputs.PrimereVAESelector,
    "PrimereMetaRead": Inputs.PrimereMetaRead,
    "PrimereEmbeddingHandler": Inputs.PrimereEmbeddingHandler,

    "PrimereMetaSave": Outputs.PrimereMetaSave,
    "PrimereAnyOutput": Outputs.PrimereAnyOutput,
    "PrimereTextOutput": Outputs.PrimereTextOutput,

    "PrimereStylePile": Styles.PrimereStylePile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrimereSamplers": "Primere Sampler Selector",
    "PrimereVAE": "Primere VAE Selector",
    "PrimereCKPT": "Primere CKPT Selector",
    "PrimereVAELoader": "Primere VAE Loader",
    "PrimereCKPTLoader": "Primere CKPT Loader",
    "PrimerePromptSwitch": "Primere Prompt Switch",
    "PrimereSeed": 'Primere Seed',
    "PrimereLatentNoise": "Primere Noise Latent",
    "PrimereCLIPEncoder": "Primere Prompt Encoder",
    "PrimereResolution": "Primere Resolution",
    "PrimereStepsCfg": "Primere Steps & Cfg",
    "PrimereClearPrompt": "Primere Prompt Cleaner",
    "PrimereLCMSelector": "Primere LCM selector",
    "PrimereResolutionMultiplier": "Primere Resolution Multiplier",

    "PrimerePrompt": "Primere Prompt",
    "PrimereStyleLoader": "Primere Styles",
    "PrimereDynamicParser": "Primere Dynamic",
    "PrimereVAESelector": "Primere VAE Selector",
    "PrimereMetaRead": "Primere Exif Reader",
    "PrimereEmbeddingHandler": "Primere Embedding Handler",

    "PrimereMetaSave": "Primere Image Meta Saver",
    "PrimereAnyOutput": "Primere Any Debug",
    "PrimereTextOutput": "Primere Text Ouput",

    "PrimereStylePile": "Primere Style Pile",
}