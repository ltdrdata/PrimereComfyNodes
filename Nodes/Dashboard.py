from custom_nodes.ComfyUI_Primere_Nodes.components.tree import TREE_DASHBOARD
import comfy.samplers
import folder_paths
import nodes

class PrimereSamplers:
    CATEGORY = TREE_DASHBOARD
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS)
    RETURN_NAMES = ("sampler_name", "scheduler_name")
    FUNCTION = "get_sampler"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler_name": (comfy.samplers.KSampler.SCHEDULERS,)
            }
        }

    def get_sampler(self, sampler_name, scheduler_name):
        return sampler_name, scheduler_name


class PrimereVAE:
    RETURN_TYPES = ("VAE_NAME",)
    RETURN_NAMES = ("vae_name",)
    FUNCTION = "load_vae_list"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_model": (folder_paths.get_filename_list("vae"),)
            },
        }

    def load_vae_list(self, vae_model):
        return vae_model,

class PrimereCKPT:
    RETURN_TYPES = ("CHECKPOINT_NAME", )
    RETURN_NAMES = ("ckpt_name", )
    FUNCTION = "load_ckpt_list"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_model": (folder_paths.get_filename_list("checkpoints"),)
            },
        }

    def load_ckpt_list(self, base_model):
        return base_model,

class PrimereVAELoader:
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("VAE",)
    FUNCTION = "load_primere_vae"
    CATEGORY = TREE_DASHBOARD

    def __init__(self):
        self.vae_loader = nodes.VAELoader()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": ("VAE_NAME",)
            },
        }

    def load_primere_vae(self, vae_name, ):
        return self.vae_loader.load_vae(vae_name)

class PrimereCKPTLoader:
    RETURN_TYPES = ("MODEL", "CLIP", "VAE",)
    RETURN_NAMES = ("MODEL", "CLIP", "VAE",)
    FUNCTION = "load_primere_ckpt"
    CATEGORY = TREE_DASHBOARD

    def __init__(self):
        self.chkp_loader = nodes.CheckpointLoaderSimple()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ckpt_name": ("CHECKPOINT_NAME",)
            },
        }

    def load_primere_ckpt(self, ckpt_name, ):
        return self.chkp_loader.load_checkpoint(ckpt_name)

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

class PrimerePromptSwitch:
    any_typ = AnyType("*")

    # RETURN_TYPES = (any_typ, any_typ, "STRING", "INT")
    RETURN_TYPES = (any_typ, "STRING", "INT")
    # RETURN_NAMES = ("selected_pos", "selected_neg", "selected_label", "selected_index")
    RETURN_NAMES = ("selected_pos", "selected_label", "selected_index")
    FUNCTION = "promptswitch"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(cls):
        any_typ = AnyType("*")

        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
            },
            "optional": {
                "prompt_p1": (any_typ,),
                # "prompt_n1": (any_typ,),
            },
        }

    def promptswitch(self, *args, **kwargs):
        selected_index = int(kwargs['select'])
        input_namep = f"prompt_p{selected_index}"
        # input_namen = f"prompt_n{selected_index}"
        selected_label = input_namep

        if input_namep in kwargs:
            # return (kwargs[input_namep], kwargs[input_namen], selected_label, selected_index)
            return (kwargs[input_namep], selected_label, selected_index)
        else:
            print(f"ImpactSwitch: invalid select index (ignored)")
            # return (None, "", "", selected_index)
            return (None, "", selected_index)