from custom_nodes.ComfyUI_Primere_Nodes.components.tree import TREE_INPUTS
from custom_nodes.ComfyUI_Primere_Nodes.components.tree import PRIMERE_ROOT
import os
import re
import folder_paths
from dynamicprompts.parser.parse import ParserConfig
from dynamicprompts.wildcards.wildcard_manager import WildcardManager
from dynamicprompts.generators import RandomPromptGenerator

class PrimereDoublePrompt:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "get_prompt"
    CATEGORY = TREE_INPUTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
        }

    def get_prompt(self, positive_prompt, negative_prompt):
        return positive_prompt, negative_prompt,


class PrimereStyleLoader:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "load_csv"
    CATEGORY = TREE_INPUTS

    @staticmethod
    def load_styles_csv(styles_path: str):
        STYLE_DIR = os.path.join(PRIMERE_ROOT, 'stylecsv')
        styles = {"Error loading styles.csv, check the console": ["", ""]}
        if not os.path.exists(styles_path):
            print(f"""Error. No styles.csv found. Put your styles.csv in the node directory then press "Refresh".
                  Your current node directory is: {STYLE_DIR}
            """)
            return styles
        try:
            with open(styles_path, "r", encoding="utf-8") as f:
                styles = [[x.replace('"', '').replace('\n', '') for x in re.split(',(?=(?:[^"]*"[^"]*")*[^"]*$)', line)]
                          for line in f.readlines()[1:]]
                styles = {x[0]: [x[1], x[2]] for x in styles}
        except Exception as e:
            print(f"""Error loading styles.csv. Make sure it is in the root directory of node. Then press "Refresh".
                    Your current node directory is: {STYLE_DIR}
                    Error: {e}
            """)
        return styles

    @classmethod
    def INPUT_TYPES(cls):
        STYLE_DIR = os.path.join(PRIMERE_ROOT, 'stylecsv')
        cls.styles_csv = cls.load_styles_csv(os.path.join(STYLE_DIR, "styles.csv"))
        return {
            "required": {
                "styles": (list(cls.styles_csv.keys()),),
            },

        }

    def load_csv(self, styles):
        return (self.styles_csv[styles][0], self.styles_csv[styles][1])

class PrimereDynParser:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "dyndecoder"
    CATEGORY = TREE_INPUTS

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dyn_prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
            }
        }

    def __init__(self):
        wildcard_dir = os.path.join(PRIMERE_ROOT, 'wildcards')
        self._wildcard_manager = WildcardManager(wildcard_dir)
        self._parser_config = ParserConfig(
            variant_start = "{",
            variant_end = "}",
            wildcard_wrap = "__"
        )

    def dyndecoder(self, dyn_prompt, seed):
        prompt_generator = RandomPromptGenerator(
            self._wildcard_manager,
            seed = seed,
            parser_config = self._parser_config,
            unlink_seed_from_prompt = False,
            ignore_whitespace = False
        )

        all_prompts = prompt_generator.generate(dyn_prompt, 1) or [""]
        prompt = all_prompts[0]

        return (prompt, )