from custom_nodes.ComfyUI_Primere_Nodes.components.tree import TREE_INPUTS
import os
import re
import folder_paths

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
        styles = {"Error loading styles.csv, check the console": ["", ""]}
        if not os.path.exists(styles_path):
            print(f"""Error. No styles.csv found. Put your styles.csv in the root directory of ComfyUI. Then press "Refresh".
                  Your current root directory is: {folder_paths.base_path}
            """)
            return styles
        try:
            with open(styles_path, "r", encoding="utf-8") as f:
                styles = [[x.replace('"', '').replace('\n', '') for x in re.split(',(?=(?:[^"]*"[^"]*")*[^"]*$)', line)]
                          for line in f.readlines()[1:]]
                styles = {x[0]: [x[1], x[2]] for x in styles}
        except Exception as e:
            print(f"""Error loading styles.csv. Make sure it is in the root directory of ComfyUI. Then press "Refresh".
                    Your current root directory is: {folder_paths.base_path}
                    Error: {e}
            """)
        return styles

    @classmethod
    def INPUT_TYPES(cls):
        cls.styles_csv = cls.load_styles_csv(os.path.join(folder_paths.base_path, "styles.csv"))
        return {
            "required": {
                "styles": (list(cls.styles_csv.keys()),),
            },

        }

    def load_csv(self, styles):
        return (self.styles_csv[styles][0], self.styles_csv[styles][1])