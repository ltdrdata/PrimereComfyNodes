from ..format.base_format import BaseFormat
from ..utility import add_quotes

PROMPT_MAPPING = {
    # "Model":                    ("sd_model",            True),
    # "prompt",
    # "negative_prompt",
    "Seed": ("seed", False),
    "Variation seed strength": ("subseed_strength", False),
    # "seed_resize_from_h",
    # "seed_resize_from_w",
    "Sampler": ("sampler_name", True),
    "Steps": ("steps", False),
    "CFG scale": ("cfg_scale", False),
    # "width",
    # "height",
    "Face restoration": ("restore_faces", False),
}


class Primere(BaseFormat):
    def __init__(self, info: dict = None, raw: str = ""):
        super().__init__(info, raw)
        if not self._raw:
            self._raw = self._info.get("parameters")
        self._sd_format()

    def _sd_format(self):
        return 1
