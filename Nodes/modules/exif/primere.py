from ..exif.base_format import BaseFormat
from custom_nodes.ComfyUI_Primere_Nodes.components.utility import remove_quotes

class Primere(BaseFormat):
    def __init__(self, info: dict = None, raw: str = ""):
        super().__init__(info, raw)
        self._pri_format()

    def _pri_format(self):
        data_json = self._info

        self._parameter["seed"] = data_json['seed']
        self._parameter["cfg"] = data_json['cfg_scale']
        self._parameter["steps"] = data_json['steps']
        self._parameter["model"] = data_json['model_name']
        self._parameter["sampler"] = data_json['sampler_name']
        self._parameter["size"] = (
            str(data_json.get("original_width")) + "x" + str(data_json.get("original_height"))
        )

        self._positive = data_json['positive_g']
        self._negative = data_json['negative_g']
        self._raw = "\n".join([self._positive, self._negative, str(data_json)])
        self._setting = remove_quotes(str(data_json)[1:-1])
