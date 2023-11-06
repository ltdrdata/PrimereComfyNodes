import math

SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".webp"]
STANDARD_SIDES = [64, 80, 96, 128, 144, 160, 192, 256, 320, 368, 400, 480, 512, 560, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048]

def merge_str_to_tuple(item1, item2):
    if not isinstance(item1, tuple):
        item1 = (item1,)
    if not isinstance(item2, tuple):
        item2 = (item2,)
    return item1 + item2

def merge_dict(dict1, dict2):
    dict3 = dict1.copy()
    for k, v in dict2.items():
        dict3[k] = merge_str_to_tuple(v, dict3[k]) if k in dict3 else v
    return dict3

def remove_quotes(string):
    return str(string).replace('"', "").replace("'", "")

def add_quotes(string):
    return '"' + str(string) + '"'

def calculate_dimensions(self, ratio: str, orientation: str, round_to_standard: bool, is_sdxl: int, default_sd: str, calculate_by_custom: bool, custom_side_a: float, custom_side_b: float):
    SD_1 = 512
    SD_2 = 768
    SDXL_1 = 1024
    DEFAULT_RES = SD_1

    if (default_sd == 'SD 2.x'):
        DEFAULT_RES = SD_2

    if (is_sdxl == 1):
        DEFAULT_RES = SDXL_1
    def calculate(ratio_1: float, ratio_2: float, side: int):
        FullPixels = side ** 2
        result_x = FullPixels / ratio_2
        result_y = result_x / ratio_1
        side_base = round(math.sqrt(result_y))
        side_a = round(ratio_1 * side_base)
        if round_to_standard == True:
            side_a = min(STANDARD_SIDES, key=lambda x: abs(side_a - x))
        side_b = round(FullPixels / side_a)
        return sorted([side_a, side_b], reverse=True)

    if (calculate_by_custom == True and isinstance(custom_side_a, (int, float)) and isinstance(custom_side_b, (int, float)) and custom_side_a >= 1 and custom_side_b >= 1):
        ratio_x = custom_side_a
        ratio_y = custom_side_b
    else:
        RatioLabel = self.ratioNames[ratio]
        ratio_x = self.sd_ratios[RatioLabel]['side_x']
        ratio_y = self.sd_ratios[RatioLabel]['side_y']

    dimensions = calculate(ratio_x, ratio_y, DEFAULT_RES)
    if (orientation == 'Vertical'):
        dimensions = sorted(dimensions)

    dimension_x = dimensions[0]
    dimension_y = dimensions[1]
    return (dimension_x, dimension_y,)

def clear_prompt(NETWORK_START, NETWORK_END, promptstring):
    promptstring_temp = promptstring

    for LABEL in NETWORK_START:
        if LABEL in promptstring:
            LabelStartIndexes = [n for n in range(len(promptstring)) if promptstring.find(LABEL, n) == n]

            for LabelStartIndex in LabelStartIndexes:
                Matches = []
                for endString in NETWORK_END:
                    Match = promptstring.find(endString, (LabelStartIndex + 1))
                    if (Match > 0):
                        Matches.append(Match)

                LabelEndIndex = sorted(Matches)[0]
                MatchedString = promptstring[LabelStartIndex:(LabelEndIndex + 1)]
                if len(MatchedString) > 0:
                    if '<' in MatchedString:
                        endString = '>'
                        Match = promptstring.find(endString, (LabelStartIndex + 1))
                        if (Match > 0):
                            LabelEndIndex = Match
                            MatchedString = promptstring[LabelStartIndex:(LabelEndIndex + 1)]
                            promptstring_temp = promptstring_temp.replace(MatchedString, "")

                    if '{' in MatchedString:
                        endString = '}'
                        Match = promptstring.find(endString, (LabelStartIndex + 1))
                        if (Match > 0):
                            LabelEndIndex = Match
                            MatchedString = promptstring[LabelStartIndex:(LabelEndIndex + 1)]
                            promptstring_temp = promptstring_temp.replace(MatchedString, "")

                    if ')' in MatchedString:
                        MatchedString = promptstring[(LabelStartIndex - 1):(LabelEndIndex + 1)]
                        promptstring_temp = promptstring_temp.replace(MatchedString, "")

                    promptstring_temp = promptstring_temp.replace(MatchedString, "")

    return promptstring_temp.replace('()', '').replace(' , ,', ',').replace('||', '').replace('{,', '').replace('  ', ' ').replace(', ,', ',').strip(', ')