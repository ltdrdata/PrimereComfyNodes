import piexif.helper
import json
import re

file = 'z:\\AIPICS\\A11\\BestAll_landscapeRealistic_v11\\2023-06-Week24\\00008-20230617204208-1954514607.jpg'

exif_dict = piexif.load(file)
user_comment = piexif.helper.UserComment.load(exif_dict["Exif"][piexif.ExifIFD.UserComment])

'''(beautiful sexy 30 years old (American black:1.4) model woman:1.2), (standing in the green park:1.3), near blue lake, full body outdoor color professional model photography, soft natural lights, happy mood, kissing mouth, (wear white, leather, jacket:1.6), (wear maroon, luxury, silk, underwear:1.6), (wear leather, long, black, trousers:1.6), (long, black, straight, hair:1.6), small cleavage, natural tits, (covered bra:1.2), snow, (Winter:1.4), blury background, bokeh, (film grain:1.6), (detailed face:1.6), (detailed skin:1.6), (funny yellow hat:0.7)
Negative prompt: porn, sex, nude, nudity, child, childish, Asian-Less-Neg, FastNegativeEmbedding, negative_hand, draw, illustration, telephoto, earrings, necklace, nsfw
Steps: 80, Sampler: DPM++ SDE Karras, CFG scale: 11, Seed: 2892028926, Face restoration: GFPGAN, Size: 512x768, Model hash: 62e2993f1d, Model: BestOf_epicrealism_pureEvolution, CFG Rescale φ: 0, NPW_weight: 1.2, Version: v1.3.2'''
# print(user_comment)

EXIF_LABELS = {
    "positive_g":'Positive prompt',
    "negative_g":'Negative prompt',
    "steps":'Steps',
    "sampler":'Sampler',
    "seed":'Seed',
    "variation_seed":'Variation seed',
    "variation_seed_strength":'Variation seed strength',
    "size_string":'Size',
    "model_hash":'Model hash',
    'model':'Model',
    "vae_hash":'VAE hash',
    "vae":'VAE',
    "lora_hashes":'Lora hashes',
    "cfg_scale":'CFG scale',
    "cfg_rescale":'CFG Rescale φ',
    "cfg_rescale_phi":'CFG Rescale phi',
    "rp_active":'RP Active',
    "rp_divide_mode":'RP Divide mode',
    "rp_matrix_submode":'RP Matrix submode',
    "rp_mask_submode":'RP Mask submode',
    "rp_prompt_submode":'RP Prompt submode',
    "rp_calc_mode":'RP Calc Mode',
    "rp_ratios":'RP Ratios',
    "rp_base_ratios":'RP Base Ratios',
    "rp_use_base":'RP Use Base',
    "rp_use_common":'RP Use Common',
    "rp_use_ncommon":'RP Use Ncommon',
    "rp_change_and":'RP Change AND',
    "rp_lora_neg_te_ratios":'RP LoRA Neg Te Ratios',
    "rp_lora_neg_u_ratios":'RP LoRA Neg U Ratios',
    "rp_threshold":'RP threshold',
    "npw_weight":'NPW_weight',
    "antiburn":'AntiBurn',
    "version":'Version',
    "template":'Template',
    "negative_template":'Negative Template',
    "face_restoration":'Face restoration',
    "postprocess_upscaler":'Postprocess upscaler',
    "postprocess_upscale_by":'Postprocess upscale by'
}

LABEL_END = ['\n', ',']
POSITION_START = 0
STRIP_FROM_VALUE = ' ";\n'
FORCE_STRING = ['model_hash', 'vae_hash', 'lora_hashes']
FORCE_FLOAT = ['cfg_scale', 'cfg_rescale', 'cfg_rescale_phi', 'npw_weight']

# FIRST_ROW = user_comment.split('\n', 1)[0]
user_comment = 'Positive prompt: ' + user_comment

SORTED_BY_STRING = dict(sorted(EXIF_LABELS.items(), key=lambda pos: user_comment.find(pos[1] + ':')))
SORTED_KEYLIST = list(SORTED_BY_STRING.keys())
FINAL_DICT = {}
FLOAT_PATTERN = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'

print('\n=================== DATA ===========================')
for LABEL_KEY, LABEL in SORTED_BY_STRING.items():
    NextValue = '\n'
    RealLabel = LABEL + ':'
    CurrentKeyIndex = (SORTED_KEYLIST.index(LABEL_KEY))
    NextKeyIndex = CurrentKeyIndex + 1

    if len(SORTED_KEYLIST) > NextKeyIndex:
        NextKey = SORTED_KEYLIST[NextKeyIndex]
        NextValue = SORTED_BY_STRING[NextKey] + ':'

    if RealLabel in user_comment:
        LabelStart = user_comment.find(RealLabel)
        NextLabelStart = user_comment.find(NextValue)
        LabelLength = len(RealLabel)
        ValueStart = user_comment.find(user_comment[(LabelStart + LabelLength):NextLabelStart])
        ValueLength = len(user_comment[(LabelStart + LabelLength):NextLabelStart])
        ValueRaw = user_comment[(LabelStart + LabelLength):NextLabelStart]
        FirstMatch = next((x for x in LABEL_END if x in user_comment[(ValueStart + ValueLength - 2):(ValueStart + ValueLength + 1)]), False)

        if CurrentKeyIndex >= 2 and FirstMatch == ',':
            isUnknownValue = all(x in ValueRaw for x in [':', ','])
            if isUnknownValue:
                FirstMatchOfFaliled = ValueRaw.find(FirstMatch)
                NextLabelStart = ValueStart

        if FirstMatch:
            LabelEnd = user_comment.find(FirstMatch, NextLabelStart - 2)
            LabelValue = user_comment[(LabelStart + LabelLength):LabelEnd]
        else:
            LabelEnd = None
            if CurrentKeyIndex >= 2 and FirstMatch == '\n' or FirstMatch == False:
                badValue = user_comment[(LabelStart + LabelLength):LabelEnd]
                isUnknownValue = all(x in badValue for x in [':', '\n'])
                if isUnknownValue:
                    FirstMatchOfFaliled = badValue.find('\n')
                    LabelEnd = user_comment.find('\n', LabelStart + LabelLength + FirstMatchOfFaliled + 2)

            LabelValue = user_comment[(LabelStart + LabelLength):LabelEnd]

        LabelValue = LabelValue.replace('Count=', '').strip(STRIP_FROM_VALUE)
        if not LabelValue:
            LabelValue = None
        elif LabelValue == 'False':
            LabelValue = False
        elif LabelValue.isdigit():
            LabelValue = int(LabelValue)
        elif bool(re.match(FLOAT_PATTERN, LabelValue)):
            LabelValue = float(LabelValue)

        if LABEL_KEY in FORCE_STRING:
            LabelValue = str(LabelValue)

        if LABEL_KEY in FORCE_FLOAT:
            LabelValue = float(LabelValue)

        if LABEL_KEY == 'size_string':
            width, height = LabelValue.split("x")
            FINAL_DICT['width'] = int(width)
            FINAL_DICT['height'] = int(height)

        FINAL_DICT[LABEL_KEY] = LabelValue
        print(RealLabel + ' -> ' + LABEL_KEY + ': ' + str(LabelValue))

print('\n=================== DICT ===========================')
print(FINAL_DICT)
print('\n=================== JSON ===========================')
jsonOutput = json.dumps(FINAL_DICT, indent=4)
print(jsonOutput)