import os
import difflib
import hashlib
import comfy.samplers

NODE_FILE = os.path.abspath(__file__)
NODE_ROOT = os.path.dirname(NODE_FILE)
print(NODE_ROOT)


def model_hash(filename):
  hash_sha256 = hashlib.sha256()
  blksize = 1024 * 1024

  with open(filename, "rb") as f:
    for chunk in iter(lambda: f.read(blksize), b""):
      hash_sha256.update(chunk)

  return hash_sha256.hexdigest()[0:10]

allcheckpoints = ['.BestNSX\\Invisidubious_Centerfold.safetensors', '.BestNSX\\Realistic_Vision_V2.0.safetensors', '.BestNSX\\URPM_v13.safetensors', '.BestNSX\\analogMadness_v40.safetensors', '.BestNSX\\purepornplus10_2.ckpt', '.BestNSX\\uberRealisticPornMergeURPM_urpmCE3Alpha.safetensors', '.BestNSX\\universalPhotorealistic_ver01.safetensors', '.NSX\\Chilloutmix.safetensors', '.NSX\\DucHaitenDarkside.safetensors', '.NSX\\SD_MagnumOpus.safetensors', '.NSX\\URPM_Dreamy_pfg.safetensors', '.NSX\\consistentFactor_v3.2.safetensors', '.NSX\\eroticVision_v3.safetensors', '.NSX\\grapelikedreamfruit.ckpt', '.NSX\\kitchensink2nsfw.safetensors', '.NSX\\perfectdeliberate.safetensors', '.NSX\\pfg111.safetensors', '.NSX\\photomerge_v1.2-inpainting.safetensors', '.NSX\\photomerge_v1.2.safetensors', '.NSX\\pureRealisticPornPRP_v10.safetensors', '.NSX\\purepornplusMerge_purepornplus10.ckpt', '.NSX\\ryanblend_v60.safetensors', '.NSX\\wondermix_V2.safetensors', '768-v-ema.ckpt', 'BestAll\\DeliberateForInvoke_v08.ckpt', 'BestAll\\Deliberate_v2.safetensors', 'BestAll\\Fruit_Fusion.safetensors', 'BestAll\\Hardblend.safetensors', 'BestAll\\Realistic_Vision_V1.4-pruned-fp16.safetensors', 'BestAll\\SD_1_5_model.ckpt', 'BestAll\\aatrok.safetensors', 'BestAll\\artiusV2.1.safetensors', 'BestAll\\consistentColorful_v104.safetensors', 'BestAll\\dvarchExterior.safetensors', 'BestAll\\epicrealism_pureEvolution.safetensors', 'BestAll\\hyper_v2.safetensors', 'BestAll\\icbinp.safetensors', 'BestAll\\icbinp_final.safetensors', 'BestAll\\landscapeRealistic_v11.safetensors', 'BestAll\\photon_v1.safetensors', 'BestAll\\realisticVisionV30.safetensors', 'BestAll\\realisticVisionV40.safetensors', 'BestAll\\realisticVision_v5.1.safetensors', 'BestAll\\rinamix2.safetensors', 'BestPhoto\\FaeTastic.safetensors', 'BestPhoto\\UnstablePhotoRealv.5.ckpt', 'BestPhoto\\canvers-photoreal-v3.2-fp16-no-ema.safetensors', 'BestPhoto\\cyberrealistic_v20.safetensors', 'BestSci-fi\\LifeLikeDiffusion_V20.safetensors', 'BestSci-fi\\RoboDiffusion_v1.ckpt', 'BestSci-fi\\UnrealEngine5B.ckpt', 'BestSci-fi\\aresMix_v02.safetensors', 'BestSci-fi\\clarity_19.safetensors', 'BestSci-fi\\deepSpaceDiffusion_v1.ckpt', 'BestSci-fi\\experience_70.safetensors', 'BestSci-fi\\experience_realistic2.safetensors', 'BestSci-fi\\level4.safetensors', 'BestSci-fi\\lyriel_v16.safetensors', 'BestSci-fi\\monsterStormExtrem_v10.safetensors', 'BestSci-fi\\providence_2110Prelude.safetensors', 'BestSci-fi\\realbiter_v10.safetensors', 'BestSci-fi\\rinamix3.safetensors', 'Character\\FantasyMix_v1.safetensors', 'Character\\VividMix_Elldreth.safetensors', 'Character\\animalHumanHybrids_v10.safetensors', 'Character\\avalonTruvision_v2.safetensors', 'Character\\sdPhotorealV2.8.safetensors', 'Character\\stablegramUSEuropean_v24.safetensors', 'Character\\verisimilitude_V14.safetensors', 'Design\\architectureInterior_v80.safetensors', 'Design\\dvarchInterior.safetensors', 'Design\\dvarchMultiPrompt.ckpt', 'Design\\gdmLuxuryModernInterior_v2GDMHighEnd.ckpt', 'Design\\interiordesignsuperm_v2.safetensors', 'Design\\knollingcase_v1.ckpt', 'Design\\productDesign_15b.safetensors', 'Design\\productDesign_minimalism.safetensors', 'Midjourney\\Midjourney-Shatter.ckpt', 'Midjourney\\Midjourney-Splatter-Art.ckpt', 'Midjourney\\Midjourney-V3.ckpt', 'Midjourney\\Midjourney-V4-PaintArt.ckpt', 'Midjourney\\Midjourney_16500.ckpt', 'Midjourney\\Midjourney_Graffiti.ckpt', 'Midjourney\\Midjourney_Papercut.ckpt', 'Midjourney\\Midjourney_V4.1.safetensors', 'Midjourney\\Midjourney_V4_finetune.ckpt', 'Photo\\Dreamlike-Photoreal-2.0.safetensors', 'Photo\\PhotoS4w3d0ffBlend_v2.1.safetensors', 'Photo\\Photography-and-Landscapes.safetensors', 'Photo\\ProtogenX34Photorealism.safetensors', 'Photo\\ProtogenX58RebuiltScifi.safetensors', 'Photo\\Realistic_Vision_V1.4-inpainting.safetensors', 'Photo\\Realistic_Vision_V1.4.safetensors', 'Photo\\copaxRealistic_v3.safetensors', 'Photo\\humanRealistic_hrv9.safetensors', 'Photo\\hypersimpsAnalogring_v1.safetensors', 'Photo\\insaneRealisticV20_insaneRealisticV20.safetensors', 'Photo\\photorealistic-fuen-v1.ckpt', 'Photo\\polyhedron_v10Pruned.safetensors', 'Photo\\realismEngine_v10.safetensors', 'SDXL\\nightvisionXLPhotorealistic.safetensors', 'SDXL\\sd_xl_base_1.0.safetensors', 'SDXL\\sd_xl_refiner_1.0.safetensors', 'Sci-fi\\AlienFactory.safetensors', 'Sci-fi\\Alien_Landscapes.safetensors', 'Sci-fi\\DreadV2.safetensors', 'Sci-fi\\Fantasialixa.safetensors', 'Sci-fi\\FantasycardDiffusion.ckpt', 'Sci-fi\\FkingScifiV2.safetensors', 'Sci-fi\\GTM_v3.safetensors', 'Sci-fi\\MonsterDiffusion_1.0.safetensors', 'Sci-fi\\RealismReborn_visiongen.safetensors', 'Sci-fi\\Sci-Fi_Diffusion_1.0.safetensors', 'Sci-fi\\colorful_v30.safetensors', 'Sci-fi\\cyberrealistic_v13.safetensors', 'Sci-fi\\epicDiffusion11.safetensors', 'Sci-fi\\epicMixV6_10NOVAE.safetensors', 'Sci-fi\\profantasy_v21.safetensors', 'Sci-fi\\sldrRealism_v20.safetensors', 'Sci-fi\\terrorYamerV2.safetensors', 'Style\\ColoringBook.ckpt', 'Style\\Inkpunk-Diffusion-v2.ckpt', 'Style\\SemiRealMix.safetensors', 'Style\\graphicArt.safetensors', 'Style\\openjourney-v2.ckpt', 'Style\\openjourney_V4.ckpt', 'Style\\photoStyle_v20.ckpt', 'Style\\tShirtPrintDesignsTest_v01.ckpt', 'v1-5-pruned-emaonly.safetensors', 'v2-1_768-ema-pruned.ckpt']

full_model_name ='BestOf_Hardblend'
source_model_name = full_model_name.split('_', 1)[-1]

'''
for checkpoint in allcheckpoints:
  full_path = NODE_ROOT + os.sep + '..' + os.sep + '..' + os.sep + '..' + os.sep + 'models' + os.sep + 'checkpoints' + os.sep + checkpoint
  check_file = os.path.isfile(full_path)
  if check_file == True:
    model_name = os.path.splitext(os.path.basename(full_path))[0]
'''

cutoff_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
is_found = []

for trycut in cutoff_list:
  is_found = difflib.get_close_matches(full_model_name, allcheckpoints, cutoff = trycut)
  if len(is_found) == 1:
    break

if len(is_found) != 1:
  for trycut in cutoff_list:
    is_found = difflib.get_close_matches(source_model_name, allcheckpoints, cutoff = trycut)
    if len(is_found) == 1:
      break

if len(is_found) == 1:
  valid_model = is_found[0]
  print(is_found[0])
  model_full_path = NODE_ROOT + os.sep + '..' + os.sep + '..' + os.sep + '..' + os.sep + 'models' + os.sep + 'checkpoints' + os.sep + valid_model
  print(model_full_path)
  match_model_hash = model_hash(model_full_path)
  print(match_model_hash)

'''
print(difflib.get_close_matches(source_model_name, allcheckpoints, cutoff=0.6))

1638fa9a88
BestOf_Hardblend

import hashlib
'''
