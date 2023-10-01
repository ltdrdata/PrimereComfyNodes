import difflib

def change_exif_samplers(sampler_name_exif, comfy_schedulers):
  lastchars = sampler_name_exif[-2:]
  if lastchars == ' a':
    sampler_name_exif = sampler_name_exif.rsplit(' a', 1)[0] + ' ancestral'
  sampler_name_exif = sampler_name_exif.replace(' a ', ' ancestral ').replace(' ', '_').replace('++', 'pp').replace('dpm2', 'dpm_2').replace('unipc', 'uni_pc')
  for comfy_scheduler in comfy_schedulers:
    sampler_name_exif = sampler_name_exif.removesuffix(comfy_scheduler).removesuffix('_')

  return sampler_name_exif

def check_sampler_from_exif(sampler_name_exif, sampler_name, scheduler_name):
  comfy_samplers = ['euler', 'euler_ancestral', 'heun', 'dpm_2', 'dpm_2_ancestral', 'lms', 'dpm_fast', 'dpm_adaptive',
                    'dpmpp_2s_ancestral', 'dpmpp_sde', 'dpmpp_sde_gpu', 'dpmpp_2m', 'dpmpp_2m_sde', 'dpmpp_2m_sde_gpu',
                    'dpmpp_3m_sde', 'dpmpp_3m_sde_gpu', 'ddpm', 'ddim', 'uni_pc', 'uni_pc_bh2']

  comfy_schedulers = ['normal', 'karras', 'exponential', 'sgm_uniform', 'simple', 'ddim_uniform']

  sampler_name_exif_for_cutoff = change_exif_samplers(sampler_name_exif, comfy_schedulers)
  is_found_sampler = []
  is_found_scheduler = []

  cutoff_list_samplers = [1, 0.9, 0.8, 0.7, 0.6]
  for trycut in cutoff_list_samplers:
    is_found_sampler = difflib.get_close_matches(sampler_name_exif_for_cutoff, comfy_samplers, cutoff=trycut)

  if len(is_found_sampler) >= 1:
    sampler_name = is_found_sampler[0]

  if " " in sampler_name_exif:
    if any((match := substring) in sampler_name_exif for substring in comfy_schedulers):
      scheduler_name = match
    else:
      cutoff_list_schedulers = [0.7, 0.6, 0.5, 0.4]
      for trycut in cutoff_list_schedulers:
        is_found_scheduler = difflib.get_close_matches(sampler_name_exif, comfy_schedulers, cutoff=trycut)

  if len(is_found_scheduler) >= 1:
    scheduler_name = is_found_scheduler[0]

  if sampler_name not in comfy_samplers:
    sampler_name = comfy_samplers[0]

  if scheduler_name not in comfy_schedulers:
    scheduler_name = comfy_schedulers[0]

  return {'sampler': sampler_name, 'scheduler': scheduler_name}

all_a11_samplers = ['DDIM', 'DPM adaptive', 'DPM fast', 'DPM++ 2M Karras', 'DPM++ 2M SDE Karras', 'DPM++ 2M SDE',
                    'DPM++ 2M', 'DPM++ 2S a Karras', 'DPM++ 2S a', 'DPM++ SDE Karras', 'DPM++ SDE', 'DPM2 a Karras',
                    'DPM2 a', 'DPM2 Karras', 'DPM2', 'Euler a', 'Euler', 'Heun', 'LMS Karras', 'LMS', 'PLMS', 'UniPC']

sampler_name = 'original_sampler_name'
scheduler_name = 'original_scheduler_name'

for sampler_name_exif in all_a11_samplers:
  print(sampler_name_exif)
  result = check_sampler_from_exif(sampler_name_exif.lower(), sampler_name, scheduler_name)
  print(result)
