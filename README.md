# Primere nodes for ComfyUI

1; Install missing Python libraries if not start. Activate venv and use 'pip install -r requirements.txt' at the root folder of Primere nodes

2; Use the last workflow on the 'Workflow' folder for first watch, all nodes visible under the 'Primere Nodes' submenu

## Nodes in the pack:

## Inputs:
### Primere Prompt: 
2 input fileds within one node for positive and negative prompts. 3 additional fields appear under the text inputs:
- Subpath: the prefered subpath for final image saving. This can be for example the subject of the generated image, for example 'sci-fi' or 'interior'
- Use model: the prefered model for image. If your prompt need special model, for example because product design or architechture, here you can force join this model to the prompt rendering process
- Use orientation: if you prefer vertical or horizontal orientation depending on your node, your prompt will be use this setting instead of global setting. Useful for example for portraits, what much better in vertical orientation

If you set these fields, what mean not set to 'None', the workflow will use these settings if rendering your prompt.

### Primere Styles:
Style file reder, compatible with A1111 syle.csv file, but little more than the original concept. The file must be copied/symlinked to the 'stylecsv' folder. Rename included 'style.example.csv' to 'style.csv' for first working example, and edit this file manually.
- A1111 compatible CSV headers required for this file: 'name,prompt,negative_prompt'. But this version have 3 more headers: 'prefered_subpath,prefered_model,prefered_orientation'. These new headers working like the simple prompt input. 
- If you fill these optional columns in the style.csv, the rendering process will be use them. These last 3 fields are optional, if you leave empty, the style will be rendering with system settings.
- You can enable/disable these settings if filled but want to use system settings instead.

### Primere Dynamic:
This node render A1111 compatible dynamic prompts, including external files. External files must be copied/symlinked to the 'wildcards' folder and use '__filepath__' within your prompt. Use this to decode all style and prompt inputs, because the output of prompt/style nodes not rendered by other comfy dynamic encoder.

### Primere exif reader:
This node read prompt-exif from loaded image. The reader is tested with A1111 'jpg' and 'png' and Comfy 'jpg' and 'png'. Another exif parsers will be included soon, but if you send me image contains prompt data, I will do parser for that.

The node output sending lot of data from exif or pnginfo, like model name and sampler.

Use switches what exif data you want/dont want to use for image rendering. If switch off something, system settings will be used instead of image meta.
#### For this node input, connect all of your system settings, like in the example workflow. If you switch off the exif reader with 'use_exif' switch, or ignore specified data, the connected values will be used instead. The example workflow is good to analize how to use this node.

### Primere Embedding Handler:
This node convert A1111 embeddings to Comfy embeddings. Use after dynamically decoded prompts.

## Dashboard:
### Primere Sampler Selector:
Select sampler and scheduler in separated node, and wire outputs to the sampler. This is very useful to separate from other non-setting nodes, and for LCM mode you need two version from this node. (see the example workflow)

### Primere VAE Selector:
This node is a simple VAE selector. Use 2 nodes, 1 for SD, 1 for SDXL compatible VAE for autimatized selection.

### Primere CKPT Selector:
Simple checkpoint selector, but with extras:
- If you fill 'sdxl_path' input field (you must store your SDXL checkpoints in separated folder), you will give value 1 in 'IS_SDXL' output if your selected model is SDXL, but the value is 0 if SD. Use for automatic VAE or size selection, and for promot encoding, see example workflow for details.

### Primere VAE loader:
Use this node to convert VAE name to VAE.

### Primere CKPT Loader:
Use this node to convert checkpoint name to 'MODEL', 'CLIP' and 'VAE'. Use 'is_lcm' input for correct LCM mode, see the example workflow.

### Primere Prompt Switch:
Use this node if you have more than one prompt input (for example several harlf-ready test prompts). Connect prompt source outputs to this node and select the input index at the bottom. 'subpath', 'model', and 'orientation' inputs are optional, only the positive and negative required.

## Primere Seed:
Use only one seed input for all. A1111 style node, connect this one node to all other seed inputs. 

### Primere Noise Latent
This node genrate 'empty' latent image, but with several settings. You can randomize these setting between min. and max. values using swithch, this cause small difference between generated images for same seed.

### Primere rompt Encoder:
- This node compatible with SD and SDXL models, important to use 'is_sdxl' input for correct working. Try several settings, you will get several results. Use internal positive and negative styles, and check the result in prompt and image outputs.

### Primere Resolution:
- Select image size by side ratios only, and use 'is_sdxl' input for correct SDXL size. 
- If the 'is_sdxl' input getting 0, select your checkpoint version in 'default_sd' switch. SD 1.x mean 512, SD 2.x mean 768 basic size.
- You can calculate image size by really custom ratios at the bottom inputs.
- Use 'round_to_standard' switch if you want to modify the calculated size by the 'official' SD values.

### Primere Steps & Cfg:
Use this separated node for sampler. If you use LCM mode, you need 2 of this node.

### Primere Prompt Cleaner:
This node remove Lora, Hypernetwirk and Embedding from the prompt. Use switches what netowok(s) you want to remove or keep in the prompt. Use 'remove_only_if_sdxl' you wanna keep all of these networks for all SD models, and remove only if SDXL checkpoint selected.

### Primere LCM Selector:
Use this node to use LCM mode in whole rendering process. Wire two sampler and cfg setting to the inputs (one of them must be compatible with LCM settings), and connect this poutput to the sampler, like in the example workflow. The 'IS_LCM' output important for CKPT loader and the Exif reader for correct rendering.

## Outputs:
### Primere Meta Saver:
This node save the image, but with/without metadata, and save meta to json if you want. Wire metadata from the Exif reader node only, and use 'prefered_subpath' input if you want to overwrite the node settings by several prompt input nodes. Set 'output_path' input correctly, depending your system.

### Primere Any Debug:
Use this node to diaspay several 'any' node values, like prompts or metadata. See the example workflow for details.

### Primere Style Pile:
Style collection for generated images. Set and connect this node to the 'Prompt Encoder'. No forget to set and play with style strenght.