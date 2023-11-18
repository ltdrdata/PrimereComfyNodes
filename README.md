# Primere nodes for ComfyUI

1; Install missing Python libraries if not start for first try. Activate Comfy venv and use 'pip install -r requirements.txt' at the root folder of Primere nodes

2; If started, use the last workflow on the 'Workflow' folder for first watch, all nodes visible under the 'Primere Nodes' submenu

3; Set the right path for image saving in the node 'Primere Image Meta Saver' on 'output_path'

4; Rename 'styles.example.csv' on the 'stylecsv' folder to 'syles.csv' or copy here your own A1111 style file

5; Use the last workflow example on the 'Workflow' folder

## Special features:
- Automatically detect the SD or SDXL checkpoint version, and control the whole process
- You can select model, subpath and orientation under the prompt input to overwrite the system settings
- One button LCM mode (see example workflow)
- Save .json and/or .txt file with process details
- Read A1111 style.csv file, and handle dynamic prompts
- Random noise generator for latent image
- Important and editable styles in the text encoder
- Resolution selector by side ratios only, editable ratio source, and auto detect checkpoint version for right size
- Image size can be convert to "standard" values, fully customizable side ratios
- Remove networks from prompts (Embedding, Lora, and Hypernetwork)
- Embedding handler of A1111 compatible prompts
- Use more than one prompt or style inputs, and select by 'Prompt Switch' node
- Special image meta reader, which handle model name and samplers from A1111 png and jpg, never was easier to recycle your older A1111 or comfy images, with switches you can change the seed/model/size/etc...
- Check/debug generation details

## Nodes in the pack by submenus:

## Inputs:
### Primere Prompt: 
2 input fileds within one node for positive and negative prompts. 3 additional fields appear under the text inputs:
- Subpath: the prefered subpath for final image saving. This can be for example the subject of the generated image, for example 'sci-fi' or 'interior'.
- Use model: the prefered model for image rendering. If your prompt need special model, for example because product design or architechture, here you can force join this model to the prompt rendering process.
- Use orientation: if you prefer vertical or horizontal orientation depending on your prompt, your rendering process will be use this setting instead of global setting. Useful for example for portraits, what much better in vertical orientation.

If you set these fields, (where 'None' mean not set and use system settings) the workflow will use these settings for rendering your prompt.

### Primere Styles:
Style file reder, compatible with A1111 syle.csv file, but little more than the original concept. The file must be copied/symlinked to the 'stylecsv' folder. Rename included 'style.example.csv' to 'style.csv' for first working example, and edit this file manually.
- A1111 compatible CSV headers required for this file: 'name,prompt,negative_prompt'. But this version have 3 required more headers: 'prefered_subpath,prefered_model,prefered_orientation'. These new headers working like the simple prompt input. 
- If you fill these optional columns in the style.csv, the rendering process will be use them. These last 3 fields are optional, if you leave empty the style will be rendering with system settings, if fill and enable to use, system setting will be overwritten.
- You can enable/disable these settings if already entered in csv, but want to use system settings instead.

### Primere Dynamic:
This node render A1111 compatible dynamic prompts, including external files of A1111 dynamic prompt plugin. External files must be copied/symlinked to the 'wildcards' folder and use the '__filepath/of/file__' keyword within your prompt. Use this to decode all style.csv and prompt inputs, because the output of prompt/style nodes not resolved by other comfy dynamic encoder/resolver.

### Primere exif reader:
This node read prompt-exif (called meta) from loaded image. The reader is tested with A1111 'jpg' and 'png' and Comfy 'jpg' and 'png'. Another exif parsers will be included soon, but if you send me AI generated image contains metadata, I will do parser for that.

The node output sending lot of data to the workflow from exif/meta or pnginfo, like model name and sampler. Use this node to distribute required data, and simple off the 'use_exif' switch ig you want to render image by this node inputs.

Use several settings of switches what exif/meta data you want/dont want to use for image rendering. If switch off something, system settings (this node input) will be used instead of image meta.
#### For this node inputs, connect all of your system settings, like in the example workflow. If you switch off the exif reader with 'use_exif' switch, or ignore specified data for example the model, the input values will be used instead of image. The example workflow help to analize how to use this node.

### Primere Embedding Handler:
This node convert A1111 embeddings to Comfy embeddings. Use after dynamically decoded prompts. No need to modify styles.csv from A1111 if you use this node.

## Dashboard:
### Primere Sampler Selector:
Select sampler and scheduler in separated node, and wire outputs to the sampler (exif reader input in the example workflow). This is very useful to separate from other non-setting nodes, and for LCM mode you need two several sampler settings. (see the example workflow, and try out LCM setting)

### Primere VAE Selector:
This node is a simple VAE selector. Use 2 nodes in workflow, 1 for SD, 1 for SDXL compatible VAE for autimatized selection.

### Primere CKPT Selector:
Simple checkpoint selector, but with extras:
- This node automatically detect if the selected model SD or SDXL. Use this output for automatic VAE or size selection and for prompt encoding, see example workflow for details.

### Primere VAE loader:
Use this node to convert VAE name to VAE.

### Primere CKPT Loader:
Use this node to convert checkpoint name to 'MODEL', 'CLIP' and 'VAE'. Use 'is_lcm' input for correct LCM mode, see the example workflow.

### Primere Prompt Switch:
Use this node if you have more than one prompt input (for example several half-ready test prompts). Connect prompt node outputs to this selector node and set the input index at the bottom. 'subpath', 'model', and 'orientation' inputs are optional, only the positive and negative prompt required.

Very important, that don't off the connected node from the middle of inputs. Connect nodes in right queue, and disconnect them only from the last to first. If you getting js error becuase disconnected inputs in wrong gueue, just reload your browser. 

## Primere Seed:
Use only one seed input for all. A1111 style node, connect this one node to all other seed inputs. 

### Primere Noise Latent
This node generate 'empty' latent image, but with several noise settings. You can randomize these setting between min. and max. values using switches, this cause small difference between generated images for same seed, but you can freeze your image if you disable variations of random noise generation.

### Primere Prompt Encoder:
- This node compatible with SD and SDXL models, important to use 'model_version' input for correct working. Try several settings, you will get several results. 
- Use internal positive and negative styles, and check the result in prompt and image outputs. 
- If you getting error if use SD model, you must update your ComfyUI.
- The style source of this node is external file at 'Toml/default_neg.toml' and 'Toml/default_pos.toml', what you can edit if you need changes.
- Comfy encoders not compatible with SD2.x version, you will get black image if select this version from model list.

### Primere Resolution:
- Select image size by side ratios only, and use 'is_sdxl' input for correct SDXL size. 
- Use 'model_version' to handle right image size depending on selected model.   
- You can calculate image size by really custom ratios at the bottom inputs (and switch).
- Use 'round_to_standard' switch if you want to modify the exactly calculated size by the 'officially' recommended SD / SDXL values.
- The ratios of this node stored in external file at 'Toml/resolution_ratios.toml', what you can edit if you need changes.

### Primere Steps & Cfg:
Use this separated node for sampler. If you use LCM mode, you need 2 of this node. See and test the attached example node.

### Primere Prompt Cleaner:
This node remove Lora, Hypernetwork and Embedding (booth A1111 and Comfy) from the prompt. Use switches what netowok(s) you want to remove or keep in the prompt. Use 'remove_only_if_sdxl' if you want keep all of these networks for all SD models, and remove only if SDXL checkpoint selected.

### Primere LCM Selector:
Use this node to switch on/off LCM mode in whole rendering process. Wire two sampler and cfg/steps setting to the inputs (one of them must be compatible with LCM settings), and connect this node output to the sampler, like in the example workflow. The 'IS_LCM' output important for CKPT loader and the Exif reader for correct rendering.

## Outputs:
### Primere Meta Saver:
This node save the image, but with/without metadata, and save meta to .json file if you want. Wire metadata from the Exif reader node only, and use 'prefered_subpath' input if you want to overwrite the node settings by several prompt input nodes. Set 'output_path' input correctly, depending your system.

### Primere Any Debug:
Use this node to display 'any' output values of several nodes, like prompts or metadata. See the example workflow for details.

### Primere Style Pile:
Style collection for generated images. Set and connect this node to the 'Prompt Encoder'. No forget to set and play with style strenght. The source of this node is external file at 'Toml/stylepile.toml', what you can edit if you need changes.

### Contact:
Sorry, but contact info later. Use 'git pull' to refrest this node pack after first install, but maybe have to use the newest workflow after the pull.