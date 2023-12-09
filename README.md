# Primere nodes for ComfyUI

1; Install missing Python libraries if not start for first try. Activate Comfy venv and use 'pip install -r requirements.txt' at the root folder of Primere nodes (or check error messages and install missing libs manually)

2; If started, use the last workflow on the 'Workflow' folder for first watch, all nodes visible under the 'Primere Nodes' submenu if you need custom workflow

3; Set the right path for image saving in the node 'Primere Image Meta Saver' on 'output_path' input

4; Rename 'styles.example.csv' on the 'stylecsv' folder to 'syles.csv' or copy here your own A1111 style file if you want to use 'Primere Styles' node

## Special features:
- Automatically detect the SD or SDXL checkpoint version, and control the whole process by the result
- You can select model, subpath and orientation under the prompt input to overwrite the system settings, same settings under the Styles node
- One button LCM mode (see example workflow)
- Save .json and/or .txt file with process details
- Read A1111 style.csv file, and handle dynamic prompts
- Random noise generator for latent image
- Important and easy editable styles in the text encoder
- Resolution selector by side ratios only, editable ratio source in external file, and auto detect checkpoint version for right size
- Image size can be convert to "standard" values, fully customizable side ratios at the bottom of the resolutionh selector node
- Original image size multiplied to upscaler by two several ratios, one for SD and another one for SDXL models
- Remove previously included networks from prompts (Embedding, Lora, and Hypernetwork), use if the used model incompatible with them
- Embedding handler of A1111 compatible prompts, this node convert A1111 Embeddings to ComfyUI
- Use more than one prompt or style inputs for testing, and select one by 'Prompt Switch' node
- Special image meta reader, which handle model name and samplers from A1111 .png and .jpg, never was easier to recycle your older A1111 or comfy images using same or several tools, with switches you can change the seed/model/size/etc...
- Check/debug generation details
- (As I see, Comfy doesn't handle SD2.x checkpoints, always geting black image)

## Nodes in the pack by submenus:

## Inputs:
### Primere Prompt: 
2 input fileds within one node for positive and negative prompts. 3 additional fields appear under the text inputs:
- Subpath: the prefered subpath for final image saving. This can be for example the subject of the generated image, like 'sci-fi' 'art' or 'interior'.
- Use model: the prefered checkpoint for image rendering. If your prompt need special checkpoint, for example because product design or architechture, here you can force apply this model to the prompt rendering process.
- Use orientation: if you prefer vertical or horizontal orientation depending on your prompt, your rendering process will be use this setting instead of global setting from 'Primere Resolution' node. Useful for example for portraits, what usually better in vertical orientation.

If you set these fields, (where 'None' mean not set and use system settings) the workflow will use all of these settings for rendering your prompt instead of settings in 'Dashboard' group.

### Primere Styles:
Style file reader, compatible with A1111 syle.csv file, but little more than the original concept. The file must be copied/symlinked to the 'stylecsv' folder. Rename included 'style.example.csv' to 'style.csv' for first working example, and edit this file manually.
- A1111 compatible CSV headers required for this file: 'name,prompt,negative_prompt'. But this version have more 3 required headers: 'prefered_subpath,prefered_model,prefered_orientation'. These new headers working like bottom fields in the simple prompt input. 
- If you fill these 3 optional columns in the style.csv, the rendering process will use them. These last 3 fields are optional, if you leave empty the style will be rendering with system 'dashboard' settings, if fill and enable to use at the bottom of node, dashboard settings will be overwritten.
- You can enable/disable these settings if already entered in csv, but want to use system settings instead, no need to delete if you failed.

### Primere Dynamic:
This node render A1111 compatible dynamic prompts, including external files of A1111 dynamic prompt plugin. External files must be copied/symlinked to the 'wildcards' folder and use the '__filepath/of/file__' keyword within your prompt. Use this to decode all style.csv and double prompt inputs, because the output of prompt/style nodes not resolved by other comfy dynamic decoder/resolver.

### Primere exif reader:
- This node read prompt-exif (called meta) from loaded image.
- This is very important (the most important) node in the example workflow, it has a central settings distribution role.
- The reader is tested with A1111 'jpg' and 'png' and Comfy 'jpg' and 'png'. Another exif parsers will be included soon, but if you send me AI generated image contains metadata what failed to read, I will do parser/debug for that.

The node output sending lot of data to the workflow from exif/meta or pnginfo is its included  to selected image like model name and sampler. Use this node to distribute settings, and simple off the 'use_exif' switch if you don't want to render image by this node inputs, then you can use your own prompts and dashboard settings.

Use several settings of switches what exif/meta data you want/don't want to use for image rendering. If switch off something, dashboard settings (must be connected this node input) will be used instead of image exif/meta.
#### For this node inputs, connect all of your dashboard settings, like in the example workflow. If you switch off the exif reader with 'use_exif' switch, or ignore specified data for example the model, the input values will be used instead of image meta. The example workflow help to analize how to use this node.

### Primere Embedding Handler:
This node convert A1111 embeddings to Comfy embeddings. Use after dynamically decoded prompts (booth text and style). No need to modify manually styles.csv from A1111 if you use this node.

## Dashboard:
### Primere Sampler Selector:
Select sampler and scheduler in separated node, and wire outputs to the sampler (through exif reader input in the example workflow). This is very useful to separate from other non-setting nodes, and for LCM mode you need two several sampler settings. (see the example workflow, and try to undestand LCM setting)

### Primere VAE Selector:
This node is a simple VAE selector. Use 2 nodes in workflow, 1 for SD, 1 for SDXL compatible VAE for autimatized selection. The checkpoint selector and loader get the loaded checkpoint version.

### Primere CKPT Selector:
Simple checkpoint selector, but with extras:
- This node automatically detect if the selected model SD or SDXL. Use this output for automatic VAE or size selection and for prompt encoding, see example workflow for details. In Comfy SD2.x checkpoints not working well, use only SD1.x and SDXL.

### Primere VAE loader:
Use this node to convert VAE name to VAE.

### Primere CKPT Loader:
Use this node to convert checkpoint name to 'MODEL', 'CLIP' and 'VAE'. Use 'is_lcm' input for correct LCM mode, see the example workflow.

### Primere Prompt Switch:
Use this node if you have more than one prompt input (for example several half-ready test prompts). Connect prompt/style node outputs to this node inputs and set the right index at the bottom. To connect 'subpath', 'model', and 'orientation' inputs are optional, only the positive and negative prompt required.

Very important, that don't remove the connected node from the middle or from the top of inputs. Connect nodes in right queue, and disconnect them only from the last to first. If you getting js error becuase disconnected inputs in wrong gueue, just reload your browser and use 'reload node' menu with right click on node. 

## Primere Seed:
Use only one seed input for all. A1111 style node, connect this one node to all other seed inputs. 

### Primere Noise Latent
This node generate 'empty' latent image, but with several noise settings. You can randomize these setting between min. and max. values using switches, this cause small difference between generated images for same seed and settings, but you can freeze your image if you disable variations of random noise generation.

### Primere Prompt Encoder:
- This node compatible with SD and SDXL models, important to use 'model_version' input for correct working. Try several settings, you will get several results. 
- Use positive and negative styles, and check the best result in prompt and image outputs. 
- If you getting error if use SD model, you must update (git pull) your ComfyUI.
- The style source of this node is external file at 'Toml/default_neg.toml' and 'Toml/default_pos.toml' files, what you can edit if you need changes.
- Comfy internal encoders not compatible with SD2.x version, you will get black image if select this SD2.x checkpoint version from model selector.

### Primere Resolution:
- Select image size by side ratios only, and use 'model_version' input for correct SD or SDXL size on the output.  
- You can calculate image size by really custom ratios at the bottom float inputs (and use switch), or just edit the ratio source file.
- The ratios of this node stored in external file at 'Toml/resolution_ratios.toml', what you can edit if you need changes.
- Use 'round_to_standard' switch if you want to modify the exactly calculated size to the 'officially' recommended SD / SDXL values. This is usually very small modification.
- Not sure what orientation the best for your prompt and want to test in batch image generation? Just set batch value on the Comfy menu and swith 'rnd_orientation' to randomize vertical and horizontal images
- Set the base model (SD1.x not SDQL) resolution to 512, 768, 1024, or 1280.
- If you want to check your prompt in batch but with several orientation, use rnd_oriantation to randomize horizontal/vertical images for same prompt and settings.

### Primere Resolution Multiplier:
Multiply the base image size for upscaling. Important to use 'model_version' if you want to use several multiplier for SD and SDXL models.

### Primere Steps & Cfg:
Use this separated node for sampler/meta reader inputs. If you use LCM mode, you need 2 settings of this node. See and test the attached example workflow.

### Primere Prompt Cleaner:
This node remove Lora, Hypernetwork and Embedding (booth A1111 and Comfy) from the prompt and style inputs. Use switches what netowok(s) you want to remove or keep in the final prompt. Use 'remove_only_if_sdxl' if you want keep all of these networks for all SD models, and remove only if SDXL checkpoint selected.

### Primere LCM Selector:
Use this node to switch on/off LCM mode in whole rendering process. Wire two sampler and cfg/steps setting to the inputs (one of them must be compatible with LCM settings), and connect this node output to the sampler/exif reader, like in the example workflow. The 'IS_LCM' output important for CKPT loader and the Exif reader for correct rendering.

## Outputs:
### Primere Meta Saver:
This node save the image, but with/without metadata, and save meta to .json file if you want. Wire metadata from the Exif reader node only, and use optional 'prefered_subpath' input if you want to overwrite the node settings by several prompt input nodes. Set 'output_path' input correctly, depending your system.

### Primere Any Debug:
Use this node to display 'any' output values of several nodes, like prompts or metadata (metadata is formatted). See the example workflow for details.

### Primere Text Output
Use this node to diaplay text. 

### Primere Style Pile:
Style collection for generated images. Set and connect this node to the 'Prompt Encoder'. No forget to set and play with style strenght. The source of this node is external file at 'Toml/stylepile.toml', what you can edit if you need changes.

## Visuals:
Here are same functions like upper, but the selection (for example checkpoints, loras, embeddings and hypernetworks) possible by image previews on modal. Very similar than in several themes of A1111.

### Primere Visual CKPT selector:
The first visual selector for checkpoints. You must reproduce your original checkpoint path to ComfyUI\web\extensions\Primere\images\checkpoints\ path but inly the preview images, same name as the checkpoint but with .jpg only extension.
As extra features you can enable/disable modal with show_modal switch, and exclude files and paths in modal start with . (point) character if show_hidden switch is off. 

# Contact:
### Discord name: primere -> ask email if you need