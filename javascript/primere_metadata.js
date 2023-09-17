import { app } from "/scripts/app.js";

// Adds an upload button to the nodes
app.registerExtension({
	name: "primere_meta.Imagebox",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "PrimereImageMetaReader") {
			nodeData.input.required.upload = ["IMAGEUPLOAD"];
		}
	},
});