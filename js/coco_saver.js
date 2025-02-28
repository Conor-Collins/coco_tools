import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Coco.Saver",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // IMPORTANT: This must match the key in NODE_CLASS_MAPPINGS
        if (nodeData.name !== "SaverNode") {  
            return;
        }

        console.log("Registering dynamic widget behavior for saver node");
        
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const me = onNodeCreated?.apply(this);
            
            console.log("Saver node created, setting up dynamic widgets");
            
            // Get all widgets
            const fileType = this.widgets.find(w => w.name === "file_type");
            const bitDepth = this.widgets.find(w => w.name === "bit_depth");
            const exrCompression = this.widgets.find(w => w.name === "exr_compression");
            const quality = this.widgets.find(w => w.name === "quality");
            const saveAsGrayscale = this.widgets.find(w => w.name === "save_as_grayscale");

            if (!fileType) {
                console.warn("Saver: file_type widget not found");
                return me;
            }

            // Define the callback that updates widget visibility
            const updateWidgetVisibility = () => {
                console.log("Saver: Updating widget visibility for type:", fileType.value);
                
                // Hide all format-specific widgets first
                if (bitDepth) bitDepth.hidden = true;
                if (exrCompression) exrCompression.hidden = true;
                if (quality) quality.hidden = true;
                if (saveAsGrayscale) saveAsGrayscale.hidden = true;

                // Show relevant widgets based on file type
                switch (fileType.value) {
                    case "exr":
                        if (exrCompression) exrCompression.hidden = false;
                        if (bitDepth) bitDepth.hidden = false;
                        if (saveAsGrayscale) saveAsGrayscale.hidden = false;
                        break;
                    case "png":
                    case "tiff":
                        if (bitDepth) bitDepth.hidden = false;
                        if (saveAsGrayscale) saveAsGrayscale.hidden = false;
                        break;
                    case "jpg":
                    case "webp":
                        if (quality) quality.hidden = false;
                        break;
                }

                // Update node size to accommodate the visible widgets
                requestAnimationFrame(() => {
                    this.setSize([this.size[0], this.computeSize([this.size[0], this.size[1]])[1]]);
                    app.canvas.setDirty(true);
                });
            };

            // Attach the callback to the file_type widget
            const originalCallback = fileType.callback;
            fileType.callback = function(...args) {
                if (originalCallback) {
                    originalCallback.apply(this, args);
                }
                updateWidgetVisibility();
            };

            // Initial update with a small delay to ensure widgets are ready
            setTimeout(updateWidgetVisibility, 1);

            return me;
        };
    }
});