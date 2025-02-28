import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "LoadExr",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // Only handle our load_exr node
        if (nodeData.name !== "LoadExr") {
            return;
        }

        console.log("Registering dictionary-based EXR loader node");
        
        // Store original methods to call them later
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        const onExecuted = nodeType.prototype.onExecuted;
        
        // Override onNodeCreated to set up node
        nodeType.prototype.onNodeCreated = function() {
            const result = onNodeCreated?.apply(this, arguments);
            
            // Store original outputs
            this.baseOutputs = [];
            for (let i = 0; i < this.outputs.length; i++) {
                this.baseOutputs.push({...this.outputs[i]});
            }
            
            // Initialize storage for layer information
            this.layerInfo = {
                count: 0,
                types: {}
            };
            this.cryptomatteCount = 0;
            this.currentExrPath = "";
            
            console.log("EXR Loader node created");
            
            return result;
        };
        
        // Override onExecuted to handle output from Python node
        nodeType.prototype.onExecuted = function(message) {
            if (onExecuted) {
                onExecuted.apply(this, arguments);
            }
            
            // Process our specific message to extract metadata
            if (!message || message.type !== "executed") {
                return;
            }
            
            console.log("EXR Loader node executed", message);
            
            try {
                // Extract metadata from the message
                if (message.outputs && message.outputs.length >= 3) {
                    // The third output is metadata (STRING)
                    const metadataStr = message.outputs[2]; 
                    
                    // Parse the metadata JSON
                    try {
                        const metadata = JSON.parse(metadataStr);
                        
                        if (metadata) {
                            console.log("Processing EXR metadata", metadata);
                            
                            // Store the current file path
                            if (metadata.file_path) {
                                this.currentExrPath = metadata.file_path;
                            }
                            
                            // Extract layer information
                            this.updateLayerInfo(metadata);
                            
                            // Update the node's title to show layer count
                            this.updateNodeTitle();
                            
                            // Notify ComfyUI that the node has been updated
                            app.graph.setDirtyCanvas(true);
                        }
                    } catch (e) {
                        console.error("Error parsing EXR metadata:", e);
                    }
                }
            } catch (e) {
                console.error("Error processing node execution message:", e);
            }
        };
        
        // Update layer information from metadata
        nodeType.prototype.updateLayerInfo = function(metadata) {
            const layerTypes = metadata.layer_types || {};
            const channelGroups = metadata.channel_groups || {};
            
            // Count how many layers of each type
            let imageCount = 0;
            let maskCount = 0;
            let cryptomatteCount = 0;
            
            // Count layers by type
            for (const [layerName, layerType] of Object.entries(layerTypes)) {
                if (layerType === "IMAGE") {
                    imageCount++;
                } else if (layerType === "MASK") {
                    maskCount++;
                }
            }
            
            // Count cryptomatte layers
            for (const groupName in channelGroups) {
                if (groupName.toLowerCase().includes("cryptomatte") || 
                    groupName.toLowerCase().startsWith("crypto")) {
                    cryptomatteCount++;
                }
            }
            
            // Update layer info
            this.layerInfo = {
                count: Object.keys(layerTypes).length,
                types: {
                    "IMAGE": imageCount,
                    "MASK": maskCount
                }
            };
            
            this.cryptomatteCount = cryptomatteCount;
        };
        
        // Update the node title to show layer count
        nodeType.prototype.updateNodeTitle = function() {
            const layerCount = this.layerInfo.count;
            const imageCount = this.layerInfo.types["IMAGE"] || 0;
            const maskCount = this.layerInfo.types["MASK"] || 0;
            const cryptoCount = this.cryptomatteCount;
            
            // Only change the title if we actually have layers
            if (layerCount > 0) {
                const baseName = this.title.split(" [")[0]; // Get base name without counter
                this.title = `${baseName} [${layerCount}L, ${imageCount}I, ${maskCount}M, ${cryptoCount}C]`;
            }
        };
        
        // Add visual feedback for layer information
        const onDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function(ctx) {
            if (onDrawBackground) {
                onDrawBackground.apply(this, arguments);
            }
            
            // Add visual indication of layer counts
            if (this.layerInfo && this.layerInfo.count > 0) {
                // Draw a colored bar at the bottom
                ctx.fillStyle = "#2a363b";
                ctx.fillRect(0, this.size[1] - 5, this.size[0], 5);
                
                // Use a color based on the number of layers
                const layerCount = this.layerInfo.count;
                const percentage = Math.min(layerCount / 50, 1.0);  // Max 50 layers for full bar
                
                ctx.fillStyle = "#00a2ff";
                ctx.fillRect(0, this.size[1] - 5, this.size[0] * percentage, 5);
                
                // Add text showing detailed layer counts
                ctx.fillStyle = "#ffffff";
                ctx.font = "10px Arial";
                ctx.textAlign = "right";
                
                const imageCount = this.layerInfo.types["IMAGE"] || 0;
                const maskCount = this.layerInfo.types["MASK"] || 0;
                const cryptoCount = this.cryptomatteCount;
                
                ctx.fillText(`Layers: ${layerCount} | Images: ${imageCount} | Masks: ${maskCount} | Crypto: ${cryptoCount}`, 
                             this.size[0] - 5, this.size[1] - 8);
            }
        };
        
        // Add a refresh menu option
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function(_, options) {
            if (getExtraMenuOptions) {
                getExtraMenuOptions.apply(this, arguments);
            }
            
            options.push({
                content: "Refresh EXR",
                callback: () => {
                    // Reset layer info
                    this.layerInfo = {
                        count: 0,
                        types: {}
                    };
                    this.cryptomatteCount = 0;
                    
                    // Reset title
                    this.title = "Load EXR";
                    
                    // Trigger a re-execution of the node
                    app.graph.setDirtyCanvas(true);
                }
            });
            
            // Add an option to show layer information if we have metadata
            if (this.layerInfo && this.layerInfo.count > 0) {
                options.push({
                    content: "Show Layer Info",
                    callback: () => {
                        // Log current layer info
                        console.log("EXR Layer Information:", this.layerInfo);
                        
                        // Create a formatted message
                        let message = "EXR Layer Summary:\n";
                        message += `- Total Layers: ${this.layerInfo.count}\n`;
                        message += `- Image Layers: ${this.layerInfo.types["IMAGE"] || 0}\n`;
                        message += `- Mask Layers: ${this.layerInfo.types["MASK"] || 0}\n`;
                        message += `- Cryptomatte Layers: ${this.cryptomatteCount}\n`;
                        message += "\nConnect to a Shamble node to select specific layers.";
                        
                        alert(message);
                    }
                });
            }
        };
        
        // Add custom handler for widget change
        const onWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function(widget, value) {
            if (onWidgetChanged) {
                onWidgetChanged.apply(this, arguments);
            }
            
            if (widget.name === "image_path" && value) {
                // When image path changes, reset layer info
                this.layerInfo = {
                    count: 0,
                    types: {}
                };
                this.cryptomatteCount = 0;
                this.title = "Load EXR";
                this.currentExrPath = value;
                
                // Trigger a re-execution of the node
                app.graph.setDirtyCanvas(true);
            }
        };
        
        // Serialization and configuration methods
        const onSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function(data) {
            if (onSerialize) {
                onSerialize.apply(this, arguments);
            }
            
            // Save layer info
            data.layer_info = this.layerInfo;
            data.cryptomatte_count = this.cryptomatteCount;
            data.current_exr_path = this.currentExrPath;
        };
        
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(data) {
            if (onConfigure) {
                onConfigure.apply(this, arguments);
            }
            
            // Restore layer info
            if (data.layer_info) {
                this.layerInfo = data.layer_info;
            }
            
            if (data.cryptomatte_count !== undefined) {
                this.cryptomatteCount = data.cryptomatte_count;
            }
            
            if (data.current_exr_path) {
                this.currentExrPath = data.current_exr_path;
            }
            
            // Update node title with layer counts
            this.updateNodeTitle();
        };
    },
});