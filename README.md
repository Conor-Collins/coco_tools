# ComfyUI-CoCoTools

A set of custom nodes for ComfyUI providing advanced image processing, file handling, and utility functions.

## Features
- Image processing utilities  
- JSON-based scaling tools for file/path management
- Utility nodes for image and data operations


## Installation
```bash
pip install -r requirements.txt
```

### Manual Installation
1. Clone the repository into your ComfyUI `custom_nodes` directory
2. Install dependencies
3. Restart ComfyUI




## To-Do
- [x] implement proper exr loading
- [ ] implement EXR sequence loader
- [ ] implement exr saver for proper exr saving using OpenImageIO
- [ ] implement minimal color management system
- [ ] implement multilayer exr system ( render passes, aovs, embedded images, etc)
- [ ] create split frequency for video node
- [ ] create some experimental frequency tools ( motion detection from frame offset seperation)
- [ ] add more info on specific nodes
- [ ] add more workflows to show how to use custom nodes

- [ ] submit to ComfyUI Registry