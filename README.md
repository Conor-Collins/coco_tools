# ComfyUI-CoCoTools

A set of custom nodes for ComfyUI providing advanced image processing, file handling, and utility functions.

## Features
- Image processing utilities
- EXR image input and outputs nodes
- JSON-based scaling tools for file/path management
- Utility nodes for image and data operations


## Installation for comfyui portable (tested on 0.3.14)

from the python_embeded/ folder

```bash
python.exe -m pip install -r ./ComfyUI/custom_nodes/ComfyUI-CoCoTools/requirements.txt
```

### Manual Installation
1. Clone the repository into your ComfyUI `custom_nodes` directory
2. Install dependencies
3. Restart ComfyUI




## To-Do
#### IO
- [x] implement proper exr loading
- [ ] implement EXR sequence loader
- [x] implement exr saver for proper exr saving using OpenImageIO
- [x] implement multilayer exr system ( render passes, aovs, embedded images, etc)  (still testing - alpha)
- [x] contextual menus from the file type picked on saver

#### Color
- [x] split colorspace conversion into separate node
- [x] implement minimal color management system
- [ ] Add ACES or OCIO color config profiles into the conversion node


#### Processing
- [ ] create split frequency for video node
- [ ] create some experimental frequency tools ( motion detection from frame offset seperation)

#### Documentation
- [ ] add more info on specific nodes
- [ ] add more workflows to show how to use custom nodes
- [] visual examples in the readme

#### Registration
- [ ] submit to ComfyUI Registry