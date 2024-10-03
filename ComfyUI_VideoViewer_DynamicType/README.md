# ComfyUI Video Viewer and Dynamic Type Extension

This extension adds video viewing, uploading capabilities, and a dynamic type converter to ComfyUI.

## Features

1. Video Viewer Node:
   - View uploaded videos directly in ComfyUI
   - Upload new video files
   - Stream videos using ffmpeg

2. Dynamic Type Converter Node:
   - Convert input to different types (STRING, INT, FLOAT, JSON)
   - Dynamically update the node's output type based on the operation

## Installation

1. Place this folder in the `custom_nodes` directory of ComfyUI.
2. Restart ComfyUI.

## Usage

### Video Viewer Node:
- Add a "Video Viewer" node to your workflow.
- Set the "video_path" parameter to the path of an existing video file.
- Click "Upload Video" to upload a new video file.
- Click "View Video" to open a modal and play the video.

### Dynamic Type Converter Node:
- Add a "Dynamic Type Converter" node to your workflow.
- Select an operation (STRING, INT, FLOAT, or JSON) from the dropdown.
- Enter the input value in the text field.
- Click "Update Type" to update the node's output type based on the selected operation.
- Connect the output to other nodes that accept the corresponding type.

## Requirements

- ffmpeg and ffprobe must be installed and accessible in the system PATH.
- ComfyUI environment with necessary dependencies.

## Note

- Ensure that the video files are in a directory accessible by ComfyUI.
- Uploaded videos are stored in the ComfyUI input directory with unique filenames.
- The Dynamic Type Converter Node updates its output type in real-time, allowing for flexible workflows.
- Error handling is implemented for both nodes to provide meaningful feedback.

## Troubleshooting

- If videos fail to load, check the server logs for detailed error messages.
- Ensure that the input value for the Dynamic Type Converter Node is valid for the selected operation.
- If the nodes don't appear in ComfyUI, verify that the extension is correctly placed in the custom_nodes directory and that ComfyUI has been restarted.

## Contributing

Contributions to improve this extension are welcome. Please submit issues or pull requests on the project's repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
