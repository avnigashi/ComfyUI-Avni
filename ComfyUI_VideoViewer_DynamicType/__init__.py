from .nodes.video_viewer_node import VideoViewerNode
from .nodes.dynamic_type_node import DynamicTypeNode

NODE_CLASS_MAPPINGS = {
    "VideoViewerNode": VideoViewerNode,
    "DynamicTypeNode": DynamicTypeNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoViewerNode": "Video Viewer",
    "DynamicTypeNode": "Dynamic Type Converter"
}

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
