import torch
import numpy as np
import os
from deepface import DeepFace
import tensorflow as tf
import comfy.utils
import folder_paths

# Configure TensorFlow to avoid the sequential layer initialization error
tf.keras.backend.clear_session()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def comfy_image_from_deepface_image(deepface_image):
    """Convert DeepFace image format to ComfyUI format"""
    image_data = np.array(deepface_image).astype(np.float32)
    return torch.from_numpy(image_data)[None,]

def deepface_image_from_comfy_image(comfy_image):
    """Convert ComfyUI image format to DeepFace format"""
    image_data = np.clip(255 * comfy_image.cpu().numpy(), 0, 255).astype(np.uint8)
    return image_data[:, :, ::-1]  # Convert RGB to BGR

def prepare_deepface_home():
    """Set up DeepFace home directory structure"""
    deepface_path = os.path.join(folder_paths.models_dir, "deepface")
    deepface_dot_path = os.path.join(deepface_path, ".deepface")
    deepface_weights_path = os.path.join(deepface_dot_path, "weights")
    if not os.path.exists(deepface_weights_path):
        os.makedirs(deepface_weights_path)
    os.environ["DEEPFACE_HOME"] = deepface_path

class DeepFaceAnalyzeNode:
    """Node for facial attribute analysis (age, gender, emotion, race)"""

    def __init__(self):
        prepare_deepface_home()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detector": (["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe", "yolov8", "yunet", "fastmtcnn"],),
                "actions": (["age", "gender", "emotion", "race"],),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze_face"
    CATEGORY = "DeepFace"

    def analyze_face(self, image, detector="opencv", actions="age"):
        try:
            # Convert image to DeepFace format
            img = deepface_image_from_comfy_image(image)

            # Convert actions to list if string
            actions_list = [actions] if isinstance(actions, str) else actions

            # Analyze face
            result = DeepFace.analyze(
                img,
                detector_backend=detector,
                actions=actions_list,
                enforce_detection=False
            )

            # Format results
            output = ""
            if isinstance(result, list):
                result = result[0]  # Take first face if multiple detected

            if "age" in actions_list and "age" in result:
                output += f"Age: {result['age']}\n"
            if "gender" in actions_list and "dominant_gender" in result:
                output += f"Gender: {result['dominant_gender']}\n"
            if "emotion" in actions_list and "dominant_emotion" in result:
                output += f"Emotion: {result['dominant_emotion']}\n"
            if "race" in actions_list and "dominant_race" in result:
                output += f"Race: {result['dominant_race']}\n"

        except Exception as e:
            output = f"Analysis failed: {str(e)}"

        return (output,)

class DeepFaceVerifyNode:
    """Node for face verification between two images"""

    def __init__(self):
        prepare_deepface_home()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "model_name": (["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"],),
                "detector": (["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe", "yolov8", "yunet", "fastmtcnn"],),
                "distance_metric": (["cosine", "euclidean", "euclidean_l2"],),
                "threshold": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                })
            }
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT")
    RETURN_NAMES = ("verified", "distance")
    FUNCTION = "verify_faces"
    CATEGORY = "DeepFace"

    def verify_faces(self, image1, image2, model_name="VGG-Face", detector="opencv",
                     distance_metric="cosine", threshold=0.6):
        try:
            # Convert images to DeepFace format
            img1 = deepface_image_from_comfy_image(image1)
            img2 = deepface_image_from_comfy_image(image2)

            # Verify faces
            result = DeepFace.verify(
                img1,
                img2,
                model_name=model_name,
                detector_backend=detector,
                distance_metric=distance_metric,
                enforce_detection=False,
                distance_threshold=threshold
            )

            verified = result.get("verified", False)
            distance = result.get("distance", 0.0)

        except Exception as e:
            print(f"Verification failed: {str(e)}")
            verified = False
            distance = float('inf')

        return (verified, distance)

class DeepFaceEmbeddingNode:
    """Node for generating face embeddings"""

    def __init__(self):
        prepare_deepface_home()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"],),
                "detector": (["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe", "yolov8", "yunet", "fastmtcnn"],),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    FUNCTION = "get_embedding"
    CATEGORY = "DeepFace"

    def get_embedding(self, image, model_name="VGG-Face", detector="opencv"):
        try:
            # Convert image to DeepFace format
            img = deepface_image_from_comfy_image(image)

            # Get embedding
            embedding_objs = DeepFace.represent(
                img,
                model_name=model_name,
                detector_backend=detector,
                enforce_detection=False
            )

            if embedding_objs:
                embedding = torch.tensor(embedding_objs[0]["embedding"])
            else:
                embedding = torch.zeros(512)  # Default size

        except Exception as e:
            print(f"Embedding generation failed: {str(e)}")
            embedding = torch.zeros(512)  # Default size

        return (embedding,)

# Node class definitions mapping
NODE_CLASS_MAPPINGS = {
    "DeepFaceAnalyze": DeepFaceAnalyzeNode,
    "DeepFaceVerify": DeepFaceVerifyNode,
    "DeepFaceEmbedding": DeepFaceEmbeddingNode
}

# Node display names
NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepFaceAnalyze": "Face Analysis",
    "DeepFaceVerify": "Face Verification",
    "DeepFaceEmbedding": "Face Embedding"
}
