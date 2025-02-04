import os
import numpy as np
import librosa
import torch
import tensorflow as tf
import requests
import folder_paths
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image
import io
import json
import socket
import asyncio

class AudioEmotionCurves:
    MODELS = {
        "SpeechEmotionRecognition": {
            "url": "https://huggingface.co/spaces/ErtugrulDemir/SpeechEmotionRecognition/resolve/main/sound_emotion_rec_model/",
            "files": [
                "fingerprint.pb",
                "keras_metadata.pb",
                "saved_model.pb",
                "variables/variables.data-00000-of-00001",
                "variables/variables.index"
            ],
            "categories": ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
        },
    }

    COLOR_SCHEME = {
        'angry': '#FF4136',
        'disgust': '#B10DC9',
        'fear': '#FF851B',
        'happy': '#FFDC00',
        'neutral': '#0074D9',
        'ps': '#2ECC40',
        'sad': '#AAAAAA'
    }

    @classmethod
    def get_available_models(cls):
        """Class method to get available models."""
        return list(cls.MODELS.keys())

    @classmethod
    def INPUT_TYPES(cls):
        """Define input types for the node."""
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (cls.get_available_models(),),
                "window_size": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "overlap": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.9, "step": 0.1}),
                "interval": ("INT", {"default": 1, "min": 1, "max": 60, "step": 1}),
                "chart_type": (["Line Chart", "Area Chart"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("curve_image", "curve_data_json")
    FUNCTION = "generate_emotion_curves"
    CATEGORY = "audio"

    def __init__(self):
        """Initialize the AudioEmotionCurves class."""
        self.loaded_model = None
        self.current_model = None

        # Configure event loop policy for Windows
        if os.name == 'nt':
            try:
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            except Exception as e:
                print(f"Warning: Could not set Windows event loop policy: {e}")

    def get_model_path(self, model_name):
        """Get the path for a model."""
        return os.path.join(folder_paths.models_dir, model_name)

    def safe_socket_shutdown(self, sock):
        """Safely shut down a socket connection."""
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except (OSError, socket.error):
            pass
        finally:
            try:
                sock.close()
            except (OSError, socket.error):
                pass

    def download_model(self, model_name):
        """Download model files if they don't exist."""
        model_info = self.MODELS[model_name]
        model_path = self.get_model_path(model_name)
        os.makedirs(model_path, exist_ok=True)
        os.makedirs(os.path.join(model_path, "variables"), exist_ok=True)

        for file in model_info["files"]:
            file_url = model_info["url"] + file
            file_path = os.path.join(model_path, file)

            if not os.path.exists(file_path):
                print(f"Downloading {file} for {model_name}...")
                try:
                    with requests.get(file_url, stream=True) as response:
                        response.raise_for_status()
                        with open(file_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                    print(f"{file} downloaded successfully.")
                except Exception as e:
                    print(f"Failed to download {file}: {str(e)}")
                    return False
            else:
                print(f"{file} already exists. Skipping download.")
        return True

    def load_model(self, model_name):
        """Load the specified model."""
        if self.loaded_model is None or self.current_model != model_name:
            model_path = self.get_model_path(model_name)
            if not all(os.path.exists(os.path.join(model_path, file)) for file in self.MODELS[model_name]["files"]):
                if not self.download_model(model_name):
                    return False
            try:
                self.loaded_model = tf.saved_model.load(model_path)
                self.current_model = model_name
                print(f"Model {model_name} loaded successfully.")
            except Exception as e:
                print(f"Error loading the model: {str(e)}")
                return False
        return True

    def extract_mfcc(self, audio_data, sr, duration=3, offset=0.5, n_mfcc=40):
        """Extract MFCC features from audio data."""
        samples = int(duration * sr)
        offset_samples = int(offset * sr)

        if len(audio_data) < offset_samples + samples:
            y = audio_data[offset_samples:]
        else:
            y = audio_data[offset_samples:offset_samples + samples]

        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T, axis=0)
        return mfcc

    def prepare_data(self, audio_data, sr):
        """Prepare audio data for emotion prediction."""
        features = self.extract_mfcc(audio_data, sr)
        features = np.array([x for x in features])
        features = np.expand_dims(features, -1)
        return features

    def predict_emotion(self, audio_data, sr, model_name):
        """Predict emotions from audio data."""
        features = self.prepare_data(audio_data, sr)
        sample = np.expand_dims(features, axis=0)

        infer = self.loaded_model.signatures["serving_default"]
        output = infer(tf.constant(sample, dtype=tf.float32))

        output_key = list(output.keys())[-1]
        preds = output[output_key].numpy()[0]

        categories = self.MODELS[model_name]["categories"]
        return dict(zip(categories, preds))

    def generate_emotion_curves(self, audio, model, window_size, overlap, interval, chart_type):
        """Generate emotion curves from audio data."""
        try:
            if not self.load_model(model):
                return (None, json.dumps({"error": f"Failed to load the emotion recognition model {model}"}))

            # Audio input validation
            if not isinstance(audio, (tuple, list)) or len(audio) != 2:
                return (None, json.dumps({"error": "Invalid audio input format"}))

            audio_data, sample_rate = audio

            # Convert torch tensor to numpy if needed
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.cpu().numpy()

            if not isinstance(audio_data, np.ndarray):
                return (None, json.dumps({"error": "Audio data must be a numpy array or torch tensor"}))

            # Process audio data
            audio_duration = len(audio_data) / sample_rate
            hop_length = int(window_size * (1 - overlap) * sample_rate)
            n_windows = int(np.ceil(audio_duration / (window_size * (1 - overlap))))

            categories = self.MODELS[model]["categories"]
            emotion_curves = {emotion: np.zeros(n_windows) for emotion in categories}

            for i in range(n_windows):
                start = i * hop_length
                end = min(start + int(window_size * sample_rate), len(audio_data))

                if start >= len(audio_data):
                    break

                window_data = audio_data[start:end]
                try:
                    emotions = self.predict_emotion(window_data, sample_rate, model)
                    for emotion, intensity in emotions.items():
                        emotion_curves[emotion][i] = intensity
                except Exception as e:
                    print(f"Warning: Error processing window {i}: {str(e)}")
                    continue

            # Generate visualization
            time_points = np.linspace(0, audio_duration, n_windows)[::interval]
            for emotion in categories:
                emotion_curves[emotion] = emotion_curves[emotion][::interval]

            fig = go.Figure()
            for emotion in categories:
                trace_kwargs = {
                    'x': time_points,
                    'y': emotion_curves[emotion],
                    'name': emotion,
                    'line': dict(width=2, color=self.COLOR_SCHEME[emotion])
                }

                if chart_type == "Area Chart":
                    trace_kwargs.update({
                        'fill': 'tozeroy',
                        'line': dict(width=1, color=self.COLOR_SCHEME[emotion])
                    })

                fig.add_trace(go.Scatter(**trace_kwargs))

            fig.update_layout(
                title="Emotion Curves Over Time",
                xaxis_title="Time (seconds)",
                yaxis_title="Emotion Intensity",
                template="plotly_white",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=50, r=50, t=80, b=50)
            )

            # Convert plot to image with error handling
            try:
                img_bytes = pio.to_image(fig, format="png", width=1000, height=600)
                img = Image.open(io.BytesIO(img_bytes))
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            except Exception as e:
                return (None, json.dumps({"error": f"Failed to generate visualization: {str(e)}"}))

            # Prepare JSON data
            json_data = {
                "time_points": time_points.tolist(),
                "emotion_curves": {emotion: curve.tolist() for emotion, curve in emotion_curves.items()}
            }

            return (img_tensor, json.dumps(json_data))

        except Exception as e:
            return (None, json.dumps({"error": f"An unexpected error occurred: {str(e)}"}))

# Node class mappings
NODE_CLASS_MAPPINGS = {
    "AudioEmotionCurves": AudioEmotionCurves
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioEmotionCurves": "Audio Emotion Curves"
}