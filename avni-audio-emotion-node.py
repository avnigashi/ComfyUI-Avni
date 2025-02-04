import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    AutoConfig,
)
MODEL_CONFIGS = {
    "Hatman": {
        "name": "Hatman/audio-emotion-detection",
        "sampling_rate": 16000,
        "description": "General-purpose emotion detection model",
    },
    "XLSR": {
        "name": "harshit345/xlsr-wav2vec-speech-emotion-recognition",
        "sampling_rate": 16000,
        "description": "Cross-lingual speech emotion recognition model",
    },
    "German": {
        "name": "padmalcom/wav2vec2-large-emotion-detection-german",
        "sampling_rate": 16000,
        "description": "German-specific emotion detection model",
    }
}

class AudioEmotionAnalysis:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = {}
        self.feature_extractors = {}
        self.configs = {}

        for key, config in MODEL_CONFIGS.items():
            self.configs[key] = AutoConfig.from_pretrained(config["name"])
            self.feature_extractors[key] = Wav2Vec2FeatureExtractor.from_pretrained(config["name"])
            self.models[key] = Wav2Vec2ForSequenceClassification.from_pretrained(config["name"]).to(self.device)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "model": (list(MODEL_CONFIGS.keys()),),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("primary_emotion", "confidence_scores", "audio_stats", "model_info", "emotion_changes", "ensemble_prediction")
    FUNCTION = "analyze_emotion"
    CATEGORY = "audio/emotion"

    def get_audio_stats(self, waveform):
        """Calculate audio statistics."""
        audio_np = waveform.numpy().flatten()
        stats = {
            "duration": len(audio_np) / 44100,  # seconds
            "rms": float(np.sqrt(np.mean(audio_np**2))),
            "peak_amplitude": float(np.max(np.abs(audio_np))),
            "zero_crossings": len(np.where(np.diff(np.signbit(audio_np)))[0]),
        }
        return str(stats)

    def analyze_segments(self, waveform, model_key, segment_duration=2.0):
        """Analyze emotions in segments to track changes."""
        sample_rate = 44100
        segment_length = int(segment_duration * sample_rate)
        segments = torch.split(waveform, segment_length, dim=1)

        segment_emotions = []
        for i, segment in enumerate(segments):
            if segment.shape[1] < segment_length / 2:  # Skip very short segments
                continue

            resampled_segment = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=MODEL_CONFIGS[model_key]["sampling_rate"]
            )(segment)

            inputs = self.feature_extractors[model_key](
                resampled_segment.numpy().flatten(),
                sampling_rate=MODEL_CONFIGS[model_key]["sampling_rate"],
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.models[model_key](**inputs).logits
                scores = F.softmax(logits, dim=1).cpu().numpy()[0]

            max_emotion = self.configs[model_key].id2label[np.argmax(scores)]
            max_confidence = float(np.max(scores)) * 100
            segment_emotions.append({
                "time": f"{i*segment_duration:.1f}-{(i+1)*segment_duration:.1f}s",
                "emotion": max_emotion,
                "confidence": f"{max_confidence:.1f}%"
            })

        return str(segment_emotions)

    def ensemble_predict(self, waveform):
        """Combine predictions from all models."""
        ensemble_results = {}

        for model_key in MODEL_CONFIGS.keys():
            resampled_waveform = torchaudio.transforms.Resample(
                orig_freq=44100,
                new_freq=MODEL_CONFIGS[model_key]["sampling_rate"]
            )(waveform)

            inputs = self.feature_extractors[model_key](
                resampled_waveform.numpy().flatten(),
                sampling_rate=MODEL_CONFIGS[model_key]["sampling_rate"],
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.models[model_key](**inputs).logits
                scores = F.softmax(logits, dim=1).cpu().numpy()[0]
                max_emotion = self.configs[model_key].id2label[np.argmax(scores)]

            ensemble_results[model_key] = {
                "emotion": max_emotion,
                "confidence": float(np.max(scores)) * 100
            }

        return str(ensemble_results)

    def analyze_emotion(self, audio, model):
        if isinstance(audio, dict):
            # Handle dictionary format
            if "samples" in audio:
                audio_data = audio["samples"]
            elif "waveform" in audio:
                audio_data = audio["waveform"]
            else:
                raise ValueError(f"Audio dictionary missing required data fields: {audio.keys()}")
            waveform = torch.tensor(audio_data, dtype=torch.float32)
        else:
            # Previous working version
            waveform = torch.tensor(audio, dtype=torch.float32)

        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)

        # Get primary emotion prediction
        resampled_waveform = torchaudio.transforms.Resample(
            orig_freq=44100,
            new_freq=MODEL_CONFIGS[model]["sampling_rate"]
        )(waveform)

        inputs = self.feature_extractors[model](
            resampled_waveform.numpy().flatten(),
            sampling_rate=MODEL_CONFIGS[model]["sampling_rate"],
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.models[model](**inputs).logits
            scores = F.softmax(logits, dim=1).cpu().numpy()[0]

        detailed_scores = [
            {"Emotion": self.configs[model].id2label[i],
             "Score": f"{round(score * 100, 1)}%",
             "Raw_Logit": float(logits[0][i])}
            for i, score in enumerate(scores)
        ]

        max_emotion = max(detailed_scores, key=lambda x: float(x["Score"][:-1]))

        # Get model information
        model_info = {
            "name": MODEL_CONFIGS[model]["name"],
            "description": MODEL_CONFIGS[model]["description"],
            "available_emotions": list(self.configs[model].id2label.values()),
            "model_type": "Wav2Vec2 for Sequence Classification"
        }

        # Additional analyses
        audio_stats = self.get_audio_stats(waveform)
        emotion_changes = self.analyze_segments(waveform, model)
        ensemble_prediction = self.ensemble_predict(waveform)

        return (
            max_emotion["Emotion"],
            str(detailed_scores),
            audio_stats,
            str(model_info),
            emotion_changes,
            ensemble_prediction
        )

NODE_CLASS_MAPPINGS = {
    "AudioEmotionAnalysis": AudioEmotionAnalysis
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioEmotionAnalysis": "Audio Emotion Analysis"
}
