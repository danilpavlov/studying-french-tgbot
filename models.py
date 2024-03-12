from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperForAudioClassification, \
    AutoFeatureExtractor
import torchaudio
import torch


def process_audio(url: str):
    """
    Process steps:
        1) load audio file as torch tensor
        2) resample tensor to right proportion

    :param  * url: Url of audio file

    :return * resample_waveform: processed audio waveform
            * BASE_FREQ: base frequency for whisper models
    """
    BASE_FREQ = 16_000
    waveform, sample_rate = torchaudio.load(url)

    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=BASE_FREQ)
    resampled_waveform = resampler(waveform).squeeze(0)

    return resampled_waveform, BASE_FREQ


def get_input_features(pre_model, waveform, sample_rate):
    return pre_model(waveform, sample_rate=sample_rate, return_tensors='pt').input_features


class AudioLangClassifier:
    def __init__(self):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
        self.model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")

    def predict(self, waveform, sample_rate):
        input_features = get_input_features(self.feature_extractor, waveform, sample_rate)

        logits = self.model(input_features).logits
        predicted_ids = torch.argmax(logits).item()
        predicted_label = self.model.config.id2label[predicted_ids]

        return predicted_label


class Transcriptor:
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="french", task="transcribe")

    def get_transcription(self, waveform, sample_rate):
        input_features = get_input_features(self.processor, waveform, sample_rate)

        predicted_ids = self.model.generate(input_features, forced_decoder_ids=self.forced_decoder_ids)

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return transcription
