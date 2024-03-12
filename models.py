from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperForAudioClassification, \
    AutoFeatureExtractor
import torchaudio
import torch


def process_audio(url: str) -> (torch.Tensor, int):
    """
    Process steps:
        1) load audio file as torch tensor
        2) resample tensor to right proportion

    :param
        * url (str): Url of audio file

    :return
        torch.Tensor: Processed audio waveform
        int: Base frequency for whisper models
    """
    BASE_FREQ = 16_000
    waveform, sample_rate = torchaudio.load(url)

    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=BASE_FREQ)
    resampled_waveform = resampler(waveform).squeeze(0)

    return resampled_waveform, BASE_FREQ


def get_input_features(pre_model, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    return pre_model(waveform, sample_rate=sample_rate, return_tensors='pt').input_features


class AudioLangClassifier:
    """
    AudioLangClassifier Class

    This class provides methods to predict the language of input audio waveforms.

    :var
        * feature_extractor (AutoFeatureExtractor): Pretrained feature extractor for audio data.
        * model (WhisperForAudioClassification): Pretrained audio classification model.
    """
    def __init__(self):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")
        self.model = WhisperForAudioClassification.from_pretrained("sanchit-gandhi/whisper-medium-fleurs-lang-id")

    def predict(self, waveform: torch.Tensor, sample_rate: int) -> str:
        """
        Predicts the language of the input audio waveform.

        :param
            * waveform (torch.Tensor): The audio waveform tensor.
            * sample_rate (int): The sample rate of the audio waveform.

        :return
           str: The predicted language label.
        """
        input_features = get_input_features(self.feature_extractor, waveform, sample_rate)

        logits = self.model(input_features).logits
        predicted_ids = torch.argmax(logits).item()
        predicted_label = self.model.config.id2label[predicted_ids]

        return predicted_label


class Transcriptor:
    """
    Transcriptor Class

    This class provides methods to generate transcriptions for input audio waveforms.

    :var
        * processor (WhisperProcessor): Pretrained processor for whisper models.
        * model (WhisperForConditionalGeneration): Pretrained conditional generation model.
        * forced_decoder_ids (List[int]): Decoder prompt IDs for forced decoding.
    """
    def __init__(self):
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language="french", task="transcribe")

    def get_transcription(self, waveform: torch.Tensor, sample_rate: int) -> str:
        """
        Generate transcriptions for the input audio waveform.

        :param
            * waveform (torch.Tensor): The audio waveform tensor.
            * sample_rate (int): The sample rate of the audio waveform.

        :return
            str: Transcription generated for the input waveform.
        """
        input_features = get_input_features(self.processor, waveform, sample_rate)

        predicted_ids = self.model.generate(input_features, forced_decoder_ids=self.forced_decoder_ids)

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)

        return transcription[0]
