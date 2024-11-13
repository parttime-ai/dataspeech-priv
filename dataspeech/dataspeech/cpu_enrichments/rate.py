import logging

from phonemizer.backend import EspeakBackend
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

backend = EspeakBackend('de', preserve_punctuation=True, with_stress=True)

def rate_apply(batch, rank=None, audio_column_name="audio", text_column_name="text"):
    if isinstance(batch[text_column_name], list):
        speaking_rates = []
        phonemes_list = []
        texts = []
        if "speech_duration" in batch:
            for text, audio_duration in zip(batch[text_column_name], batch["speech_duration"]):
                phonemes = backend.phonemize([text], strip=True)[0]
                audio_duration = audio_duration if audio_duration != 0 else 0.01
                speaking_rate = len(phonemes) / audio_duration
                speaking_rates.append(speaking_rate)
                phonemes_list.append(phonemes)
                texts.append(text)
        else:
            for text, audio in zip(batch[text_column_name], batch[audio_column_name]):
                phonemes = backend.phonemize([text], strip=True)[0]

                sample_rate = audio["sampling_rate"]
                audio_length = len(audio["array"].squeeze()) / sample_rate

                speaking_rate = len(phonemes) / audio_length


                speaking_rates.append(speaking_rate)
                phonemes_list.append(phonemes)
                texts.append(text)

        logger.debug("Phonemes: %s for text: %s", phonemes_list, ' '.join(texts))

        batch["speaking_rate"] = speaking_rates
        batch["phonemes"] = phonemes_list
    else:
        phonemes = backend.phonemize(batch[text_column_name], strip=True)[0]
        if "speech_duration" in batch:
            audio_length = batch["speech_duration"] if batch["speech_duration"] != 0 else 0.01
        else:
            sample_rate = batch[audio_column_name]["sampling_rate"]
            audio_length = len(batch[audio_column_name]["array"].squeeze()) / sample_rate

        speaking_rate = len(phonemes) / audio_length

        batch["speaking_rate"] = speaking_rate
        batch["phonemes"] = phonemes

    return batch