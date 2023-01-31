import typing
import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.preprocessors import WavReader
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer

class WavToTextModel(OnnxInferenceModel):
    def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.char_list = char_list

    def predict(self, data: np.ndarray):
        data_pred = np.expand_dims(data, axis=0)

        preds = self.model.run(None, {self.input_name: data_pred})[0]

        text = ctc_decoder(preds, self.char_list)[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm
    from mltu.configs import BaseModelConfigs
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.interactive(False)

    configs = BaseModelConfigs.load("Models/05_sound_to_text/202301221900/configs.yaml")

    model = WavToTextModel(model_path=configs.model_path, char_list=configs.vocab, force_cpu=True)

    df = pd.read_csv("Models/05_sound_to_text/202301221900/val.csv").values.tolist()

    accum_cer, accum_wer = [], []
    for wav_path, label in tqdm(df):
        
        spectrogram = WavReader.get_spectrogram(wav_path, frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length)
        WavReader.plot_raw_audio(wav_path, label)

        padded_spectrogram = np.pad(spectrogram, ((configs.max_spectrogram_length - spectrogram.shape[0], 0),(0,0)), mode='constant', constant_values=0)

        WavReader.plot_spectrogram(spectrogram, label)

        text = model.predict(padded_spectrogram)

        cer = get_cer(text, label.lower())
        wer = get_wer(text, label.lower())

        accum_cer.append(cer)
        accum_wer.append(wer)

    print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")