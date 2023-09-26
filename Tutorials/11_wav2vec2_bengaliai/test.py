import numpy as np

from mltu.inferenceModel import OnnxInferenceModel
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer
from mltu.preprocessors import AudioReader

class Wav2vec2(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def predict(self, audio: np.ndarray):

        audio = np.expand_dims(audio, axis=0).astype(np.float32)

        preds = self.model.run(None, {self.input_name: audio})[0]

        text = ctc_decoder(preds, self.metadata["vocab"])[0]

        return text

if __name__ == "__main__":
    import pandas as pd
    from tqdm import tqdm

    model = Wav2vec2(model_path="Models/11_wav2vec2_bengaliai/202309242229/model.onnx")

    # The list of multiple [audio_path, label] for validation
    val_dataset = pd.read_csv("Models/11_wav2vec2_bengaliai/202309252213/val.csv").values.tolist()

    audioReader = AudioReader(sample_rate=16000)

    accum_cer, accum_wer = [], []
    pbar = tqdm(val_dataset)
    for vaw_path, label in pbar:

        audio, label = audioReader(vaw_path, label)

        prediction_text = model.predict(audio)

        cer = get_cer(prediction_text, label)
        wer = get_wer(prediction_text, label)

        accum_cer.append(cer)
        accum_wer.append(wer)

        pbar.set_description(f"Average CER: {np.average(accum_cer):.4f}, Average WER: {np.average(accum_wer):.4f}")