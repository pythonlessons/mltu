import cv2
import typing
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
    import onnxruntime as ort

    # model_path = "Models/11_wav2vec2_torch/202309131152/model.onnx"
    # session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    # audio_len = 246000
    # # Prepare input data (replace 'input' with the actual input name)
    # input_data = {'input': np.random.randn(1, audio_len).astype(np.float32)}

    # # Run inference
    # output = session.run(None, input_data)

    model = Wav2vec2(model_path="Models/11_wav2vec2_torch/202309141138/model.onnx")

    # The list of multiple [audio_path, label] for validation
    val_dataset = pd.read_csv("Models/11_wav2vec2_torch/202309141138/val.csv").values.tolist()


    # model.vocab = [' ', "'", 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    audioReader = AudioReader(sample_rate=16000)


    # dataset_path = "Datasets/LJSpeech-1.1"
    # metadata_path = dataset_path + "/metadata.csv"
    # wavs_path = dataset_path + "/wavs/"

    # # Read metadata file and parse it
    # metadata_df = pd.read_csv(metadata_path, sep="|", header=None, quoting=3)
    # dataset = []
    # # vocab = [' ', "'", 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    # for file_name, transcription, normalized_transcription in metadata_df.values.tolist():
    #     path = f"Datasets/LJSpeech-1.1/wavs/{file_name}.wav"
    #     new_label = "".join([l for l in normalized_transcription.lower() if l in model.vocab])
    #     dataset.append([path, new_label])


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