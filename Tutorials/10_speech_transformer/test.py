from mltu.inferenceModel import OnnxInferenceModel
import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass
import numpy as np
import json
import tensorflow_datasets as tfds
from mltu.utils.text_utils import ctc_decoder, get_cer, get_wer

from mltu.tokenizers import CustomTokenizer

# examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)

class PtEnTranslator(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.encoder = OnnxInferenceModel(kwargs["encoder"])
        self.decoder = OnnxInferenceModel(kwargs["decoder"])
        self.tokenizer = CustomTokenizer.load(self.decoder.metadata["tokenizer"])  

        encoder_input_dict = {
            self.encoder.model._inputs_meta[0].name: np.expand_dims(spectrogram, axis=0).astype(np.float32)
        }
        encoder_output = self.encoder.model.run(None, encoder_input_dict)[0] # encoder_output shape (1, 206, 512)

        tokenized_results = [self.tokenizer.start_token_index]
        for index in range(self.tokenizer.max_length - 1):
            decoder_input = np.pad(tokenized_results, (0, self.tokenizer.max_length - len(tokenized_results)), constant_values=0).astype(np.int64)
            decoder_input_dict = {
                self.decoder.model._inputs_meta[0].name: np.expand_dims(decoder_input, axis=0).astype(np.int64),
                self.decoder.model._inputs_meta[1].name: encoder_output,
            }
            preds = self.decoder.model.run(None, decoder_input_dict)[0] # preds shape (1, 206, 29110)
            pred_results = np.argmax(preds, axis=2)
            tokenized_results.append(pred_results[0][index])

            if tokenized_results[-1] == self.tokenizer.end_token_index:
                break




        # tokenized_results = [self.tokenizer.start_token_index]
        # for index in range(self.tokenizer.max_length - 1):
        #     decoder_input = np.pad(tokenized_results, (0, self.tokenizer.max_length - len(tokenized_results)), constant_values=0).astype(np.int64)
        #     input_dict = {
        #         self.model._inputs_meta[0].name: np.expand_dims(spectrogram, axis=0).astype(np.float32),
        #         self.model._inputs_meta[1].name: np.expand_dims(decoder_input, axis=0).astype(np.int64)
        #     }
        #     preds = self.model.run(None, input_dict)[0] # preds shape (1, 206, 29110)
        #     pred_results = np.argmax(preds, axis=2)
        #     tokenized_results.append(pred_results[0][index])

        #     if tokenized_results[-1] == self.tokenizer.end_token_index:
        #         break
        
        results = self.tokenizer.detokenize([tokenized_results])
        return results[0]


translator = PtEnTranslator(
    model_path = "Models/10_speech_transformer/202307241132/model.onnx",
    encoder = "Models/10_speech_transformer/202307241132/encoder.onnx",
    decoder = "Models/10_speech_transformer/202307241132/decoder.onnx"
)
# encoder = translator.model.get_layer("encoder")

import pandas as pd
from tqdm import tqdm
from mltu.configs import BaseModelConfigs
from mltu.preprocessors import WavReader

configs = BaseModelConfigs.load("Models/10_speech_transformer/202307241132/configs.yaml")

df = pd.read_csv("Models/10_speech_transformer/202307241132/val.csv").values.tolist()

accum_cer, accum_wer = [], []
for wav_path, label in tqdm(df):
    
    spectrogram = WavReader.get_spectrogram(wav_path, frame_length=configs.frame_length, frame_step=configs.frame_step, fft_length=configs.fft_length)
    # WavReader.plot_raw_audio(wav_path, label)

    padded_spectrogram = np.pad(spectrogram, ((configs.max_spectrogram_length - spectrogram.shape[0], 0),(0,0)), mode="constant", constant_values=0)

    results = translator.predict(padded_spectrogram)

    cer = get_cer(label.lower(), results)
    wer = get_wer(label.lower(), results)

    # print(label)
    # print(results)
    # print(cer, wer)
    accum_cer.append(cer)
    accum_wer.append(wer)

print(f"Average CER: {np.average(accum_cer)}, Average WER: {np.average(accum_wer)}")
# train_examples, val_examples = examples['train'], examples['validation']

# val_dataset = []
# for pt, en in val_examples:
#     pt_sentence = pt.numpy().decode('utf-8')
#     en_sentence = en.numpy().decode('utf-8')
#     results = translator.predict(pt_sentence)
#     print(en_sentence)
#     print(results)
#     print()
    # val_dataset.append([pt.numpy().decode('utf-8'), en.numpy().decode('utf-8')])