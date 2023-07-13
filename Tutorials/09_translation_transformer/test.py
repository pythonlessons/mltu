from mltu.inferenceModel import OnnxInferenceModel
import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass
import numpy as np
import json
import tensorflow_datasets as tfds

from mltu.tokenizers import CustomTokenizer

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True, as_supervised=True)

class PtEnTranslator(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.new_inputs = self.model.get_inputs()
        self.pt_tokenizer = CustomTokenizer.load(self.metadata["pt_tokenizer"])
        self.eng_tokenizer = CustomTokenizer.load(self.metadata["eng_tokenizer"])
        # self.eng_tokenizer = CustomTokenizer.load("Tutorials/09_transformers/eng_tokenizer.json")
        # self.pt_tokenizer = CustomTokenizer.load("Tutorials/09_transformers/pt_tokenizer.json")  

    def predict(self, sentence):
        tokenized_sentence = self.pt_tokenizer.texts_to_sequences([sentence])[0]
        encoder_input = np.pad(tokenized_sentence, (0, self.pt_tokenizer.max_length - len(tokenized_sentence)), constant_values=0).astype(np.int64)

        tokenized_results = [self.eng_tokenizer.start_token_index]
        for index in range(self.eng_tokenizer.max_length - 1):
            decoder_input = np.pad(tokenized_results, (0, self.eng_tokenizer.max_length - len(tokenized_results)), constant_values=0).astype(np.int64)
            input_dict = {
                self.model._inputs_meta[0].name: np.expand_dims(encoder_input, axis=0),
                self.model._inputs_meta[1].name: np.expand_dims(decoder_input, axis=0),
            }
            preds = self.model.run(None, input_dict)[0] # preds shape (1, 206, 29110)
            pred_results = np.argmax(preds, axis=2)
            tokenized_results.append(pred_results[0][index])

            if tokenized_results[-1] == self.eng_tokenizer.end_token_index:
                break
        
        results = self.eng_tokenizer.detokenize([tokenized_results])
        return results[0]


translator = PtEnTranslator("Models/09_translation_transformer/202307101211/model.onnx")


train_examples, val_examples = examples['train'], examples['validation']

val_dataset = []
for pt, en in val_examples:
    pt_sentence = pt.numpy().decode('utf-8')
    en_sentence = en.numpy().decode('utf-8')
    results = translator.predict(pt_sentence)
    print(en_sentence)
    print(results)
    print()
    # val_dataset.append([pt.numpy().decode('utf-8'), en.numpy().decode('utf-8')])