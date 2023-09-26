import numpy as np
import time

from mltu.tokenizers import CustomTokenizer
from mltu.inferenceModel import OnnxInferenceModel

class PtEnTranslator(OnnxInferenceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.new_inputs = self.model.get_inputs()
        self.tokenizer = CustomTokenizer.load(self.metadata["tokenizer"])
        self.detokenizer = CustomTokenizer.load(self.metadata["detokenizer"])

    def predict(self, sentence):
        start = time.time()
        tokenized_sentence = self.tokenizer.texts_to_sequences([sentence])[0]
        encoder_input = np.pad(tokenized_sentence, (0, self.tokenizer.max_length - len(tokenized_sentence)), constant_values=0).astype(np.int64)

        tokenized_results = [self.detokenizer.start_token_index]
        for index in range(self.detokenizer.max_length - 1):
            decoder_input = np.pad(tokenized_results, (0, self.detokenizer.max_length - len(tokenized_results)), constant_values=0).astype(np.int64)
            input_dict = {
                self.model._inputs_meta[0].name: np.expand_dims(encoder_input, axis=0),
                self.model._inputs_meta[1].name: np.expand_dims(decoder_input, axis=0),
            }
            preds = self.model.run(None, input_dict)[0] # preds shape (1, 206, 29110)
            pred_results = np.argmax(preds, axis=2)
            tokenized_results.append(pred_results[0][index])

            if tokenized_results[-1] == self.detokenizer.end_token_index:
                break
        
        results = self.detokenizer.detokenize([tokenized_results])
        return results[0], time.time() - start

def read_files(path):
    with open(path, "r", encoding="utf-8") as f:
        en_train_dataset = f.read().split("\n")[:-1]
    return en_train_dataset

# Path to dataset
en_validation_data_path = "Datasets/en-es/opus.en-es-dev.en"
es_validation_data_path = "Datasets/en-es/opus.en-es-dev.es"

en_validation_data = read_files(en_validation_data_path)
es_validation_data = read_files(es_validation_data_path)

# Consider only sentences with length <= 500
max_lenght = 500
val_examples = [[es_sentence, en_sentence] for es_sentence, en_sentence in zip(es_validation_data, en_validation_data) if len(es_sentence) <= max_lenght and len(en_sentence) <= max_lenght]

translator = PtEnTranslator("Models/09_translation_transformer/202308241514/model.onnx")

val_dataset = []
for es, en in val_examples:
    results, duration = translator.predict(es)
    print("Spanish:     ", es.lower())
    print("English:     ", en.lower())
    print("English pred:", results)
    print(duration)
    print()