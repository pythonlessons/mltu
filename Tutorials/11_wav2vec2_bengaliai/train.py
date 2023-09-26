# https://www.kaggle.com/code/heyytanay/pytorch-training-wav2vec2-for-bengaliai?scriptVersionId=138448878
import os
import pandas as pd

import torch
from torch import nn
from transformers import Wav2Vec2ForCTC
import torch.nn.functional as F

from mltu.torch.model import Model
from mltu.torch.losses import CTCLoss
from mltu.torch.dataProvider import DataProvider
from mltu.torch.metrics import CERMetric, WERMetric
from mltu.torch.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Model2onnx, WarmupCosineDecay
from mltu.augmentors import RandomAudioNoise, RandomAudioPitchShift, RandomAudioTimeStretch

from mltu.preprocessors import AudioReader
from mltu.transformers import LabelIndexer, LabelPadding, AudioPadding
from tqdm import tqdm

from configs import ModelConfigs

configs = ModelConfigs()

metadata_path = "/home/rokbal/Downloads/bengaliai-speech/bengaliai-speech/train.csv"
train_mp3s_path = "/home/rokbal/Downloads/bengaliai-speech/bengaliai-speech/train_mp3s"

metadata_df = pd.read_csv(metadata_path, header=None)



from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

audioReader = AudioReader(sample_rate=16000)
def process_file(data):
    id, sentence, split = data

    mp3_file_path = f"{train_mp3s_path}/{id}.mp3"

    if not os.path.exists(mp3_file_path):
        return None
    
    audio, label = audioReader(mp3_file_path, sentence)
    if len(audio) > configs.max_audio_length or len(label) > configs.max_label_length:
        return None
    
    return [mp3_file_path, label, split]

# Assuming you have a list of file-label pairs in train_dataset
dataset = metadata_df.values.tolist()# [:100000]
with ThreadPoolExecutor(max_workers=int(os.cpu_count()/2)) as executor:  # You can adjust the number of threads as needed
    results = tqdm(executor.map(process_file, dataset), total=len(dataset))

vocab = []
train_dataset, val_dataset = [], []
for result in list(results):
    if result is None:
        continue

    mp3_file_path, label, split = result
    vocab += list(label)
    if split == "train":
        train_dataset.append([mp3_file_path, label])
    else:
        val_dataset.append([mp3_file_path, label])

vocab = sorted(list(set(vocab)))
configs.vocab = vocab
# configs.save()

print(len(train_dataset), len(val_dataset), configs.vocab)

# Create a data provider for the dataset
train_dataProvider = DataProvider(
    dataset=train_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        AudioReader(sample_rate=16000),
        ],
    transformers=[
        LabelIndexer(vocab),
        LabelPadding(max_word_length=configs.max_label_length, padding_value=len(vocab)),
        ],
    use_cache=False,
    batch_postprocessors=[
        AudioPadding(max_audio_length=configs.max_audio_length, padding_value=0, use_on_batch=True)
    ],
    augmentors = [
        RandomAudioNoise(), 
        RandomAudioPitchShift(), 
        RandomAudioTimeStretch()
    ],
    use_multiprocessing=True,
    max_queue_size=10,
    workers=64,
)

test_dataProvider = DataProvider(
    dataset=val_dataset,
    skip_validation=True,
    batch_size=configs.batch_size,
    data_preprocessors=[
        AudioReader(sample_rate=16000),
        ],
    transformers=[
        LabelIndexer(vocab),
        LabelPadding(max_word_length=configs.max_label_length, padding_value=len(vocab)),
        ],
    use_cache=False,
    batch_postprocessors=[
        AudioPadding(max_audio_length=configs.max_audio_length, padding_value=0, use_on_batch=True, limit=False)
    ],
    use_multiprocessing=True,
    max_queue_size=5,
    workers=32,
)

vocab = sorted(vocab)
configs.vocab = vocab
configs.save()


class CustomWav2Vec2Model(nn.Module):
    def __init__(self, hidden_states, dropout_rate=0.2, **kwargs):
        super(CustomWav2Vec2Model, self).__init__( **kwargs)
        pretrained_name = "facebook/wav2vec2-base-960h"
        # pretrained_name = "arijitx/wav2vec2-xls-r-300m-bengali"
        # self.model = Wav2Vec2ForCTC.from_pretrained(pretrained_name).wav2vec2
        self.model = Wav2Vec2ForCTC.from_pretrained(pretrained_name, vocab_size=hidden_states, ignore_mismatched_sizes=True)
        self.model.freeze_feature_encoder() # https://huggingface.co/blog/fine-tune-wav2vec2-english
        # self.model.freeze_feature_encoder()
        # self.dropout = nn.Dropout(p=dropout_rate)
        # self.linear = nn.Linear(self.model.config.hidden_size, hidden_states)

    def forward(self, inputs):
        output = self.model(inputs, attention_mask=None).logits
        # output = self.model(inputs, attention_mask=None).last_hidden_state
        # Apply dropout
        # output = self.dropout(output)
        # Apply linear layer
        # output = self.linear(output)
        # Apply softmax
        output = F.log_softmax(output, -1)
        return output

custom_model = CustomWav2Vec2Model(hidden_states = len(configs.vocab)+1)

# load weights from pretrained model
custom_model.load_state_dict(torch.load("Models/11_wav2vec2_bengaliai/202309242229/model.pt"))

# put on cuda device if available
if torch.cuda.is_available():
    custom_model = custom_model.cuda()

# create callbacks
warmupCosineDecay = WarmupCosineDecay(
    lr_after_warmup=configs.lr_after_warmup,
    warmup_epochs=configs.warmup_epochs,
    decay_epochs=configs.decay_epochs,
    final_lr=configs.final_lr,
    initial_lr=configs.init_lr,
    warmup_steps=len(train_dataProvider),
    verbose=True,
)
tb_callback = TensorBoard(configs.model_path + "/logs")
earlyStopping = EarlyStopping(monitor="val_CER", patience=16, mode="min", verbose=1)
modelCheckpoint = ModelCheckpoint(configs.model_path + "/model.pt", monitor="val_CER", mode="min", save_best_only=True, verbose=1)
model2onnx = Model2onnx(
    saved_model_path=configs.model_path + "/model.pt",
    input_shape=(1, configs.max_audio_length), 
    verbose=1,
    metadata={"vocab": configs.vocab},
    dynamic_axes={"input": {0: "batch_size", 1: "sequence_length"}, "output": {0: "batch_size", 1: "sequence_length"}}
)

# create model object that will handle training and testing of the network
model = Model(
    custom_model, 
    loss = CTCLoss(blank=len(configs.vocab), zero_infinity=True),
    # optimizer = torch.optim.AdamW(custom_model.parameters(), lr=configs.init_lr), # weight_decay=1e-5),
    # optimizer = torch.optim.AdamW(custom_model.parameters(), lr=configs.init_lr, weight_decay=configs.weight_decay),
    optimizer = torch.optim.AdamW(custom_model.parameters(), lr=configs.lr_after_warmup, weight_decay=configs.weight_decay),
    metrics=[
        CERMetric(configs.vocab), 
        WERMetric(configs.vocab)
    ],
    mixed_precision=configs.mixed_precision,
)

# Save training and validation datasets as csv files
train_dataProvider.to_csv(os.path.join(configs.model_path, "train.csv"))
test_dataProvider.to_csv(os.path.join(configs.model_path, "val.csv"))

model.fit(
    train_dataProvider, 
    test_dataProvider, 
    epochs=configs.train_epochs, 
    callbacks=[
        # warmupCosineDecay, 
        tb_callback, 
        earlyStopping,
        modelCheckpoint, 
        model2onnx
    ]
)