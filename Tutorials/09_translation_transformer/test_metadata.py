import onnx

from mltu.tokenizers import CustomTokenizer


eng_tokenizer = CustomTokenizer.load("Tutorials/09_transformers/eng_tokenizer.json")
pt_tokenizer = CustomTokenizer.load("Tutorials/09_transformers/pt_tokenizer.json") 

metadata={"pt_tokenizer": pt_tokenizer.dict(), "eng_tokenizer": eng_tokenizer.dict()}

# metadata = pt_tokenizer.dict()

onnx_model_path = "test/model.onnx"
onnx_model = onnx.load(onnx_model_path)

# Add the metadata dictionary to the model's metadata_props attribute
# metadata_props = []
for key, value in metadata.items():
    # metadata_entry = onnx.StringStringEntryProto()
    # metadata_entry.key = str(key)
    # metadata_entry.value = str(value)
    # metadata_props.append(metadata_entry)
    meta = onnx_model.metadata_props.add()
    meta.key = str(key)
    meta.value = str(value)
# onnx_model.metadata_props.extend(metadata_props)

# Save the modified ONNX model
onnx.save(onnx_model, onnx_model_path)