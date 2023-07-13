import pandas as pd

file_path = "/home/rokbal/Downloads/train-00000-of-00001-bfc7b63751c36ab0 (1).parquet"

df = pd.read_parquet(file_path)

print(df.head())