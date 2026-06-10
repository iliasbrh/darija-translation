import pandas as pd
import numpy as np
import ast
import torch

print("Loading CSV... (This may take a minute due to string-to-list conversion)")
df = pd.read_csv("tokenized_dataset.csv")

print("Converting English strings to lists...")
df['english'] = df['english'].apply(ast.literal_eval)
print("Converting Darija strings to lists...")
df['darija'] = df['darija'].apply(ast.literal_eval)

print("Filtering corrupted/long Darija sequences...")
initial_len = len(df)
df = df[df['darija'].map(len) <= 256]
df = df[df['english'].map(len) <= 256]
print(f"Dropped {initial_len - len(df)} rows. Remaining: {len(df)}")

PAD_ID = 1 # standard for roberta pretrained
MAX_LEN = 256

def pad_sequence(seq, max_len, pad_id):
    return seq + [pad_id] * (max_len - len(seq))

print(f"Applying padding (target length: {MAX_LEN})...")
df['english'] = df['english'].apply(lambda x: pad_sequence(x, MAX_LEN, PAD_ID))
df['darija'] = df['darija'].apply(lambda x: pad_sequence(x, MAX_LEN, PAD_ID))

print("Converting to np 3D array...")
final_matrix = np.zeros((len(df), 2, MAX_LEN), dtype=np.int32)

final_matrix[:, 0, :] = np.array(df['english'].tolist())
final_matrix[:, 1, :] = np.array(df['darija'].tolist())
print(f"Final tensor shape: {final_matrix.shape}")

print("Saving binary file...")

tensor_data = torch.from_numpy(final_matrix)
torch.save(tensor_data, "dataset_fixed_256.pt")
print("Process complete. Ready for dataloader.")

print("Sample check") 
print(tensor_data[0, 0, :])
