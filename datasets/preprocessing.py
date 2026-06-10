import pandas as pd
import re
from transformers import AutoTokenizer

df = pd.read_csv("darija_translations.csv")

df = df.astype(str)

pattern = r'[-\u0600-\u06FF]'
def clean_string(input_string):
    cleaned = re.sub(pattern, '', input_string)
    return cleaned

print("Pattern loaded.")
print(clean_string("Had lmo9abala l-li ma kanetchi mخط-taṭ liha ghay"))

df = df.map(clean_string)
df['english'] = df['english'].str.lower()
df['darija'] = df['darija'].str.lower()
print("Darija text cleaned and text in lowercase.")
df.to_csv("clean_dataset.csv", index=False)
print("Saved the clean dataset.")

total_rows = len(df)

# combined corpus generator for both languages to train the shared tokenizer
def get_joint_corpus():
    for i in range(0, total_rows, 1000):
        english_batch = df['english'].iloc[i : i + 1000].tolist()
        darija_batch = df['darija'].iloc[i : i + 1000].tolist()
        yield english_batch + darija_batch

print("Training shared BPE tokenizer...")
old_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
shared_tokenizer = old_tokenizer.train_new_from_iterator(
    get_joint_corpus(),
    vocab_size=2**15 # 32k vocab
)
shared_tokenizer.save_pretrained("../../tokenizers/shared-tokenizer")

print("Applying BPE dropout for dataset tokenization...")
shared_tokenizer._tokenizer.model.dropout = 0.1

print("Creating tokenized dataframe")
df_tokens = pd.DataFrame()
df_tokens["english"] = df["english"].apply(lambda x: shared_tokenizer.encode(x))
df_tokens["darija"] = df["darija"].apply(lambda x: shared_tokenizer.encode(x))

df_tokens.to_csv("tokenized_dataset.csv", index=False)
print("Saved the tokenized dataset.")

eng_lengths = df_tokens["english"].map(len)
dar_lengths = df_tokens["darija"].map(len)

print(f"Longest english sentence : {eng_lengths.max()} tokens")
print(f"Longest darija sentence : {dar_lengths.max()} tokens")
