import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ExponentialLR

from transformers import AutoTokenizer
import pandas as pd

from utils.dataset_class import LanguageTranslationDataset
from utils.modules import Transformer
from inference import translate

from time import time

torch.cuda.empty_cache()

en_tokenizer = AutoTokenizer.from_pretrained("tokenizers/shared-tokenizer")
da_tokenizer = AutoTokenizer.from_pretrained("tokenizers/shared-tokenizer")

generator = torch.Generator().manual_seed(42)

# ---- Hyperparameters ----
config = {"N_tokens_src" : len(en_tokenizer),
          "N_tokens_tgt" : len(da_tokenizer),
          "N_encoder_layers" : 10,
          "N_decoder_layers" : 12,
          "T" : 256,
          "D" : 512,
          "n_heads" : 8,
          "projection_dim" : 4*512,
          "dropout" : 0.2}

batch_size = 128

# --------- Model ---------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Selected device : " + device, flush=True)

model = Transformer(config)

model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total amount of parameters : {total_params}\n"+"-"*30, flush=True)

# --------- Dataset --------
data = torch.load("datasets/dataset_fixed_256.pt")
dataset = LanguageTranslationDataset(data, config["T"], en_tokenizer, da_tokenizer)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [0.95, 0.05], generator=generator)

N = len(dataset)
N_train, N_val = len(train_dataset), len(val_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# --------- Training Parameters --------
epochs = 15

total_steps = epochs * (N_train // batch_size)
warmup_steps = int(0.07 * epochs * (N_train // batch_size)) # 7% warmup

steps_between_eval = 100

max_lr = 0.0004

optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.01, lr=max_lr)
criterion = nn.CrossEntropyLoss(ignore_index=da_tokenizer.pad_token_id, label_smoothing=0.1).to(device)

scheduler_warmup = LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=warmup_steps)
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_steps])
# --------------------------------------

if __name__ == "__main__":
    
    steps_done = 0
    running_loss = 0
    start_time = time()
            
    model.train()
    print("Starting training loop.")

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}", flush=True)
        for item in train_dataloader:

            optimizer.zero_grad()
                
            src_input = item["src"].to(device, non_blocking=True)
            tgt_input = item["tgt"].to(device, non_blocking=True)
            target = item["output"].to(device, non_blocking=True)
            
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(src_input, tgt_input)
                loss = criterion(outputs.view(-1, len(da_tokenizer)), target.view(-1))

            running_loss += loss.item()
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
                
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            steps_done += 1
            
            if steps_done % steps_between_eval == 0:
                print(f"Current lr : {scheduler.get_last_lr()[0]:.6f}", flush=True)
                print(f"Training loss : {(running_loss/steps_between_eval):.2f}", flush=True)
                running_loss = 0
                print("-"*70, flush=True)

        model.eval()

        translate(inference_model=model, src_tokenizer=en_tokenizer, tgt_tokenizer=da_tokenizer)
        with torch.no_grad():
            val_loss = 0
            for item in val_dataloader:
                src_input = item["src"].to(device, non_blocking=True)
                tgt_input = item["tgt"].to(device, non_blocking=True)
                target = item["output"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(src_input, tgt_input)
                    loss = criterion(outputs.view(-1, len(da_tokenizer)), target.view(-1))
                val_loss += loss.item()
                    
            print(f"Validation loss : {(val_loss/len(val_dataloader)):.2f}", flush=True)

        model.train()
        print(f"Expected total time : {((time()-start_time)*(epochs*N/(steps_done*batch_size))/3600):.2f}h", flush=True)
                
        print("-"*70, flush=True)

        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, f'models/model2_{epoch+1}.pth')
                    
    model_name = "model2"
    import json
    with open(f"models/{model_name}.json", 'w') as f:
        json.dump(config, f) 
