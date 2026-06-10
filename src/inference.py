import torch

device = "cuda"

def topk_sampling(logits, k=10):
    sample = torch.topk(logits, k)
    values = sample.values
    indices = sample.indices

    softmax_values = torch.softmax(values, dim=0).to(device)
    random_index = torch.multinomial(softmax_values, num_samples=1).to(device)

    return indices[random_index.item()].item()

@torch.no_grad()
def translate_sentence_topk(inference_model, src_input, src_tokenizer, tgt_tokenizer, frequency_penalty=0.6): # src_input is raw sentence text
    src_tensor = torch.tensor(src_tokenizer.encode(src_input), dtype=torch.long).to(device).unsqueeze(0)
    predicted_tokens = [tgt_tokenizer.bos_token_id]
    idx = 0

    frequencies = torch.zeros((len(tgt_tokenizer)), dtype=torch.float32, device=device)

    ended = False
    while not ended and idx < inference_model.T:
        tgt_tensor = torch.tensor(predicted_tokens, dtype=torch.long, device=device)
        tgt_tensor = tgt_tensor.unsqueeze(0)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = inference_model(src_tensor, tgt_tensor)
        logits = output[0, idx, :].float()
        logits -= frequency_penalty * frequencies

        new_token = topk_sampling(logits, k=3)
        frequencies[new_token] += 1
        predicted_tokens = predicted_tokens + [new_token]

        if new_token == tgt_tokenizer.eos_token_id:
            ended = True
        idx += 1

    predicted_text = tgt_tokenizer.decode(predicted_tokens[1:])
    print(predicted_text, flush=True)

@torch.no_grad()
def translate_sentence_beam(inference_model, src_input, src_tokenizer, tgt_tokenizer, beam_size=4, frequency_penalty=0.6):
    src_tensor = torch.tensor(src_tokenizer.encode(src_input), dtype=torch.long).to(device).unsqueeze(0)
    
    # each beam item tracks (cumulative_log_probability, token_list, frequency_tensor)
    init_frequencies = torch.zeros((len(tgt_tokenizer)), dtype=torch.float32, device=device)
    beams = [(0.0, [tgt_tokenizer.bos_token_id], init_frequencies)]
    completed_beams = []

    while len(beams) > 0:
        all_candidates = []
        
        if len(beams[0][1]) - 1 >= inference_model.T:
            break

        for log_prob, predicted_tokens, frequencies in beams:
            idx = len(predicted_tokens) - 1
            tgt_tensor = torch.tensor(predicted_tokens, dtype=torch.long, device=device).unsqueeze(0)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output = inference_model(src_tensor, tgt_tensor)
            logits = output[0, idx, :].float()
            logits -= frequency_penalty * frequencies

            log_probs = torch.log_softmax(logits, dim=-1)
            topk_probs, topk_ids = torch.topk(log_probs, beam_size)

            for i in range(beam_size):
                new_token = topk_ids[i].item()
                next_log_prob = log_prob + topk_probs[i].item()

                next_frequencies = frequencies.clone()
                next_frequencies[new_token] += 1

                all_candidates.append((next_log_prob, predicted_tokens + [new_token], next_frequencies))

        all_candidates.sort(key=lambda x: x[0], reverse=True)

        beams = []
        for candidate in all_candidates:
            if len(beams) >= beam_size:
                break
                
            next_log_prob, next_tokens, next_frequencies = candidate
            if next_tokens[-1] == tgt_tokenizer.eos_token_id:
                completed_beams.append((next_log_prob, next_tokens))
            else:
                beams.append(candidate)

        if len(completed_beams) >= beam_size:
            break

    # fallback to the current active paths if the model did not emit a natural EOS token
    if len(completed_beams) == 0:
        completed_beams = [(b[0], b[1]) for b in beams]

    completed_beams.sort(key=lambda x: x[0], reverse=True)
    best_tokens = completed_beams[0][1]

    predicted_text = tgt_tokenizer.decode(best_tokens[1:])
    print(predicted_text, flush=True)


def translate(inference_model, src_tokenizer, tgt_tokenizer):
    with open("src/sentences3.txt") as f:
        for line in f.readlines():
            print("------\n" + line[:-1], flush=True)
            translate_sentence_topk(inference_model, line[:-1].lower(), src_tokenizer, tgt_tokenizer)
        print("------", flush=True)


if __name__ == "__main__":
    model_name = "model"

    import json
    with open(f"models/{model_name}.json") as f:
        config = json.load(f)
        print(config, flush=True)

    from utils.modules import Transformer
    from transformers import AutoTokenizer

    en_tokenizer = AutoTokenizer.from_pretrained("tokenizers/shared-tokenizer")
    da_tokenizer = AutoTokenizer.from_pretrained("tokenizers/shared-tokenizer")

    model = Transformer(config)
    model = model.to(device)

    checkpoint = torch.load("models/model.pth")
    model.load_state_dict(checkpoint["model"])
    model.eval()

    translate(model, en_tokenizer, da_tokenizer)
