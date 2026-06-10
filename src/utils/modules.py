import torch
import torch.nn as nn
import torch.nn.functional as f

from math import sqrt, log

"""
B : Batch size
T : Time
D : embedding Dimension
"""
device = "cuda"

def generateCausalMask(B, n_heads, seq_len):
    mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    mask = mask.view(1, 1, seq_len, seq_len).expand(B, n_heads, seq_len, seq_len)
    res = torch.zeros((B, n_heads, seq_len, seq_len), device=device)
    res = res.masked_fill(~mask, float('-inf'))
    return res
   
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

# ---- Self Attention ----        
    
class MaskedMultiHeadSelfAttention(nn.Module):
    def __init__(self, n_heads, D, dropout):
        assert(D % n_heads == 0), "n_heads must divide the embedding dimension"
        super().__init__()
        
        self.n_heads = n_heads
        self.D = D
        
        self.layer_norm = RMSNorm(D)
        
        self.query = nn.Linear(D, D)
        self.key = nn.Linear(D, D)
        self.value = nn.Linear(D, D)
        
        self.sqrt_head_dim = sqrt(D // n_heads)
        
        self.dropout = nn.Dropout(dropout)

        self.output_projection = nn.Linear(D, D)

    def forward(self, inputs, mask):
        B = inputs.size(0)
        seq_len = inputs.size(1)
        normalized_inputs = self.layer_norm(inputs)
        
        q = self.query(normalized_inputs).view(B, seq_len, self.n_heads, self.D//self.n_heads).transpose(1, 2)
        k = self.key(normalized_inputs).view(B, seq_len, self.n_heads, self.D//self.n_heads).transpose(1, 2)
        v = self.value(normalized_inputs).view(B, seq_len, self.n_heads, self.D//self.n_heads).transpose(1, 2)
    
        answering_matrix = torch.matmul(q, k.transpose(-1, -2)) / self.sqrt_head_dim
        answering_matrix = self.dropout(answering_matrix)
        masked = answering_matrix + mask
        
        softmax_matrix = f.softmax(masked, dim=-1)
        
        value_update = torch.matmul(softmax_matrix, v).transpose(1, 2)
        
        concatenated = value_update.reshape(B, seq_len, self.D)
        
        output = self.output_projection(concatenated)
        
        return inputs + self.dropout(output) # residual
    
    
# ---- Cross Attention ----    
    
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_heads, D, dropout):
        assert(D % n_heads == 0), "n_heads must divide the embedding dimension"
        super().__init__()
        
        self.n_heads = n_heads
        self.D = D 
        
        self.layer_norm = RMSNorm(D)
        
        self.query = nn.Linear(D, D)
        self.key = nn.Linear(D, D)
        self.value = nn.Linear(D, D)
        
        self.dropout = nn.Dropout(dropout)
        
        self.sqrt_head_dim = sqrt(D // n_heads)

        self.output_projection = nn.Linear(D, D)

    def forward(self, inputs, encoder_outputs, mask):
        B = inputs.size(0)
        seq_len_src = encoder_outputs.size(1)
        seq_len_tgt = inputs.size(1)
        
        normalized_inputs = self.layer_norm(inputs)
        
        q = self.query(normalized_inputs).view(B, seq_len_tgt, self.n_heads, self.D//self.n_heads).transpose(1, 2)
        k = self.key(encoder_outputs).view(B, seq_len_src, self.n_heads, self.D//self.n_heads).transpose(1, 2)
        v = self.value(encoder_outputs).view(B, seq_len_src, self.n_heads, self.D//self.n_heads).transpose(1, 2)
        
        answering_matrix = torch.matmul(q, k.transpose(-1, -2)) / self.sqrt_head_dim
        answering_matrix = self.dropout(answering_matrix)
        masked = answering_matrix + mask
        
        softmax_matrix = f.softmax(masked, dim=-1)
        # softmax_matrix has size (B, n_heads, seq_len_tgt, seg_len_src)
        value_update = torch.matmul(softmax_matrix, v).transpose(1, 2)
        
        concatenated = value_update.reshape(B, seq_len_tgt, self.D)
        # concatenated of shape (B, seq_len_tgt, D)
        
        output = self.output_projection(concatenated)
        
        return inputs + self.dropout(output) # residual
  
  
  
# ---- Feed Forward ----
  
class FeedForward(nn.Module):
    def __init__(self, projection_dim, D, dropout=0.1):
        super().__init__()
        
        self.up_projection = nn.Linear(D, projection_dim)
        self.down_projection = nn.Linear(projection_dim, D)

        self.layer_norm = RMSNorm(D)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, inputs):        
        normalized_inputs = self.layer_norm(inputs)
        
        up = self.up_projection(normalized_inputs)
        activation = f.gelu(up)
        activation = self.dropout(activation)
        down = self.down_projection(activation)
        
        return inputs + down
    
    

# ---- Encoder and Decoder ----

class EncoderLayer(nn.Module):
    def __init__(self, n_heads, projection_dim, D, dropout):
        super().__init__()
        
        self.attention = MaskedMultiHeadSelfAttention(n_heads, D, dropout)
        self.feed_forward = FeedForward(projection_dim, D, dropout)
        
    def forward(self, inputs, mask):
        x = self.attention(inputs, mask)
        x = self.feed_forward(x)
        
        return x
        

class Encoder(nn.Module):
    def __init__(self, N, n_heads, projection_dim, D, dropout):
        super().__init__()
        
        self.layers = nn.ModuleList([EncoderLayer(n_heads, projection_dim, D, dropout) for _ in range(N)])
        self.layer_norm = RMSNorm(D)
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
            
        return self.layer_norm(x)
    

class DecoderLayer(nn.Module):
    def __init__(self, n_heads, projection_dim, D, dropout):
        super().__init__()
        
        self.self_attention = MaskedMultiHeadSelfAttention(n_heads, D, dropout)
        self.cross_attention = MultiHeadCrossAttention(n_heads, D, dropout)
        self.feed_forward = FeedForward(projection_dim, D, dropout)
    
    def forward(self, inputs, encoder_outputs, causal_mask, src_padding_mask):
        x = self.self_attention(inputs, causal_mask)
        x = self.cross_attention(x, encoder_outputs, src_padding_mask)
        x = self.feed_forward(x)
        
        return x
        

class Decoder(nn.Module):
    def __init__(self, N, n_heads, projection_dim, D, dropout):
        super().__init__()
        
        self.layers = nn.ModuleList([DecoderLayer(n_heads, projection_dim, D, dropout) for _ in range(N)])
        self.layer_norm = RMSNorm(D)
    
    def forward(self, x, encoder_outputs, causal_mask, src_padding_mask):
        for layer in self.layers:
            x = layer(x, encoder_outputs, causal_mask, src_padding_mask)
            
        return self.layer_norm(x)
    
# ---- Embeddings ----

class EmbeddingLayer(nn.Module):
    def __init__(self, N_tokens, T, D, dropout):
        super().__init__()
        self.token_embeddings = nn.Embedding(N_tokens, D)
        nn.init.normal_(self.token_embeddings.weight, mean=0, std=0.02)

        positional_embeddings = torch.zeros((T, D))
        
        position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2).float() * (-log(10000.0) / D))
        
        positional_embeddings[:, 0::2] = torch.sin(position * div_term)
        positional_embeddings[:, 1::2] = torch.cos(position * div_term)
        
        positional_embeddings = positional_embeddings.unsqueeze(0)
                
        self.register_buffer('pe', positional_embeddings)
        
        self.sqrt_D = sqrt(D)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        return self.dropout(self.token_embeddings(inputs)*self.sqrt_D + self.pe[:, :inputs.size(1)])

# ---- Full translating transformer ----

class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.N_tokens_src = config["N_tokens_src"]
        self.N_tokens_tgt = config["N_tokens_tgt"]
        
        self.encoder = Encoder(config["N_encoder_layers"], config["n_heads"], config["projection_dim"], config["D"], config["dropout"])
        self.decoder = Decoder(config["N_decoder_layers"], config["n_heads"], config["projection_dim"], config["D"], config["dropout"])
        
        self.output_linear = nn.Linear(config["D"], config["N_tokens_tgt"])
        
        # weight tying
        self.shared_embeddings = EmbeddingLayer(config["N_tokens_src"], config["T"], config["D"], config["dropout"]) 
        self.src_embeddings = self.shared_embeddings
        self.tgt_embeddings = self.shared_embeddings
        self.output_linear.weight = self.shared_embeddings.token_embeddings.weight

        self.n_heads = config["n_heads"]
        self.T = config["T"]
        
    def forward(self, src_input, tgt_input):
        B = src_input.size(0)
        seq_len_src = src_input.size(1)
        seq_len_tgt = tgt_input.size(1)
        

        mask = (src_input == 1).view(B, 1, 1, seq_len_src) # 1 is the padding token for roberta pretrained tokenizers
        src_padding_mask = torch.zeros_like(mask, dtype=torch.float32, device=device)
        src_padding_mask = src_padding_mask.masked_fill(mask, float('-inf'))
        
        tgt_causal_mask = generateCausalMask(B, self.n_heads, seq_len_tgt)
        
        embed_src = self.src_embeddings(src_input)
        embed_tgt = self.tgt_embeddings(tgt_input)
        
        encoded = self.encoder(embed_src, src_padding_mask)
        decoded = self.decoder(embed_tgt, encoded, tgt_causal_mask, src_padding_mask)
        
        logits = self.output_linear(decoded)
        # shape (B, T, N_tokens_tgt)
        
        return logits
        
        
   
