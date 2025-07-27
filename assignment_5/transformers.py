import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
import math


def hello_transformers():
    print("Hello from transformers.py!")


def generate_token_dict(vocab):
    token_dict = {}
    for i, token in enumerate(vocab):
        token_dict[token] = i
    return token_dict


def prepocess_input_sequence(
    input_str: str, token_dict: dict, spc_tokens: list
) -> list:
    out = []
    tokens = input_str.split()
    
    for token in tokens:
        if token in spc_tokens:
            out.append(token_dict[token])
        else:
            for digit in token:
                out.append(token_dict[digit])
    
    return out


def scaled_dot_product_two_loop_single(
    query: Tensor, key: Tensor, value: Tensor
) -> Tensor:
    K, M = query.shape
    out = torch.zeros_like(query)
    
    for i in range(K):
        scores = torch.zeros(K)
        for j in range(K):
            scores[j] = torch.dot(query[i], key[j]) / math.sqrt(M)
        
        weights = F.softmax(scores, dim=0)
        out[i] = torch.sum(weights.unsqueeze(1) * value, dim=0)
    
    return out


def scaled_dot_product_two_loop_batch(
    query: Tensor, key: Tensor, value: Tensor
) -> Tensor:
    N, K, M = query.shape
    out = torch.zeros_like(query)
    
    for n in range(N):
        scores = torch.bmm(query[n:n+1], key[n:n+1].transpose(1, 2)).squeeze(0) / math.sqrt(M)
        weights = F.softmax(scores, dim=1)
        out[n] = torch.bmm(weights.unsqueeze(0), value[n:n+1]).squeeze(0)
    
    return out


def scaled_dot_product_no_loop_batch(
    query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
) -> Tensor:
    _, _, M = query.shape
    
    scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(M)
    
    if mask is not None:
        mask = mask.to(scores.device)
        scores = scores.masked_fill(mask, -1e9)
    
    weights_softmax = F.softmax(scores, dim=2)
    y = torch.bmm(weights_softmax, value)
    
    return y, weights_softmax


class SelfAttention(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_v: int):
        super().__init__()
        
        c_q = math.sqrt(6 / (dim_in + dim_q))
        c_v = math.sqrt(6 / (dim_in + dim_v))
        
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_q)
        self.v = nn.Linear(dim_in, dim_v)
        
        nn.init.uniform_(self.q.weight, -c_q, c_q)
        nn.init.uniform_(self.k.weight, -c_q, c_q)
        nn.init.uniform_(self.v.weight, -c_v, c_v)
        
        self.weights_softmax = None

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
    ) -> Tensor:
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        
        y, self.weights_softmax = scaled_dot_product_no_loop_batch(q, k, v, mask)
        
        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_out: int):
        super().__init__()
        
        self.heads = nn.ModuleList([
            SelfAttention(dim_in, dim_out, dim_out) for _ in range(num_heads)
        ])
        
        c = math.sqrt(6 / (num_heads * dim_out + dim_in))
        self.linear = nn.Linear(num_heads * dim_out, dim_in)
        nn.init.uniform_(self.linear.weight, -c, c)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
    ) -> Tensor:
        head_outputs = [head(query, key, value, mask) for head in self.heads]
        concatenated = torch.cat(head_outputs, dim=-1)
        y = self.linear(concatenated)
        
        return y


class LayerNormalization(nn.Module):
    def __init__(self, emb_dim: int, epsilon: float = 1e-10):
        super().__init__()
        
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = torch.sqrt(var + self.epsilon)
        y = self.gamma * (x - mean) / std + self.beta
        
        return y


class FeedForwardBlock(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim_feedforward: int):
        super().__init__()
        
        c1 = math.sqrt(6 / (inp_dim + hidden_dim_feedforward))
        c2 = math.sqrt(6 / (hidden_dim_feedforward + inp_dim))
        
        self.linear1 = nn.Linear(inp_dim, hidden_dim_feedforward)
        self.linear2 = nn.Linear(hidden_dim_feedforward, inp_dim)
        
        nn.init.uniform_(self.linear1.weight, -c1, c1)
        nn.init.uniform_(self.linear2.weight, -c2, c2)

    def forward(self, x):
        y = self.linear2(F.relu(self.linear1(x)))
        return y


class EncoderBlock(nn.Module):
    def __init__(
        self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float
    ):
        super().__init__()
        
        if emb_dim % num_heads != 0:
            raise ValueError(
                f"The value emb_dim = {emb_dim} is not divisible by num_heads = {num_heads}. Please select an appropriate value."
            )
        
        dim_head = emb_dim // num_heads
        
        self.multi_head_attention = MultiHeadAttention(num_heads, emb_dim, dim_head)
        self.feed_forward = FeedForwardBlock(emb_dim, feedforward_dim)
        self.norm1 = LayerNormalization(emb_dim)
        self.norm2 = LayerNormalization(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out1 = self.multi_head_attention(x, x, x)
        out2 = self.dropout(self.norm1(out1 + x))
        out3 = self.feed_forward(out2)
        y = self.dropout(self.norm2(out3 + out2))
        
        return y


def get_subsequent_mask(seq):
    N, K = seq.shape
    mask = torch.triu(torch.ones(K, K), diagonal=1).bool()
    mask = mask.unsqueeze(0).expand(N, -1, -1)
    
    return mask


class DecoderBlock(nn.Module):
    def __init__(
        self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float
    ):
        super().__init__()
        
        if emb_dim % num_heads != 0:
            raise ValueError(
                f"The value emb_dim = {emb_dim} is not divisible by num_heads = {num_heads}. Please select an appropriate value."
            )
        
        dim_head = emb_dim // num_heads
        
        self.attention_self = MultiHeadAttention(num_heads, emb_dim, dim_head)
        self.attention_cross = MultiHeadAttention(num_heads, emb_dim, dim_head)
        self.feed_forward = FeedForwardBlock(emb_dim, feedforward_dim)
        self.norm1 = LayerNormalization(emb_dim)
        self.norm2 = LayerNormalization(emb_dim)
        self.norm3 = LayerNormalization(emb_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, dec_inp: Tensor, enc_inp: Tensor, mask: Tensor = None
    ) -> Tensor:
        out1 = self.attention_self(dec_inp, dec_inp, dec_inp, mask)
        out2 = self.dropout(self.norm1(dec_inp + out1))
        out3 = self.attention_cross(out2, enc_inp, enc_inp)
        out4 = self.dropout(self.norm2(out2 + out3))
        out5 = self.feed_forward(out4)
        y = self.dropout(self.norm3(out4 + out5))
        
        return y


class Encoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src_seq: Tensor):
        for _layer in self.layers:
            src_seq = _layer(src_seq)

        return src_seq


class Decoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        num_layers: int,
        dropout: float,
        vocab_len: int,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList(
            [
                DecoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.proj_to_vocab = nn.Linear(emb_dim, vocab_len)
        a = (6 / (emb_dim + vocab_len)) ** 0.5
        nn.init.uniform_(self.proj_to_vocab.weight, -a, a)

    def forward(self, target_seq: Tensor, enc_out: Tensor, mask: Tensor):
        out = target_seq.clone()
        for _layer in self.layers:
            out = _layer(out, enc_out, mask)
        out = self.proj_to_vocab(out)
        return out


def position_encoding_simple(K: int, M: int) -> Tensor:
    pos = torch.arange(K).float() / K
    y = pos.unsqueeze(1).repeat(1, M).unsqueeze(0)
    return y


def position_encoding_sinusoid(K: int, M: int) -> Tensor:
    pos = torch.arange(K).float().unsqueeze(1)
    div_term = torch.exp(torch.arange(0, M, 2).float() * -(math.log(10000.0) / M))
    
    y = torch.zeros(K, M)
    sin_indices = torch.arange(0, M, 2)
    cos_indices = torch.arange(1, M, 2)

    y[:, sin_indices] = torch.sin(pos * div_term[:len(sin_indices)])
    y[:, cos_indices] = torch.cos(pos * div_term[:len(cos_indices)])

    return y.unsqueeze(0)


class Transformer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        dropout: float,
        num_enc_layers: int,
        num_dec_layers: int,
        vocab_len: int,
    ):
        super().__init__()
        
        self.emb_layer = nn.Embedding(vocab_len, emb_dim)
        
        self.encoder = Encoder(
            num_heads, emb_dim, feedforward_dim, num_enc_layers, dropout
        )
        self.decoder = Decoder(
            num_heads,
            emb_dim,
            feedforward_dim,
            num_dec_layers,
            dropout,
            vocab_len,
        )

    def forward(
        self, ques_b: Tensor, ques_pos: Tensor, ans_b: Tensor, ans_pos: Tensor
    ) -> Tensor:
        q_emb = self.emb_layer(ques_b)
        a_emb = self.emb_layer(ans_b)
        q_emb_inp = q_emb + ques_pos
        a_emb_inp = a_emb[:, :-1] + ans_pos[:, :-1]
        
        enc_out = self.encoder(q_emb_inp)
        mask = get_subsequent_mask(ans_b[:, :-1])
        dec_out = self.decoder(a_emb_inp, enc_out, mask)
        
        return dec_out.view(-1, dec_out.size(-1))


class AddSubDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_seqs,
        target_seqs,
        convert_str_to_tokens,
        special_tokens,
        emb_dim,
        pos_encode,
    ):
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.convert_str_to_tokens = convert_str_to_tokens
        self.emb_dim = emb_dim
        self.special_tokens = special_tokens
        self.pos_encode = pos_encode

    def preprocess(self, inp):
        return prepocess_input_sequence(
            inp, self.convert_str_to_tokens, self.special_tokens
        )

    def __getitem__(self, idx):
        inp = self.input_seqs[idx]
        out = self.target_seqs[idx]
        preprocess_inp = torch.tensor(self.preprocess(inp))
        preprocess_out = torch.tensor(self.preprocess(out))
        inp_pos = len(preprocess_inp)
        inp_pos_enc = self.pos_encode(inp_pos, self.emb_dim)
        out_pos = len(preprocess_out)
        out_pos_enc = self.pos_encode(out_pos, self.emb_dim)

        return preprocess_inp, inp_pos_enc[0], preprocess_out, out_pos_enc[0]

    def __len__(self):
        return len(self.input_seqs)


def LabelSmoothingLoss(pred, ground):
    ground = ground.contiguous().view(-1)
    eps = 0.1
    n_class = pred.size(1)
    one_hot = torch.nn.functional.one_hot(ground).to(pred.dtype)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.sum()
    return loss


def CrossEntropyLoss(pred, ground):
    loss = F.cross_entropy(pred, ground, reduction="sum")
    return loss