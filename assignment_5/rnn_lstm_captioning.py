import math
from typing import Optional, Tuple

import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torchvision.models import feature_extraction


def hello_rnn_lstm_captioning():
    print("Hello from rnn_lstm_captioning.py!")


class ImageEncoder(nn.Module):
    def __init__(self, pretrained: bool = True, verbose: bool = True):
        super().__init__()
        self.cnn = torchvision.models.regnet_x_400mf(pretrained=pretrained)

        self.backbone = feature_extraction.create_feature_extractor(
            self.cnn, return_nodes={"trunk_output.block4": "c5"}
        )

        dummy_out = self.backbone(torch.randn(2, 3, 224, 224))["c5"]
        self._out_channels = dummy_out.shape[1]

        if verbose:
            print("For input images in NCHW format, shape (2, 3, 224, 224)")
            print(f"Shape of output c5 features: {dummy_out.shape}")

        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @property
    def out_channels(self):
        return self._out_channels

    def forward(self, images: torch.Tensor):
        if images.dtype == torch.uint8:
            images = images.to(dtype=self.cnn.stem[0].weight.dtype)
            images /= 255.0

        images = self.normalize(images)
        features = self.backbone(images)["c5"]
        return features


def rnn_step_forward(x, prev_h, Wx, Wh, b):
    next_h = torch.tanh(x @ Wx + prev_h @ Wh + b)
    cache = (x, prev_h, Wx, Wh, next_h)
    return next_h, cache


def rnn_step_backward(dnext_h, cache):
    x, prev_h, Wx, Wh, next_h = cache
    
    dtanh = dnext_h * (1 - next_h**2)
    
    dx = dtanh @ Wx.T
    dprev_h = dtanh @ Wh.T
    dWx = x.T @ dtanh
    dWh = prev_h.T @ dtanh
    db = dtanh.sum(dim=0)
    
    return dx, dprev_h, dWx, dWh, db


def rnn_forward(x, h0, Wx, Wh, b):
    N, T, D = x.shape
    _, H = h0.shape
    
    h = torch.zeros(N, T, H, dtype=x.dtype, device=x.device)
    cache = []
    
    prev_h = h0
    for t in range(T):
        next_h, step_cache = rnn_step_forward(x[:, t, :], prev_h, Wx, Wh, b)
        h[:, t, :] = next_h
        cache.append(step_cache)
        prev_h = next_h
    
    return h, cache


def rnn_backward(dh, cache):
    N, T, H = dh.shape
    x, prev_h, Wx, Wh, next_h = cache[0]
    D = x.shape[1]
    
    dx = torch.zeros(N, T, D, dtype=dh.dtype, device=dh.device)
    dh0 = torch.zeros_like(prev_h)
    dWx = torch.zeros_like(Wx)
    dWh = torch.zeros_like(Wh)
    db = torch.zeros(H, dtype=dh.dtype, device=dh.device)
    
    dprev_h = torch.zeros_like(prev_h)
    
    for t in range(T-1, -1, -1):
        dnext_h = dh[:, t, :] + dprev_h
        dx_t, dprev_h, dWx_t, dWh_t, db_t = rnn_step_backward(dnext_h, cache[t])
        
        dx[:, t, :] = dx_t
        dWx += dWx_t
        dWh += dWh_t
        db += db_t
    
    dh0 = dprev_h
    return dx, dh0, dWx, dWh, db


class RNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, x, h0):
        hn, _ = rnn_forward(x, h0, self.Wx, self.Wh, self.b)
        return hn

    def step_forward(self, x, prev_h):
        next_h, _ = rnn_step_forward(x, prev_h, self.Wx, self.Wh, self.b)
        return next_h


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()
        self.W_embed = nn.Parameter(
            torch.randn(vocab_size, embed_size).div(math.sqrt(vocab_size))
        )

    def forward(self, x):
        out = self.W_embed[x]
        return out


def temporal_softmax_loss(x, y, ignore_index=None):
    loss = F.cross_entropy(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1), 
                            ignore_index=ignore_index, reduction='sum') / x.size(0)
    return loss


class CaptioningRNN(nn.Module):
    def __init__(
        self,
        word_to_idx,
        input_dim: int = 512,
        wordvec_dim: int = 128,
        hidden_dim: int = 128,
        cell_type: str = "rnn",
        image_encoder_pretrained: bool = True,
        ignore_index: Optional[int] = None,
    ):
        super().__init__()
        if cell_type not in {"rnn", "lstm", "attn"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)
        self.ignore_index = ignore_index

        self.image_encoder = ImageEncoder(pretrained=image_encoder_pretrained, verbose=False)
        self.word_embedding = WordEmbedding(vocab_size, wordvec_dim)
        
        if cell_type == "rnn":
            self.rnn = RNN(wordvec_dim, hidden_dim)
        elif cell_type == "lstm":
            self.rnn = LSTM(wordvec_dim, hidden_dim)
        elif cell_type == "attn":
            self.rnn = AttentionLSTM(wordvec_dim, hidden_dim)
        
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        
        if cell_type == "attn":
            self.feature_projection = nn.Linear(self.image_encoder.out_channels, hidden_dim)
        else:
            self.feature_projection = nn.Linear(self.image_encoder.out_channels, hidden_dim)

    def forward(self, images, captions):
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        features = self.image_encoder(images)
        
        if self.cell_type == "attn":
            N, C, H, W = features.shape
            features_proj = self.feature_projection(features.permute(0, 2, 3, 1))
            A = features_proj.permute(0, 3, 1, 2)
        else:
            features_pooled = features.mean(dim=(2, 3))
            h0 = self.feature_projection(features_pooled)
        
        word_embeddings = self.word_embedding(captions_in)
        
        if self.cell_type == "attn":
            hidden_states = self.rnn(word_embeddings, A)
        else:
            hidden_states = self.rnn(word_embeddings, h0)
        
        scores = self.output_projection(hidden_states)
        
        loss = temporal_softmax_loss(scores, captions_out, ignore_index=self.ignore_index)
        
        return loss

    def sample(self, images, max_length=15):
        N = images.shape[0]
        captions = self._null * images.new(N, max_length).fill_(1).long()

        if self.cell_type == "attn":
            attn_weights_all = images.new(N, max_length, 4, 4).fill_(0).float()

        features = self.image_encoder(images)
        
        if self.cell_type == "attn":
            features_proj = self.feature_projection(features.permute(0, 2, 3, 1))
            A = features_proj.permute(0, 3, 1, 2)
            prev_h = A.mean(dim=(2, 3))
            prev_c = prev_h
        else:
            features_pooled = features.mean(dim=(2, 3))
            prev_h = self.feature_projection(features_pooled)
            if self.cell_type == "lstm":
                prev_c = torch.zeros_like(prev_h)
        
        prev_word = torch.full((N,), self._start, dtype=torch.long, device=images.device)
        
        for t in range(max_length):
            word_embed = self.word_embedding(prev_word)
            
            if self.cell_type == "rnn":
                next_h = self.rnn.step_forward(word_embed, prev_h)
                prev_h = next_h
                scores = self.output_projection(next_h)
            elif self.cell_type == "lstm":
                next_h, next_c = self.rnn.step_forward(word_embed, prev_h, prev_c)
                prev_h, prev_c = next_h, next_c
                scores = self.output_projection(next_h)
            elif self.cell_type == "attn":
                attn, attn_weights = dot_product_attention(prev_h, A)
                next_h, next_c = self.rnn.step_forward(word_embed, prev_h, prev_c, attn)
                prev_h, prev_c = next_h, next_c
                scores = self.output_projection(next_h)
                attn_weights_all[:, t, :, :] = attn_weights
            
            next_word = scores.argmax(dim=1)
            captions[:, t] = next_word
            prev_word = next_word

        if self.cell_type == "attn":
            return captions, attn_weights_all.cpu()
        else:
            return captions


class LSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
        self, x: torch.Tensor, prev_h: torch.Tensor, prev_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H = prev_h.shape[1]
        a = x @ self.Wx + prev_h @ self.Wh + self.b
        
        i = torch.sigmoid(a[:, :H])
        f = torch.sigmoid(a[:, H:2*H])
        o = torch.sigmoid(a[:, 2*H:3*H])
        g = torch.tanh(a[:, 3*H:])
        
        next_c = f * prev_c + i * g
        next_h = o * torch.tanh(next_c)
        
        return next_h, next_c

    def forward(self, x: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        c0 = torch.zeros_like(h0)
        
        N, T, D = x.shape
        H = h0.shape[1]
        
        hn = torch.zeros(N, T, H, dtype=x.dtype, device=x.device)
        
        prev_h, prev_c = h0, c0
        for t in range(T):
            next_h, next_c = self.step_forward(x[:, t, :], prev_h, prev_c)
            hn[:, t, :] = next_h
            prev_h, prev_c = next_h, next_c
        
        return hn


def dot_product_attention(prev_h, A):
    N, H, D_a, _ = A.shape
    
    A_flat = A.view(N, H, -1).transpose(1, 2)
    scores = torch.bmm(A_flat, prev_h.unsqueeze(2)).squeeze(2)
    attn_weights = F.softmax(scores / math.sqrt(H), dim=1)
    attn = torch.bmm(attn_weights.unsqueeze(1), A_flat).squeeze(1)
    attn_weights = attn_weights.view(N, D_a, D_a)
    
    return attn, attn_weights


class AttentionLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()

        self.Wx = nn.Parameter(
            torch.randn(input_dim, hidden_dim * 4).div(math.sqrt(input_dim))
        )
        self.Wh = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.Wattn = nn.Parameter(
            torch.randn(hidden_dim, hidden_dim * 4).div(math.sqrt(hidden_dim))
        )
        self.b = nn.Parameter(torch.zeros(hidden_dim * 4))

    def step_forward(
        self,
        x: torch.Tensor,
        prev_h: torch.Tensor,
        prev_c: torch.Tensor,
        attn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        H = prev_h.shape[1]
        a = x @ self.Wx + prev_h @ self.Wh + attn @ self.Wattn + self.b
        
        i = torch.sigmoid(a[:, :H])
        f = torch.sigmoid(a[:, H:2*H])
        o = torch.sigmoid(a[:, 2*H:3*H])
        g = torch.tanh(a[:, 3*H:])
        
        next_c = f * prev_c + i * g
        next_h = o * torch.tanh(next_c)
        
        return next_h, next_c

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        h0 = A.mean(dim=(2, 3))
        c0 = h0
        
        N, T, D = x.shape
        H = h0.shape[1]
        
        hn = torch.zeros(N, T, H, dtype=x.dtype, device=x.device)
        
        prev_h, prev_c = h0, c0
        for t in range(T):
            attn, _ = dot_product_attention(prev_h, A)
            next_h, next_c = self.step_forward(x[:, t, :], prev_h, prev_c, attn)
            hn[:, t, :] = next_h
            prev_h, prev_c = next_h, next_c
        
        return hn