# NeuReduce_pytorch

## Transformer

![image](https://user-images.githubusercontent.com/45285053/124846264-6b867700-dfd3-11eb-8c37-1ec3b3c55a20.png)
<caption>https://wikidocs.net/65348</caption>

## 인코더(Encoder)

```python
# Transformer의 인코더
class Encoder(nn.Module):
    def __init__(self,
                 input_dim, # 입력데이터 차원
                 hid_dim, # 임베딩레이어의 차원 -> 멀티헤드어텐션 입력 차원 (assert hid_dim % n_heads == 0)
                 n_layers, # 멀티헤드어텐션 레이어 중첩 수
                 n_heads, # 멀티헤드어텐션 레이어의 헤드
                 pf_dim, # FFNN 출력레이어 차원
                 dropout, # 임베딩레이어의 DropOut수행
                 device, # Cuda Device
                 max_length=100):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        # src = [batch size, src len]
        # src_mask = [batch size, src len]
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = [batch size, src len]
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        # src = [batch size, src len, hid dim]
        for layer in self.layers:
            src = layer(src, src_mask)
        # src = [batch size, src len, hid dim]
        return src


class EncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]
        # self attention
        _src, _ = self.self_attention(src, src, src, src_mask)
        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]
        # positionwise feedforward
        _src = self.positionwise_feedforward(src)
        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]
        return src
```

