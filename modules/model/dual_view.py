from modules.model.base import *
from modules.model.transformer import PositionalEncoding


class TimeEmbedding(nn.Module):
    def __init__(self, output_size, embed_size=12):
        super().__init__()
        self.embed_size = embed_size

        self.week_embedding = nn.Embedding(366 // 7 + 2, embed_size)
        self.day_embedding = nn.Embedding(366 + 2, embed_size)
        self.hour_embedding = nn.Embedding(24 + 2, embed_size)
        self.minute_embedding = nn.Embedding(60 + 2, embed_size)
        self.second_embedding = nn.Embedding(60 + 2, embed_size)

        self.out_linear = nn.Linear(embed_size, output_size)

    def forward(self, x):
        week = self.week_embedding(x[:, :, 0])
        day = self.day_embedding(x[:, :, 1])
        hour = self.hour_embedding(x[:, :, 2])
        minute = self.minute_embedding(x[:, :, 3])
        second = self.second_embedding(x[:, :, 4])

        time_embed = week + day + hour + minute + second

        return self.out_linear(time_embed)


class UserEmbedding(nn.Module):
    def __init__(self, output_size, num_users, embed_size=12):
        super().__init__()
        self.embed_size = embed_size
        self.embedding = nn.Embedding(num_users, embed_size)
        self.output_linear = nn.Linear(embed_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.output_linear(x)
        return x


class RoadEmbedding(nn.Module):
    def __init__(self, output_size, embed_size, road_col, embed_type='road'):
        super(RoadEmbedding, self).__init__()

        assert embed_type in ['quadkey', 'road'], "embed_type should be 'quadkey' or 'road'."
        self.embed_type = embed_type
        self.road_col = road_col

        if embed_type == 'quadkey':
            self.road_embeds = nn.Linear(embed_size, output_size)
        else:
            self.road_embeds = nn.Embedding(embed_size, output_size)

    def forward(self, x):
        if self.embed_type == 'quadkey':
            return self.road_embeds(x)
        else:
            return self.road_embeds(x[:, :, self.road_col].long())


class DualViewEncoder(Encoder):
    """
    Dual-View Encoder used in PreCLN.
    Yan, Bingqi, et al. "PreCLN: Pretrained-based contrastive learning network for vehicle trajectory prediction." World Wide Web 26.4 (2023): 1853-1875.
    """
    def __init__(self, d_model, output_size, sampler,
                 road_col, embed_type, embed_para, num_users,
                 num_heads=8, num_layers=3, hidden_size=128):
        super().__init__(sampler, 'DualViewEncoder-' + str(road_col) +
                         f'-d{d_model}-h{hidden_size}-l{num_layers}-h{num_heads}-o{output_size}')
        self.d_model = d_model
        self.output_size = output_size
        self.road_col = road_col
        self.embed_type = embed_type

        self.pos_encode = PositionalEncoding(d_model, max_len=2001)

        self.road_embeds = RoadEmbedding(d_model, embed_para, road_col, embed_type)
        self.time_embeds = TimeEmbedding(d_model)
        self.user_embeds = UserEmbedding(d_model, num_users)

        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads, hidden_size, dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.out_linear = nn.Sequential(nn.Linear(d_model, output_size, bias=False),
                                        nn.LayerNorm(output_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(output_size, output_size))

    def forward(self, trip, valid_len, user_info, time_feas, **kwargs):
        B, L, E_in = trip.shape

        src_mask = repeat(torch.arange(end=L, device=trip.device),
                          'L -> B L', B=B) >= repeat(valid_len, 'B -> B L', L=L)  # (B, L)
        # x, src_mask, time_feas = self.sampler(trip, src_mask, time_feas)
        x, src_mask, time_feas = self.sampler(trip, src_mask, time_feas)

        h = torch.zeros(B, x.size(1), self.d_model).to(x.device)
        h += self.pos_encode(h)
        h += self.road_embeds(x)
        h += self.time_embeds(time_feas)
        h += repeat(self.user_embeds(user_info), 'B E -> B L E', L=x.size(1))

        memory = self.encoder(h, src_key_padding_mask=src_mask)
        memory = torch.nan_to_num(memory)
        memory = memory.mean(1)
        memory = self.out_linear(memory)  # (B, E_out) or (B, L, E_out)
        return memory
