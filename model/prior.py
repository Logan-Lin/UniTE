from model.base import *


class PriorEncoder(Encoder):
    def __init__(self, dis_feats, num_embeds, con_feats, embed_size, output_size, sampler):
        super().__init__(sampler, 'Prior-' + ''.join(map(str, dis_feats + con_feats)) + 
                         f'-e{embed_size}-o{output_size}')
        self.dis_feats = dis_feats
        self.con_feats = con_feats

        if len(dis_feats):
            assert len(dis_feats) == len(num_embeds), \
                'length of num_embeds list should be equal to the number of discrete features.'
            self.dis_embeds = nn.ModuleList([nn.Embedding(num_embed, embed_size) for num_embed in num_embeds])
        else:
            self.dis_embeds = None
        
        if len(con_feats):
            self.con_linear = nn.Sequential(nn.Linear(len(con_feats), embed_size, bias=False), 
                                            nn.BatchNorm1d(embed_size))
        else:
            self.con_linear = None
        
        self.total_embed_size = embed_size * (len(dis_feats) + int(len(con_feats) > 0))
        self.out_linear = nn.Sequential(nn.Linear(self.total_embed_size, output_size, bias=False),
                                        nn.BatchNorm1d(output_size),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(output_size, output_size, bias=False),
                                        nn.BatchNorm1d(output_size))

    def forward(self, trip, valid_len):
        x, = self.sampler(trip)

        h = []
        if self.dis_embeds is not None:
            for dis_feat, dis_embed in zip(self.dis_feats, self.dis_embeds):
                h.append(dis_embed(x[..., dis_feat].long()))
        if self.con_linear is not None:
            h.append(self.con_linear(x[..., self.con_feats]))
        h = torch.cat(h, -1)  # (B, total_embed_size)
        h = self.out_linear(h)
        return h