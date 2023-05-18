import torch
import torch.nn.functional as F
import torch.nn as nn
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat
from torchvision.models import resnet18
import torchvision
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
# Position Encoding
def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


#This is the first module after input go into our model, first unfold the image, and add the position encoding to the input
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size_W, patch_size_H, img_size_H, img_size_W ):

        super().__init__()
        self.projection = nn.Sequential(
            nn.Unfold(kernel_size=(patch_size_H,patch_size_W),stride=(patch_size_H,patch_size_W)),
            Rearrange('b e H -> b H e'),
        )
        self.img_size_H = img_size_H
        self.img_size_W = img_size_W
        self.patch_size_H = patch_size_H
        self.patch_size_W = patch_size_W

        self.pe_cnn = nn.Conv2d(2,1,1)


    def forward(self, x):

        pe = position(self.img_size_H, self.img_size_W)
        #
        # #CNN for Poisition Encoding
        pe = self.pe_cnn(pe)
        #
        x = pe + x

        #add position encoding to input
        # ped_x = x
        # ped_x = x
        # B,_,_,_ = x.shape
        # pe = [pe]*B
        # pe = torch.cat(pe,dim=0)


        # pe = torch.zeros(B, 1, 720, 1280).cuda()
        #
        # ped_x = torch.cat([x,pe],dim=1)

        # ped_x = self.channel_attention_1(ped_x)
        # ped_x = self.spatial_attention_2(ped_x)
        # print(ped_x.shape)

        out = self.projection(x)

        B,T,_ = out.shape


        # out = out.permute(0,2,1)

        out = out.view(B,T,3,self.patch_size_H,self.patch_size_W)

        # # B, T, _, W,H = out.shape
        # vis = out.detach()
        # # vis = vis.view(B*T,3,W, H)
        #
        #
        #
        #
        # for i in range(len(vis)):
        #     for j in range(256):
        #         # print(vis[i][0].shape)
        #         # vis[i][0] =vis[i][0].view(45,80,3
        #         print(vis[i][j])
        #         cv2.imshow('output', np.array(vis[i][j]))
        #         cv2.waitKey(0)

        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 最大池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP  除以16是降维系数
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # kernel_size=1
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        # 结果相加
        out = avg_out + max_out
        return self.sigmoid(out)



class Covolutional_layer(nn.Sequential):
    def __init__(self, emb_size, ):
        super().__init__(
            Rearrange('B T C H W-> (B T) C H W'),
            nn.Conv2d(3, emb_size//2, (3, 3)),
            nn.BatchNorm2d(emb_size//2),
            nn.MaxPool2d(4,4),
            nn.Conv2d(emb_size//2, emb_size, (3, 3)),
            nn.BatchNorm2d(emb_size),
            nn.MaxPool2d(2,2),
            # nn.Conv2d(emb_size , emb_size*2, (3, 3)),
            # nn.BatchNorm2d(emb_size),
            # nn.MaxPool2d(2, 2),


        )





class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 将查询、键和值融合到一个矩阵中
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x):
        # 分割num_heads中的键、查询和值
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)


        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # print('queries', values.shape)
        # 最后一个轴上求和
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim=-1) / scaling

        # print(att.shape)
        att = self.att_drop(att)
        # 在第三个轴上求和
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        # print(out.shape)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        res = x
        x = self.fn(x)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 drop_p: float = 0.1,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.1,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))

class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(6)],
                         # nn.AvgPool2d(32, 256),


                         )


class Model(nn.Module):
    def __init__(self, channels,patch_size_W, patch_size_H, img_size_H, img_size_W ):
        super().__init__()
        self.pe = PatchEmbedding(patch_size_W, patch_size_H, img_size_H, img_size_W)

        self.cnn = Covolutional_layer(channels)

        # self.cnn_R = Rearrange('B T C H W-> (B T) C H W')
        # self.cnn = nn.Sequential(*list(resnet18(pretrained=False).children())[:-2])

        self.attention_token = TransformerEncoder(6, emb_size=32)

        self.rerange = Reduce('B C H W-> B (H W) ', reduction='mean')
        self.mean_c = Reduce('B T H -> B T ', reduction='mean')

        self.c_att = ChannelAttention(channels)
        # self.s_att = SpatialAttention()

        self.mean = Reduce('B T H -> B T ', reduction='mean')
        self.linear = nn.Linear(256,256,bias=False)



    def forward(self,x):

        B,C,H,W = x.shape

        x = self.pe(x)
        _,T,_,_,_ =x.shape

        # x = self.cnn_R(x)
        x = self.cnn(x)

        # print(x.shape)


        x = x * self.c_att(x)


        x = self.rerange(x)

        _, W = x.shape
        x= x.view(B,T,W)

        x = self.attention_token(x)

        x = x.view(B,T,32)

        x = self.mean(x)

        x = self.linear(x)

        return x




if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis,flop_count_str
    a = torch.randn(1,3,720,1280).cuda()

    b = Model (16,80,45,720,1280).cuda()
    flops = FlopCountAnalysis(b,a)

    print(flop_count_str(flops))

    # def get_n_params(Model):
    #     pp=0
    #     for p in list(Model.parameters()):
    #         nn=1
    #         for s in list(p.size()):
    #             nn = nn*s
    #         pp += nn
    #     return pp
    # o = get_n_params(b)
    # print(o)