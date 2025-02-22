# -*- coding: UTF-8 -*- #
"""
@filename:encoder.py
@auther:Rtnow
@time:2025-01-29
"""
import torch
import torch.nn as nn
import utils
from torchvision.models import resnet


def _get_out_shape(in_shape, layers, attn=False):
    x = torch.randn(*in_shape).unsqueeze(0)
    if attn:
        return layers(x, x, x).squeeze(0).shape
    else:
        return layers(x).squeeze(0).shape

class NormalizeImg(nn.Module):
    def __init__(self, mean_zero=False):
        super().__init__()
        self.mean_zero = mean_zero

    def forward(self, x):
        if self.mean_zero:
            return x/255. - 0.5
        return x/255.


class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Identity(nn.Module):
    def __init__(self, obs_shape=None, out_dim=None):
        super().__init__()
        self.out_shape = obs_shape
        self.out_dim = out_dim

    def forward(self, x):
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activate = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(utils.weight_init)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activate(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_query = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        self.in_channels = in_channels

    def forward(self, query, key, value):
        N, C, H, W = query.shape
        assert query.shape == key.shape == value.shape, "Key, query and value inputs must be of the same dimensions in this implementation"
        q = self.conv_query(query).reshape(N, C, H*W)#.permute(0, 2, 1)
        k = self.conv_key(key).reshape(N, C, H*W)#.permute(0, 2, 1)
        v = self.conv_value(value).reshape(N, C, H*W)#.permute(0, 2, 1)
        attention = k.transpose(1, 2)@q / C**0.5
        attention = attention.softmax(dim=1)
        output = v@attention
        output = output.reshape(N, C, H, W)
        return query + output # Add with query and output


class AttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=1, qkv_bias=False, qk_scale=None, norm_layer=nn.LayerNorm, contextualReasoning=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.attn = SelfAttention(dim[0])
        self.context = contextualReasoning
        temp_shape = _get_out_shape(dim, self.attn, attn=True)
        self.out_shape = _get_out_shape(temp_shape, nn.Flatten())
        self.apply(utils.weight_init)

    def forward(self, query, key, value):
        x = self.attn(self.norm1(query), self.norm2(key), self.norm3(value))
        if self.context:
            return x
        else:
            x = x.flatten(start_dim=1)
            return x


class SharedCNN(nn.Module):
    def __init__(self, obs_shape, num_layers=11, num_filters=32, mean_zero=False):
        super().__init__()
        print(f"obs_shape: {obs_shape}, length: {len(obs_shape)}")
        assert len(obs_shape) == 3
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.model = resnet.resnet18()
        self.model.conv1 = nn.Conv2d(obs_shape[0], self.model.conv1.out_channels, kernel_size=7, stride=2, padding=1,
                                     bias=False)
        # self.model = self.model.cuda()

        self.layers = [NormalizeImg(mean_zero),
                       self.model.conv1,
                       self.model.bn1,
                       self.model.relu,
                       self.model.maxpool,
                       self.model.layer1,
                       self.model.layer2
                       ]
        # self.layers = [NormalizeImg(mean_zero), nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        # for _ in range(1, num_layers):
        #     self.layers.append(nn.ReLU())
        #     self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(obs_shape, self.layers)
        self.apply(utils.weight_init)

    def forward(self, x):
        return self.layers(x)


class HeadCNN(nn.Module):
    def __init__(self, in_shape, num_layers=0, num_filters=32, flatten=True):
        super().__init__()
        self.layers = []
        for _ in range(0, num_layers):
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))
        if flatten:
            self.layers.append(Flatten())
        self.layers = nn.Sequential(*self.layers)
        self.out_shape = _get_out_shape(in_shape, self.layers)
        self.apply(utils.weight_init)

    def forward(self, x):
        return self.layers(x)


class Integrator(nn.Module):
    def __init__(self, in_shape_1, in_shape_2, num_filters=32, concatenate=True):
        super().__init__()
        self.relu = nn.ReLU()
        if concatenate:
            self.conv1 = nn.Conv2d(in_shape_1[0]+in_shape_2[0], num_filters, (1,1))
            self.out_shape = _get_out_shape(in_shape_1[0]+in_shape_2[0], self.conv1)
        else:
            self.conv1 = nn.Conv2d(in_shape_1[0], num_filters, (1,1))
            self.out_shape = _get_out_shape(in_shape_1, self.conv1)
        self.apply(utils.weight_init)

    def forward(self, x):
        x = self.conv1(self.relu(x))
        return x


class MultiViewEncoder(nn.Module):
    def __init__(self, third_shared_cnn, ego_shared_cnn, integrator,
                 head_cnn, projection, attention1=None, attention2=None,
                 mlp1=None, mlp2=None, norm1=None, norm2=None,
                 contextualReasoning1=False,
                 contextualReasoning2=False):
        super().__init__()
        self.third_shared_cnn = third_shared_cnn
        self.ego_shared_cnn = ego_shared_cnn
        self.integrator = integrator
        self.head_cnn = head_cnn
        self.projection = projection
        self.relu = nn.ReLU()
        self.contextualReasoning1 = contextualReasoning1
        self.contextualReasoning2 = contextualReasoning2
        self.attention1 = attention1
        self.attention2 = attention2

        self.mlp1 = mlp1
        self.norm1 = norm1
        self.mlp2 = mlp2
        self.norm2 = norm2

        self.out_dim = projection.out_dim
        # self.concatenate = concatenate

    def forward(self, x1, x2, detach=False):

        x1 = self.third_shared_cnn(x1)  # 3rd Person
        x2 = self.ego_shared_cnn(x2)
        # x = None
        x = x1

        B, C, H, W = x1.shape

        if self.contextualReasoning1:
            x1 = self.attention1(x1, x2, x2)  # Contextual reasoning on 3rd person image based on 1st person image
            x1 = self.norm1(x1)
            x1 = x1.view(B, C, -1).permute(0, 2, 1)
            x1 = self.mlp1(x1).permute(0, 2, 1).contiguous().view(B, C, H, W)
            # x1 = x1.view(B, C, -1).permute(0, 2, 1).view(-1, C)
            # x1 = self.mlp1(x1).view(B, -1, C).permute(0, 2, 1).contiguous().view(B, C, H, W)
            x = x1 + x

        if self.contextualReasoning2:
            x2 = self.attention2(x2, x1, x1)  # Contextual reasoning on 1st person image based on 3rd person image
            x2 = self.norm2(x2)
            x2 = x2.view(B, C, -1).permute(0, 2, 1)
            x2 = self.mlp2(x2).permute(0, 2, 1).contiguous().view(B, C, H, W)
            x = x2 + x

        # if self.concatenate:
        #     # Concatenate features along channel dimension
        #     x = torch.cat((x1, x2), dim=1)  # 1, 64, 21, 21
        # else:
        #     x = x1 + x2  # 1, 32, 21, 21

        x = self.integrator(x)
        x = self.head_cnn(x)

        if detach:
            x = x.detach()

        x = self.projection(x)

        return x

class OneEncoder(nn.Module):
    def __init__(self, third_cnn, head, norm, projection, attention=None):
        super().__init__()
        self.shared_cnn = third_cnn
        self.head_cnn = head
        self.norm = norm
        self.projection = projection
        self.attention = attention
        self.out_dim = projection.out_dim

    def forward(self, x, y, detach=False):
        x = self.shared_cnn(x)
        x = self.norm(x)
        x = self.head_cnn(x)
        if detach:
            x = x.detach()
        x = self.projection(x)
        return x


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(
            self,
            obs_shape,
            num_shared_layers=11,
            num_head_layers=0,
            num_filters=32,
            mean_zero=False,
            context_third=1,
            context_ego=0,
            attention=True
    ):
        super().__init__()

        self.context_third = bool(context_third)
        self.context_ego = bool(context_ego)
        self.attention = attention

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.obs_shape = list(self.obs_shape)

        self.third_cnn = SharedCNN(
            self.obs_shape, num_shared_layers, num_filters, mean_zero
        )
        self.ego_cnn = SharedCNN(
            self.obs_shape, num_shared_layers, num_filters, mean_zero
        )

        assert self.third_cnn.out_shape == self.ego_cnn.out_shape, 'Image features must be the same'
        self.CNN_out_shape = self.third_cnn.out_shape
        mlp_hidden_dim = int(self.CNN_out_shape[0] * 4)

        if self.attention:
            attention1 = None
            attention2 = None
            mlp1, mlp2 = None, None
            norm1, norm2 = None, None

            assert self.context_third or self.context_ego
            if self.context_third:
                attention1 = AttentionBlock(dim=self.CNN_out_shape, contextualReasoning=self.context_third)
                mlp1 = Mlp(in_features=self.CNN_out_shape[0], hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
                norm1 = nn.LayerNorm(self.CNN_out_shape)
            if self.context_ego:
                attention2 = AttentionBlock(dim=self.CNN_out_shape, contextualReasoning=self.context_ego)
                mlp2 = Mlp(in_features=self.CNN_out_shape[0], hidden_features=mlp_hidden_dim, act_layer=nn.GELU)
                norm2 = nn.LayerNorm(self.CNN_out_shape)

            integrator = Integrator(
                self.CNN_out_shape, self.CNN_out_shape, num_filters,
                concatenate=False
            )  # Change channel dimensions of concatenated features

            head = HeadCNN(
                integrator.out_shape, num_head_layers, num_filters,
                flatten=True
            )

            self.forward_encoder = MultiViewEncoder(
                self.third_cnn,
                self.ego_cnn,
                integrator,
                head,
                Identity(out_dim=head.out_shape[0]),
                attention1,
                attention2,
                mlp1,
                mlp2,
                norm1,
                norm2,
                # concatenate=self.concatenate,
                contextualReasoning1=self.context_third,
                contextualReasoning2=self.context_ego
            ).cuda()
        else:
            head = HeadCNN(
                self.CNN_out_shape, num_head_layers, self.CNN_out_shape[0],
                flatten=True
            )
            norm = nn.LayerNorm(self.CNN_out_shape)
            self.forward_encoder = OneEncoder(
                self.third_cnn,
                head,
                norm,
                Identity(out_dim=head.out_shape[0])
            ).cuda()

        self.feature_dim = head.out_shape[0]

    def forward(self, fix_obs, ego_obs, detach=False):
        out = self.forward_encoder(fix_obs, ego_obs)
        if detach:
            out = out.detach()
        return out


class IdentityEncoder(nn.Module):  # TODO
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters,*args):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder, 'identity': IdentityEncoder}


def make_encoder(
    encoder_type,
    obs_shape,
    num_shared_layers,
    num_head_layers,
    num_filters,
    context_third,
    context_ego,
    mean_zero=False,
    attention=False
):
    assert encoder_type in _AVAILABLE_ENCODERS

    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape,
        num_shared_layers,
        num_head_layers,
        num_filters,
        mean_zero,
        context_third,
        context_ego,
        attention
    )
