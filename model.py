import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import tie_weights, weight_init, grad_false


class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(4, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )

        test_out = torch.zeros(args.state_dim).unsqueeze(0)
        for m in self.convs:
            test_out = F.relu(m(test_out))
        out_size = test_out.view(1, -1).size(1)

        self.fc = nn.Sequential(nn.Linear(out_size, 512), nn.LayerNorm(512), nn.Tanh())
        self.fc_dim = 512

    def forward(self, input):
        out = input / 255.0
        for m in self.convs:
            out = F.elu(m(out))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def copy_conv_weights_from(self, source):
        for i in range(len(self.convs)):
            tie_weights(src=source.convs[i], trg=self.convs[i])


class Critic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder = Encoder(args)
        self.fc_dim = self.encoder.fc_dim
        self.fc = nn.Sequential(nn.Linear(self.fc_dim, args.critic_hs), nn.ReLU(), nn.Linear(args.critic_hs, args.action_num))
        self.apply(weight_init)

    def forward(self, state):
        state = self.encoder(state)  # (b, s)
        q = self.fc(state).view(state.size(0), -1)  # (b, a)
        return q


class EncoderNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(4, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )
        test_out = torch.zeros(args.state_dim).unsqueeze(0)
        for m in self.convs:
            test_out = F.relu(m(test_out))
        out_size = test_out.view(1, -1).size(1)
        self.fc = nn.Linear(out_size, args.embed_os)

    def forward(self, input):
        out = input / 255.0
        for m in self.convs:
            out = F.elu(m(out))
        h = out.view(out.size(0), -1)
        out = self.fc(h)
        return out

    def copy_conv_weights_from(self, source):
        for i in range(len(self.convs)):
            tie_weights(src=source.convs[i], trg=self.convs[i])


class RndEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.encoder_net = EncoderNet(args)
        grad_false(self.encoder_net)

    def forward(self, state, cat=True):
        return self.encoder_net(state)
