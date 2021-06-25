import torch.nn as nn
import torch.nn.functional as F


class SimpleConv_Atari(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(4, 32, 5, stride=5),
                nn.Conv2d(32, 64, 4, stride=5),
            ]
        )

        # test_out = torch.zeros(args.state_dim).unsqueeze(0)
        # for m in self.convs:
        #     test_out = F.relu(m(test_out))
        # out_size = test_out.view(1, -1).size(1)
        # print(out_size)
        out_size = 576
        self.embed_dim = args.embed_dim = out_size
        self.proj_dim = args.proj_dim
        self.fc = nn.Linear(self.embed_dim, self.proj_dim)

    def forward(self, input):
        out = input
        for m in self.convs:
            out = F.relu(m(out))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def project(self, input):
        out = input
        for m in self.convs:
            out = F.relu(m(out))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class RainbowConv_Atari(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(4, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )

        # test_out = torch.zeros(args.state_dim).unsqueeze(0)
        # for m in self.convs:
        #     test_out = F.relu(m(test_out))
        # out_size = test_out.view(1, -1).size(1)
        # print(out_size)
        out_size = 3136
        self.embed_dim = args.embed_dim = out_size
        self.proj_dim = args.proj_dim
        self.fc = nn.Linear(self.embed_dim, self.proj_dim)

    def forward(self, input):
        out = input
        for m in self.convs:
            out = F.relu(m(out))
        out = out.view(out.size(0), -1)
        return out

    def project(self, input):
        out = input
        for m in self.convs:
            out = F.relu(m(out))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def Encoder(args):
    return SimpleConv_Atari(args) if not args.rainbow_conv else RainbowConv_Atari(args)


class Critic(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.action_num = args.action_num

        self.fc_v = nn.Sequential(nn.Linear(args.embed_dim, args.critic_hs), nn.ReLU(), nn.Linear(args.critic_hs, 1))
        self.fc_a = nn.Sequential(nn.Linear(args.embed_dim, args.critic_hs), nn.ReLU(), nn.Linear(args.critic_hs, args.action_num))

    def forward(self, state):
        a = self.fc_a(state).view(state.size(0), -1)  # (b, a)
        v = self.fc_v(state).view(state.size(0), -1)  # (b, 1)
        q = v + a - a.mean(1, keepdim=True)  # (b, a)
        return q
