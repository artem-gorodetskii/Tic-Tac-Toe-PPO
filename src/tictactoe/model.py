import itertools
import math
from typing import Dict, Iterator, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import Tensor
from torch.nn import functional as F

from tictactoe.config import Config


class Encoder(nn.Module):
    r"""Transformer encoder."""

    def __init__(
        self,
        num_states: int,
        input_dim: int,
        hid_dim: int,
        fwd_dim: int,
        nheads: int,
        nlayers: int,
    ) -> None:
        super().__init__()

        self.cell_embedding = nn.Embedding(num_states, hid_dim)

        trans_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=nheads,
            dim_feedforward=fwd_dim,
            activation="gelu",
            batch_first=True,
        )
        self.trans_encoder = nn.TransformerEncoder(
            trans_encoder_layer, num_layers=nlayers
        )
        self.fc_layer = nn.Linear(hid_dim, hid_dim // 2)

        self.register_buffer(
            "scale", torch.sqrt(torch.tensor(hid_dim, dtype=torch.float32))
        )
        pos = self._get_sinusoidal_positional_encoding(input_dim, hid_dim)
        self.register_buffer("pos_embedding", pos, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass.

        Args:
            x: board indexes.

        Returns:
            Encoded tensor.
        """
        x = self.cell_embedding(x) * self.scale + self.pos_embedding[None, :, :]
        x = self.trans_encoder(x)
        x = x.sum(dim=1)
        x = F.gelu(self.fc_layer(x))
        return x

    @torch.no_grad()
    def _get_sinusoidal_positional_encoding(
        self, seq_len: int, dim: int
    ) -> torch.Tensor:
        position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pos = torch.zeros(seq_len, dim)
        pos[:, 0::2] = torch.sin(position * div_term)
        pos[:, 1::2] = torch.cos(position * div_term)
        return pos


class ActorCriticModel(nn.Module):
    r"""Actor-critic model."""

    def __init__(self, params: Config) -> None:
        super().__init__()
        self.params = params

        self.actor_encoder = Encoder(
            num_states=self.params.num_states,
            input_dim=self.params.input_dim,
            hid_dim=self.params.actor_hid_dim,
            fwd_dim=self.params.actor_fwd_dim,
            nheads=self.params.actor_nheads,
            nlayers=self.params.actor_nlayers,
        )
        self.critic_encoder = Encoder(
            num_states=self.params.num_states,
            input_dim=self.params.input_dim,
            hid_dim=self.params.critic_hid_dim,
            fwd_dim=self.params.critic_fwd_dim,
            nheads=self.params.critic_nheads,
            nlayers=self.params.critic_nlayers,
        )
        self.actor_head = nn.Linear(
            self.params.actor_hid_dim // 2, self.params.input_dim
        )
        self.critic_head = nn.Linear(self.params.critic_hid_dim // 2, 1)

        self.register_buffer("step", torch.zeros(1, dtype=torch.long))

        self._init_parameters()

    def actor_forward(self, x: Tensor) -> Tensor:
        r"""Compute the action logits from the actor network.

        Args:
            x: board indexes.

        Returns:
            Logits for each action.
        """
        x = self.actor_encoder(x)
        logits = self.actor_head(x)
        return logits

    def critic_forward(self, x: Tensor) -> Tensor:
        r"""Compute the value estimate from the critic network.

        Args:
            x: board indexes.

        Returns:
            Estimated state value.
        """
        x = self.critic_encoder(x)
        value = self.critic_head(x)
        return value

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Forward pass of the full actor-critic model.

        Args:
            x: board indexes.

        Returns:
            Action logits and state value estimates.
        """
        logits = self.actor_forward(x)
        value = self.critic_forward(x)
        return logits, value

    def get_step(self) -> int:
        r"""Get the current training step.

        Returns:
            The current step value.
        """
        return self.step.data.item()

    def set_step(self, value: int) -> None:
        r"""Set the current training step to a specific value.

        Args:
            value: new step value.
        """
        self.step.fill_(value)

    def increment_step(self) -> None:
        r"""Increment the training step by 1."""
        self.step += 1

    def get_weight_decay_groups(
        self,
    ) -> List[Dict[str, Union[float, List[nn.Parameter]]]]:
        r"""Return parameter groups for actor and critic with weight decay settings.

        Returns:
            A list with two dictionaries (actor and critic), each containing
            parameters  split into 'decay' and 'no_decay' groups with appropriate
            weight decay values.
        """
        actor_params = {"decay": [], "no_decay": []}
        critic_params = {"decay": [], "no_decay": []}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue

            name_lower = name.lower()
            is_blacklisted = any(
                b in name_lower for b in self.params.weight_decay_blacklist
            )

            if name_lower.startswith("actor"):
                target = actor_params
            elif name_lower.startswith("critic"):
                target = critic_params
            else:
                continue

            key = "no_decay" if is_blacklisted else "decay"
            target[key].append(param)

        actor_groups = [
            {
                "params": actor_params["decay"],
                "weight_decay": self.params.actor_weight_decay,
            },
            {"params": actor_params["no_decay"], "weight_decay": 0.0},
        ]
        critic_groups = [
            {
                "params": critic_params["decay"],
                "weight_decay": self.params.critic_weight_decay,
            },
            {"params": critic_params["no_decay"], "weight_decay": 0.0},
        ]
        return actor_groups, critic_groups

    def actor_parameters(self) -> Iterator[nn.Parameter]:
        r"""Yield all trainable parameters of the actor network.

        Returns:
            An iterator over actor parameters.
        """
        return itertools.chain(
            self.actor_encoder.parameters(), self.actor_head.parameters()
        )

    def critic_parameters(self) -> Iterator[nn.Parameter]:
        r"""Yield all trainable parameters of the critic network.

        Returns:
            An iterator over critic parameters.
        """
        return itertools.chain(
            self.critic_encoder.parameters(), self.critic_head.parameters()
        )

    def _init_transformer_weights(self, module: nn.Module) -> None:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def _init_parameters(self) -> None:
        init.xavier_uniform_(self.actor_encoder.cell_embedding.weight)
        init.xavier_uniform_(self.critic_encoder.cell_embedding.weight)

        self._init_transformer_weights(self.actor_encoder)
        self._init_transformer_weights(self.critic_encoder)

        init.xavier_uniform_(self.actor_encoder.fc_layer.weight)
        init.constant_(self.actor_encoder.fc_layer.bias, 0.0)

        init.xavier_uniform_(self.critic_encoder.fc_layer.weight)
        init.constant_(self.critic_encoder.fc_layer.bias, 0.0)

        # Small init for policy head.
        init.normal_(self.actor_head.weight, mean=0.0, std=0.01)
        init.constant_(self.actor_head.bias, 0.0)

        # Xavier init for value head.
        init.xavier_uniform_(self.critic_head.weight)
        init.constant_(self.critic_head.bias, 0.0)
