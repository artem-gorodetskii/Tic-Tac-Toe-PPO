from unittest import TestCase

import torch

from tictactoe.config import CFG
from tictactoe.model import ActorCriticModel


class TestActorCriticModel(TestCase):
    r"""Unit tests for the ActorCriticModel class."""

    @classmethod
    def setUp(cls) -> None:
        r"""Set up test environment for all test cases."""
        cls.model = ActorCriticModel(CFG)
        cls.batch_size = 1
        cls.sample_input = torch.randint(
            0, CFG.num_states, (cls.batch_size, CFG.input_dim)
        )

    def test_forward_shapes(self) -> None:
        r"""Test that the model's forward() method returns outputs of correct shape."""
        logits, value = self.model(self.sample_input)
        self.assertEqual(
            logits.shape,
            (self.batch_size, CFG.input_dim),
            "Actor output logits should have shape (batch, board_size)",
        )
        self.assertEqual(
            value.shape,
            (self.batch_size, 1),
            "Critic output value should have shape (batch, 1)",
        )

    def test_step_tracking(self) -> None:
        r"""Test get_step(), set_step() and increment_step() methods."""
        self.assertEqual(self.model.get_step(), 0, "Initial step should be 0.")
        self.model.set_step(5)
        self.assertEqual(self.model.get_step(), 5, "Step should be set to 5.")
        self.model.increment_step()
        self.assertEqual(self.model.get_step(), 6, "Step should increment to 6.")

    def test_weight_decay_groups(self) -> None:
        r"""Test that get_weight_decay_groups() returns correct optimizer groups."""
        actor_groups, critic_groups = self.model.get_weight_decay_groups()
        expected = 2
        self.assertEqual(
            len(actor_groups),
            expected,
            "Actor groups should contain decay and no_decay.",
        )
        self.assertEqual(
            len(critic_groups),
            expected,
            "Critic groups should contain decay and no_decay.",
        )
