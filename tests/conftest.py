import warnings

import pytest


@pytest.fixture(autouse=True, scope="function")
def suppress_torch_warnings() -> None:
    r"""Automatically suppress specific torch user warnings during tests.

    This fixture ignores the UserWarning related to nested
    tensors and number of attention heads in encoder layers.
    """
    warnings.filterwarnings(
        "ignore",
        message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd",  # noqa
        category=UserWarning,
    )


@pytest.fixture(autouse=True, scope="function")
def suppress_ppo_agent_std_warnings() -> None:
    r"""Automatically suppress specific UserWarnings from PPO agent during tests.

    This fixture ignores warnings matching a regular expression related
    to standard deviation calculations in the `tictactoe.agents.ppo_agent` module.
    """
    warnings.filterwarnings(
        "ignore",
        message=r".*std\(\): degrees of freedom is <= 0.*",
        category=UserWarning,
        module=r"tictactoe\.agents\.ppo_agent",
    )
