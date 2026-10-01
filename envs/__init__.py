"""The eight decision scenarios used in the paper.

Each scenario is a sequential evidence-accumulation task with a hidden
parameter held fixed for the whole session. See envs/README.md.
"""
from .bandit import BanditEnv
from .monty import MontyEnv
from .trust_game import TrustGameEnv
from .machine import MachineEnv
from .stag_hunt import StagHuntEnv
from .bos import BoSEnv
from .pennies import PenniesEnv
from .chicken import ChickenEnv

# Bayesian-inference family, then game-theoretic family (paper ordering).
ENVS = {
    "bandit":     BanditEnv,
    "monty":      MontyEnv,
    "trust_game": TrustGameEnv,
    "machine":    MachineEnv,
    "stag_hunt":  StagHuntEnv,
    "bos":        BoSEnv,
    "pennies":    PenniesEnv,
    "chicken":    ChickenEnv,
}

BAYESIAN = ["bandit", "monty", "trust_game", "machine"]
GAME_THEORETIC = ["stag_hunt", "bos", "pennies", "chicken"]
# Opponent's move depends on the agent's history (reactive tit-for-tat/counter).
REACTIVE = ["stag_hunt", "bos", "pennies"]


def make_env(name, rng, T, seed=None, **kwargs):
    """seed is used only by scenarios that stratify their hidden-parameter draw
    across seeds (class attr stratify_types); pass it from every call site or
    replays will not match the sweep."""
    cls = ENVS[name]
    if seed is not None and getattr(cls, "stratify_types", False):
        kwargs["seed"] = seed
    return cls(rng, T, **kwargs)
