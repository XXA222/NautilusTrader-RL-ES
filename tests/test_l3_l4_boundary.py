from __future__ import annotations

import numpy as np
import pytest
from decimal import Decimal
from types import SimpleNamespace

from rles_gym_bridge.env import ContractNotReady, NautilusBarrierEnv
from rles_gym_bridge.episode import EpisodeReplay
from rles_nautilus_adapter.session import EngineBoundaryError, VisibleBatch

pytestmark = pytest.mark.contracts


class BoundaryEpisode:
    def __init__(self) -> None:
        self.fingerprint = "a" * 64
        self.token = "c" * 64

    def transition_fingerprint_hash(self) -> str:
        return self.fingerprint

    def checkpoint_boundary_token(self) -> str:
        return self.token

    def export_replay(self) -> EpisodeReplay:
        return EpisodeReplay(self.fingerprint, (Decimal("0.25"), Decimal("-1")), self.token)


def _env_with_episode(episode: object) -> NautilusBarrierEnv:
    env = object.__new__(NautilusBarrierEnv)
    env._episode = episode
    env._transition_hash = None
    env._last_observation = np.asarray([1.0, 2.0], dtype=np.float32)
    return env


def test_l3_exposes_dynamic_l4_transition_and_restore_boundary() -> None:
    episode = BoundaryEpisode()
    env = _env_with_episode(episode)
    assert env.rles_transition_hash == "a" * 64
    assert env.rles_boundary_token == "c" * 64
    observation = env.rles_boundary_observation
    assert observation.dtype == np.float32
    assert observation.tolist() == [1.0, 2.0]
    observation[0] = 999.0
    assert env.rles_boundary_observation.tolist() == [1.0, 2.0]

    episode.fingerprint = "b" * 64
    episode.token = "d" * 64
    assert env.rles_transition_hash == "b" * 64
    assert env.rles_boundary_token == "d" * 64


def test_l3_exports_json_safe_episode_replay_only_at_matching_boundary() -> None:
    episode = BoundaryEpisode()
    env = _env_with_episode(episode)

    replay = env.rles_export_replay()

    assert replay == {
        "schema": "rles.episode-replay.v1",
        "transition_hash": "a" * 64,
        "actions": ["0.25", "-1"],
        "boundary_token": "c" * 64,
    }
    assert EpisodeReplay.from_dict(replay).to_dict() == replay

    scheduled = EpisodeReplay("a" * 64, (Decimal("0.5"),), "c" * 64, 7, 19)
    assert scheduled.to_dict() == {
        "schema": "rles.episode-replay.v2",
        "transition_hash": "a" * 64,
        "actions": ["0.5"],
        "boundary_token": "c" * 64,
        "episode_index": 7,
        "episode_start_offset": 19,
    }
    assert EpisodeReplay.from_dict(scheduled.to_dict()) == scheduled

    with pytest.raises(ValueError, match="decimal strings"):
        EpisodeReplay.from_dict({**replay, "actions": [0.25]})

    class MismatchedEpisode(BoundaryEpisode):
        def export_replay(self) -> EpisodeReplay:
            return EpisodeReplay("b" * 64, (), "c" * 64)

    with pytest.raises(ContractNotReady, match="does not match"):
        _env_with_episode(MismatchedEpisode()).rles_export_replay()


@pytest.mark.parametrize(
    ("fingerprint", "token"),
    (("not-a-hash", "c" * 64), ("a" * 64, "")),
)
def test_l3_rejects_invalid_l4_boundary_proof(fingerprint: str, token: str) -> None:
    episode = BoundaryEpisode()
    episode.fingerprint = fingerprint
    episode.token = token
    env = _env_with_episode(episode)
    with pytest.raises(ContractNotReady):
        _ = env.rles_transition_hash if fingerprint == "not-a-hash" else env.rles_boundary_token


def test_l3_rejects_episode_without_checkpoint_protocol() -> None:
    env = _env_with_episode(object())
    with pytest.raises(ContractNotReady, match="transition_fingerprint_hash"):
        _ = env.rles_transition_hash


def test_l3_visible_batch_requires_causal_time_and_sha256_provenance() -> None:
    event = SimpleNamespace(
        event_id="market-event-1", source_hash="a" * 64,
        ts_event_ns=100, ts_init_ns=101, available_at_ns=101,
    )
    assert VisibleBatch(101, (event,), True).timestamp_ns == 101

    event.available_at_ns = 100
    with pytest.raises(EngineBoundaryError, match="not proven visible"):
        VisibleBatch(101, (event,), True)

    event.available_at_ns = 101
    event.source_hash = "not-a-sha256"
    with pytest.raises(EngineBoundaryError, match="provenance missing"):
        VisibleBatch(101, (event,), True)
