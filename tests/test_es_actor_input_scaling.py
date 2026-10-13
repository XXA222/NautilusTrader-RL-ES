from dataclasses import replace

import numpy as np
import pytest

from rles_es_evotorch.actor import ActorGenomeSpec, act, decode, encode, genome_hash, policy_inputs

pytestmark = pytest.mark.unit


def test_legacy_signature_and_genome_identity_are_unchanged():
    spec = ActorGenomeSpec(("cash", "signal"), (2,))
    assert spec.signature == "23c6ea74b8e1800d0a179c5097dca9733c0a50d6e3cc3c4987fe8bf5a0116cc2"
    assert genome_hash(spec, np.arange(spec.size, dtype=np.float32) / 10) == "ddcd67e7d0ca43df779be4af95d20777899166caa626e79ce337c2412120ba51"


def test_frozen_account_input_scale_retains_market_response_and_folds_into_weights():
    raw = np.asarray([[10000, -1], [10000, 1]], dtype=np.float32)
    legacy = ActorGenomeSpec(("cash", "signal"), (2,))
    scaled = replace(legacy, version=2, input_scale=(0.0001, 1.0))
    weights = {
        "layers.0.weight": np.asarray([[0.1, 1], [-0.1, 1]], dtype=np.float32),
        "layers.0.bias": np.zeros(2, dtype=np.float32),
        "layers.1.weight": np.asarray([[0.5, 0.5]], dtype=np.float32),
        "layers.1.bias": np.zeros(1, dtype=np.float32),
    }
    genome = encode(legacy, weights)
    assert act(legacy, genome, raw)[0] == act(legacy, genome, raw)[1]
    response = act(scaled, genome, raw)
    assert response[0, 0] < -0.5 and response[1, 0] > 0.5
    folded = decode(scaled, genome)
    folded["layers.0.weight"] *= np.asarray(scaled.input_scale, dtype=np.float32)
    assert np.allclose(act(legacy, encode(legacy, folded), raw), response, atol=1e-7)
    assert np.array_equal(raw[:, 0], [10000, 10000])  # Native facts were not overwritten.
    assert policy_inputs(scaled, raw)[0, 0] == 1.0
    assert scaled.signature != legacy.signature
    assert genome_hash(scaled, genome) != genome_hash(replace(scaled, input_scale=(0.001, 1.0)), genome)


@pytest.mark.parametrize("scales", [(1.0,), (0.0, 1.0), (float("nan"), 1.0), (True, 1.0), [1.0, 1.0]])
def test_scaled_actor_rejects_missing_unsafe_or_mutable_scale_vectors(scales):
    with pytest.raises(ValueError, match="input scales"):
        ActorGenomeSpec(("cash", "signal"), (2,), version=2, input_scale=scales)


def test_legacy_actor_does_not_silently_apply_a_new_scale():
    with pytest.raises(ValueError, match="identity input scaling"):
        ActorGenomeSpec(("cash", "signal"), (2,), input_scale=(0.0001, 1.0))
