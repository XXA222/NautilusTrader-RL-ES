"""Pinned upstream import/device/RSSM action-isolation smoke; run in JAX env."""
from __future__ import annotations

import argparse
from io import BytesIO
import json
from pathlib import Path
import sys
import traceback
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    checkout = Path(args.checkout).resolve()
    project_package = Path(__file__).resolve().parents[2] / "packages" / "world_model" / "src"
    sys.path.insert(0, str(project_package))
    output = Path(args.output)
    result: dict[str, Any] = {
        "status": "FAILED", "upstream_commit": None, "jax_version": None,
        "devices": [], "rssm_import": False, "action_independence": False,
        "checkpoint_resume": False, "detail": None,
    }
    try:
        import subprocess
        commit = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True).stdout.strip()
        result["upstream_commit"] = commit
        if commit != "e01491fad6434b2245a3b8ca201dd7faedcc458c":
            raise RuntimeError(f"unexpected upstream commit: {commit}")
        sys.path.insert(0, str(checkout))
        import elements
        import jax
        import jax.numpy as jnp
        import ninjax as nj
        import numpy as np
        from dreamerv3.rssm import RSSM
        from rles_world_model.upstream import rssm_market_imagine, rssm_market_observe

        result["jax_version"] = jax.__version__
        result["devices"] = [str(device) for device in jax.devices()]
        if not result["devices"] or any(device.platform != "cpu" for device in jax.devices()):
            raise RuntimeError("CPU smoke requires JAX CPU devices only")
        class SmallRSSM(nj.Module):
            def __call__(self, policy_action):
                model = self.sub(
                    "rssm", RSSM,
                    act_space={"market_dummy": elements.Space(np.float32, (1,))},
                    deter=32, hidden=32, stoch=2, classes=4, blocks=4,
                    imglayers=1, obslayers=1, dynlayers=1, free_nats=0.0,
                )
                carry = model.initial(1)
                tokens = jnp.ones((1, 1, 8), dtype=jnp.float32)
                reset = jnp.zeros((1, 1), dtype=jnp.bool_)
                posterior = rssm_market_observe(
                    model, carry, tokens, reset, training=False,
                    trading_action=policy_action,
                )
                imagined = rssm_market_imagine(
                    model, posterior[0], trading_action=policy_action[:, 0, :],
                    training=False,
                )
                return posterior, imagined

        model = SmallRSSM(name="route_b_cpu_smoke")
        actions = [jnp.full((1, 1, 1), value, dtype=jnp.float32) for value in (-1.0, 0.0, 1.0)]
        params = nj.init(model)({}, actions[0], seed=20261018)
        outputs = [nj.pure(model)(params, action, seed=20261018)[1] for action in actions]
        if not all(jax.tree_util.tree_all(
                jax.tree.map(lambda a, b: jnp.array_equal(a, b), outputs[0], item))
                   for item in outputs[1:]):
            raise RuntimeError("pinned RSSM posterior changed across policy actions")
        result["rssm_import"] = True
        result["action_independence"] = True
        carry = outputs[0][0][0]
        stream = BytesIO()
        np.savez_compressed(stream,
                            deter=np.asarray(jax.device_get(carry["deter"]), dtype=np.float32),
                            stoch=np.asarray(jax.device_get(carry["stoch"]), dtype=np.float32))
        payload = stream.getvalue()
        output.parent.mkdir(parents=True, exist_ok=True)
        state_path = output.with_suffix(".npz")
        state_path.write_bytes(payload)
        with np.load(BytesIO(state_path.read_bytes()), allow_pickle=False) as checkpoint:
            restored = {name: jnp.asarray(checkpoint[name]) for name in ("deter", "stoch")}
        if not all(np.array_equal(np.asarray(carry[name]), np.asarray(restored[name]))
                   for name in restored):
            raise RuntimeError("checkpoint save/resume changed state values")
        result["checkpoint_resume"] = True
        result["status"] = "PASS"
    except Exception as exc:
        result["detail"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
