"""Explicit fail-closed boundary errors for the L4 provider."""

class L4Error(RuntimeError):
    """Base L4 domain error."""

class ContractError(L4Error):
    """Frozen interface or configuration is invalid."""

class GateBlocked(L4Error):
    """A prerequisite was not signed off by the supervising run."""

class ReplayMismatch(L4Error):
    """Off-policy transition semantics do not match saved replay."""

class ArtifactError(L4Error):
    """A checkpoint or inference artifact cannot be trusted."""

class TransferBlocked(L4Error):
    """Actor transfer has failed exact mapping or parity."""

class ActorActionParityMismatch(TransferBlocked):
    """Validated weights loaded, but destination math failed action parity."""
