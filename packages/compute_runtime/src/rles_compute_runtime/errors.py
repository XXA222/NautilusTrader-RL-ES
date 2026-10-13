"""Fail-closed compute runtime exceptions."""

class ComputeError(RuntimeError):
    """A compute invariant has been violated."""

class QuotaExceeded(ComputeError):
    """No resource lease is available; never silently oversubscribe."""

class DeviceUnavailable(ComputeError):
    """The requested concrete device cannot be used."""

class ComputeProtocolError(ComputeError):
    """A worker response does not satisfy the frozen compute contract."""

class WorkerFailure(ComputeError):
    """One or more CPU workers failed or timed out."""
