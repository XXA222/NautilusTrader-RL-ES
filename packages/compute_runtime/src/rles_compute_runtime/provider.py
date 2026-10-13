"""Minimal entry-point discoverable compute provider; L1 contracts remain external."""
from . import ComputeAllocation, CPUWorkerPool, DeviceResolver, ESEvalLane, GPULearnerLease, RLTrainLane, ResourceQuota
from rles_contracts.registry import ProviderDescriptor


class LocalComputeProvider:
    """Return implementation classes without defining competing shared protocols."""

    def __init__(self):
        self.device_resolver = DeviceResolver
        self.quota = ResourceQuota
        self.gpu_lease = GPULearnerLease
        self.cpu_pool = CPUWorkerPool
        self.compute_allocation = ComputeAllocation
        self.rl_lane = RLTrainLane
        self.es_lane = ESEvalLane


def _create_compute(settings):
    if settings:
        raise ValueError("local compute provider takes no construction settings")
    return LocalComputeProvider()


provider = ProviderDescriptor(kind="compute", contract_version="0.4.0", factory=_create_compute)
