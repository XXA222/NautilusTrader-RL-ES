"""L6 Compute implementation only. All shared DTOs/Protocols belong to L1 contracts."""
from .device import DeviceResolver, ResolvedDevice
from .resources import ResourceQuota, GPULearnerLease, Lease
from .workers import CPUWorkerPool
from .allocation import ComputeAllocation
from .lanes import RLTrainLane, ESEvalLane
from .benchmark import benchmark, device_decision, worker_scaling_benchmark, validate_benchmark_report
from .faults import decide_fault, FaultDecision, FaultInjector

__all__ = ['DeviceResolver','ResolvedDevice','ResourceQuota','GPULearnerLease','Lease',
           'CPUWorkerPool','ComputeAllocation','RLTrainLane','ESEvalLane','benchmark','device_decision','worker_scaling_benchmark','validate_benchmark_report',
           'decide_fault','FaultDecision','FaultInjector']
