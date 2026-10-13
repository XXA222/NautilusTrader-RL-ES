"""Declarative CI dimensions for the central repo's existing test commands."""

CI_LAYERS={
 'pr': {'platforms':('windows','linux'),'devices':('cpu',),
        'suites':('contracts','unit','micro_golden','algorithm_smoke')},
 'nightly': {'platforms':('windows','linux'),'devices':('cpu',),
             'suites':('real_golden','short_wfo','dependency_resolver','artifact_restore','perf')},
 'gpu': {'platforms':('approved_self_hosted',),'devices':('cuda','hybrid'),
         'suites':('cuda_train','device_parity','cuda_oom','fault_stress')},
 'release': {'platforms':('windows','linux'),'devices':('cpu','cuda'),
             'suites':('gates_Q0_Q9','sbom','wheel_hash','doc_cards','repro_statement')},
}


def ci_plan(layer:str)->dict:
    if layer not in CI_LAYERS:raise ValueError('unrecognized CI layer')
    return {'layer':layer,'matrix':{k:list(v) for k,v in CI_LAYERS[layer].items()},
            'tests_owned_by':'integration_reviewer_not_L6','default_gate_status':'BLOCKED'}
