"""Block-bootstrap, DSR and CSCV/PBO computations with declared trial accounting."""
from __future__ import annotations

from collections.abc import Callable,Sequence
from itertools import combinations
from math import comb, exp, log, sqrt, isfinite
import random
import hashlib
import json
import statistics as st
from .numeric import finite_real



class StatisticsError(ValueError):
    pass


def _finite_real(value: object) -> bool:
    return finite_real(value)


def block_bootstrap(returns: Sequence[float], *, block_length: int, draws: int,
                    seed: int, confidence: float, statistic: Callable[[list[float]],float]) -> dict:
    if (type(block_length) is not int or type(draws) is not int or type(seed) is not int or
            not 0 <= seed < 2**64 or not _finite_real(confidence) or
            not callable(statistic) or len(returns) < 4 or
            not 1 <= block_length <= len(returns) or draws < 100 or not 0 < confidence < 1):
        raise StatisticsError('invalid predeclared bootstrap parameters')
    vals=list(returns)
    if any(not _finite_real(x) for x in vals):raise StatisticsError('nonfinite or non-real returns')
    rng=random.Random(seed);n=len(vals);out: list[float] = []
    for _ in range(draws):
        res: list[float] = []
        while len(res)<n:
            start=rng.randrange(n)
            res.extend(vals[(start+i)%n] for i in range(block_length))
        score=statistic(res[:n])
        if not _finite_real(score):raise StatisticsError('undefined bootstrap statistic')
        out.append(score)
    out.sort();left=(1-confidence)/2
    q=lambda prob:out[min(draws-1,max(0,int(prob*(draws-1))))]
    return {'lower':q(left),'upper':q(1-left),'confidence':confidence,'draws':draws,
            'block_length':block_length,'seed':seed,'observations':n,
            'method':'circular_moving_block_percentile_v1'}


def deflated_sharpe(returns: Sequence[float], *, effective_trials: int,
                    trial_sharpes: Sequence[float], risk_free_per_period: float=0) -> dict:
    """Bailey/Lopez de Prado DSR approximation; requires complete predeclared trials."""
    if type(effective_trials) is not int or effective_trials < 1 or len(trial_sharpes) != effective_trials:
        raise StatisticsError('DSR requires an audited Sharpe for every declared trial')
    return _deflated_sharpe_from_sample(
        returns, effective_trials=effective_trials, trial_sharpes=trial_sharpes,
        risk_free_per_period=risk_free_per_period,
    )


def deflated_sharpe_trial_count_sensitivity(
        returns: Sequence[float], *, trial_sharpes: Sequence[float],
        risk_free_per_period: float=0) -> dict:
    """Bracket current-protocol DSR under perfect-dependence and independence scenarios.

    The endpoints are scenarios, not an estimate of cross-candidate correlation
    or project-history trial count. The empirical candidate Sharpe spread is
    held fixed while the expected-maximum trial-count input varies.
    """
    count = len(trial_sharpes)
    if count < 1:
        raise StatisticsError('DSR sensitivity requires observed candidate Sharpes')
    cases = []
    for name, effective_trials in (
        ('perfect_dependence_lower_bound', 1),
        ('independent_candidates_upper_bound', count),
    ):
        result = _deflated_sharpe_from_sample(
            returns, effective_trials=effective_trials, trial_sharpes=trial_sharpes,
            risk_free_per_period=risk_free_per_period,
        )
        cases.append({
            'scenario': name,
            'effective_trials': effective_trials,
            'dsr_probability': result['dsr_probability'],
            'expected_max_sharpe': result['expected_max_sharpe'],
        })
    probabilities = [case['dsr_probability'] for case in cases]
    return {
        'status': 'scenario_sensitivity_only',
        'effective_trial_count_range': [1, count],
        'dsr_probability_range': [min(probabilities), max(probabilities)],
        'candidate_sharpe_sample_count': count,
        'scenarios': cases,
        'correlation_estimated': False,
        'project_historical_trials': 'not_estimated',
        'method': 'dsr_effective_trial_count_endpoint_sensitivity_v1',
    }


def _deflated_sharpe_from_sample(returns: Sequence[float], *, effective_trials: int,
                                 trial_sharpes: Sequence[float],
                                 risk_free_per_period: float) -> dict:
    vals=list(returns)
    if len(vals)<4 or type(effective_trials) is not int or effective_trials<1 or len(trial_sharpes)<1:
        raise StatisticsError('DSR requires >=4 returns and audited trial Sharpe sample')
    if (not _finite_real(risk_free_per_period)
        or any(not _finite_real(x) for x in (*vals,*trial_sharpes))):
        raise StatisticsError('nonfinite Sharpe inputs')
    n=len(vals)
    sd=st.stdev(vals)
    if sd<=0:raise StatisticsError('undefined Sharpe for constant returns')
    mean=st.fmean(vals);sr=(mean-risk_free_per_period)/sd
    centered=[(v-mean)/sd for v in vals]
    skew=(n/((n-1)*(n-2)))*sum(x**3 for x in centered)
    excess=(n*(n+1)/((n-1)*(n-2)*(n-3))*sum(x**4 for x in centered)
            -3*(n-1)**2/((n-2)*(n-3))) if n>3 else 0
    kurtosis=excess+3
    sigma_sr=st.pstdev(trial_sharpes)
    if effective_trials==1:
        expected_max=0.0
    else:
        normal=st.NormalDist();gamma=0.5772156649015329
        expected_max=sigma_sr*((1-gamma)*normal.inv_cdf(1-1/effective_trials)+
                     gamma*normal.inv_cdf(1-1/(effective_trials*exp(1))))
    variance_term=1-skew*sr+((kurtosis-1)/4)*sr*sr
    if variance_term<=0:raise StatisticsError('DSR asymptotic variance nonpositive')
    z=(sr-expected_max)*sqrt(n-1)/sqrt(variance_term)
    return {'dsr_probability':st.NormalDist().cdf(z),'observed_sharpe_per_period':sr,
            'expected_max_sharpe':expected_max,'effective_trials':effective_trials,
            'trial_sharpe_std':sigma_sr,'skew':skew,'kurtosis':kurtosis,'sample_size':n}


def _sharpe(values):
    if len(values)<2:return float('-inf')
    sd=st.stdev(values)
    return st.fmean(values)/sd if sd>0 else float('-inf')


def candidate_set_sha256(candidate_ids: Sequence[str]) -> str:
    """Frozen ordered candidate identities, independent of performance outcomes."""
    if (not candidate_ids or any(not isinstance(i,str) or not i or len(i)>160 for i in candidate_ids)
        or len(set(candidate_ids)) != len(candidate_ids)):
        raise StatisticsError('unique, nonempty frozen candidate IDs required')
    raw=json.dumps(list(candidate_ids),ensure_ascii=False,separators=(',',':')).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def pbo_cscv(candidate_returns: Sequence[Sequence[Sequence[float]]], *,
             candidate_ids: Sequence[str], max_partitions: int=10000,
             candidate_set_hash: str) -> dict:
    """[candidate][time_slice][returns] in the frozen candidate_id order."""
    if type(max_partitions) is not int or max_partitions < 1:
        raise StatisticsError('max_partitions must be a positive integer budget')
    if candidate_set_hash != candidate_set_sha256(candidate_ids):
        raise StatisticsError('candidate set changed since freeze')
    if len(candidate_returns) != len(candidate_ids):
        raise StatisticsError('candidate returns do not match declared frozen IDs')
    n=len(candidate_returns)
    if n<2:raise StatisticsError('PBO needs >=2 candidates')
    slices=len(candidate_returns[0]);half=slices//2
    if slices<4 or slices%2 or comb(slices,half)>max_partitions:
        raise StatisticsError('CSCV requires even number of slices; partition budget explicitly limited')
    if any(len(candidate)!=slices for candidate in candidate_returns):
        raise StatisticsError('inconsistent candidate time slices')
    for i in range(slices):
        if len({len(candidate[i]) for candidate in candidate_returns})!=1:
            raise StatisticsError('candidate return timestamps/slice sizes are not aligned')
    for candidate in candidate_returns:
        for segment in candidate:
            if len(segment)<2 or any(not _finite_real(x) for x in segment):
                raise StatisticsError('invalid slice returns')
    indices=tuple(range(slices));logits=[]
    for train in combinations(indices,half):
        train_indices=set(train)
        ins=[];outs=[]
        for candidate in candidate_returns:
            ins.append(_sharpe([x for i in indices if i in train_indices for x in candidate[i]]))
            outs.append(_sharpe([x for i in indices if i not in train_indices for x in candidate[i]]))
        if any(not isfinite(x) for x in ins+outs):
            raise StatisticsError('constant returns; PBO rank not identifiable')
        best=max(range(n),key=lambda i:(ins[i],-i))
        # average-rank ties; rank increases with OOS performance
        rank=1+sum(x<outs[best] for x in outs)+0.5*(sum(x==outs[best] for x in outs)-1)
        omega=rank/(n+1)
        logits.append(log(omega/(1-omega)))
    return {'pbo':sum(x<0 for x in logits)/len(logits),'logit_oos_rank':logits,
            'partitions':len(logits),'candidate_count':n,'slice_count':slices,
            'candidate_set_sha256':candidate_set_hash,'candidate_ids':list(candidate_ids),
            'method':'symmetric_cscv_v1'}
