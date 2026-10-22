"""
Algoritmi:

1. BEAM SEARCH mbi renditjen e planeteve (pjesa kombinatorike, 7! = 5040 renditje).
   Tour-i nderohet hap pas hapi. Ne cdo nivel, per cdo tour te pjesshem provojme
   cdo planet te pavizituar si destinacion te radhes dhe ruajme vetem `width`
   tour-et me te lira (me DV me te vogel).
2. DIFFERENTIAL EVOLUTION (scipy) per 4 parametrat e vazhdueshem te cdo hapi.
3. Kohen e cdo hapi e kufizojme me T <= tcap. Per vlera te ndryshme te tcap
   marrim tour-e me te shpejte (DV me e madhe) ose me te ngadalte (DV me e vogel).
4. FRONTI PARETO i te gjitha tour-eve + HYPERVOLUME (pika referuese [2500, 4000]).
"""
import json
import math
from scipy.optimize import differential_evolution

import problem as P

REF = (2500.0, 4000.0)      # pika referuese e ESA-s
LAM = 0.5                   # sa "kushton" nje dite kohe [m/s per dite]


def optimize_entry(planet, seed):
    """Hapi hyres me DV minimale (T0 rreth 1500 dite, kur hyrja eshte pothuajse falas)."""
    def f(z):
        dv, viol, _ = P.entry_leg(z[0], z[1], z[2], planet)
        return dv + (1e7 if viol > 0 else 0)
    res = differential_evolution(f, [(0, 1), (0.25, 0.75), (1400, 1600)],
                                 maxiter=60, popsize=10, seed=seed, polish=False, tol=0)
    dv, viol, st = P.entry_leg(*res.x, planet)
    return dv, list(res.x), st


def optimize_leg(state, planet, tcap, seed):
    """4 parametrat [beta, rp, eta, T] te hapit, me T <= tcap."""
    def f(z):
        dv, viol, _ = P.next_leg(state, z[0], z[1], z[2], z[3], planet)
        return dv + LAM * z[3] + (1e7 if viol > 0 else 0)
    bounds = list(zip(P.LEG_LB, P.LEG_UB))
    bounds[3] = (5.0, tcap)
    res = differential_evolution(f, bounds, maxiter=60, popsize=10, seed=seed, polish=False, tol=0)
    dv, viol, st = P.next_leg(state, *res.x, planet)
    if viol > 0:
        return math.inf, list(res.x), None
    return dv, list(res.x), st


def beam_search(tcap, width=8, seed=0):
    """Kthen listen e tour-eve te plota: (DV, T, x)."""
    # niveli 1: nje tour i pjesshem per cdo planet te pare
    beam = []                               # elementet: (DV, x, state, renditja)
    for p in range(7):
        dv, x, st = optimize_entry(p, seed + p)
        beam.append((dv, x, st, [p]))
    # nivelet 2..7: shtojme nga nje planet
    for level in range(6):
        children = []
        for i, (dv, x, st, seq) in enumerate(beam):
            for p in range(7):
                if p in seq:
                    continue
                d, z, st2 = optimize_leg(st, p, tcap, seed + 100 * level + 10 * i + p)
                if st2 is not None:
                    children.append((dv + d, x + z, st2, seq + [p]))
        children.sort(key=lambda c: c[0])
        beam = children[:width]             # ruajme vetem me te mirat
    tours = []
    for dv, x, st, seq in beam:
        full = x + [float(s) for s in seq]
        f = P.fitness(full)
        if f[2] == 0 and f[3] <= 0:
            tours.append((f[0], f[1], full))
    return tours


def pareto_front(tours):
    """Zgjidhjet jo-te-dominuara (asnje tjeter nuk eshte me e mire ne te dy objektivat)."""
    front, best_T = [], math.inf
    for dv, T, x in sorted(tours):          # renditur sipas DV
        if T < best_T:
            front.append((dv, T, x))
            best_T = T
    return front


def hypervolume(points):
    """Siperfaqja e dominuar nga pikat deri te pika referuese (2 objektiva)."""
    pts = [(dv, T) for dv, T, *_ in points if dv < REF[0] and T < REF[1]]
    hv, prev_T = 0.0, REF[1]
    for dv, T in sorted(pts):
        if T < prev_T:
            hv += (REF[0] - dv) * (prev_T - T)
            prev_T = T
    return hv


def save_submission(front, path="submission.json"):
    xs = [x for dv, T, x in front if dv < REF[0] and T < REF[1]][:100]
    with open(path, "w") as fh:
        json.dump([{"decisionVector": xs, "problem": "trappist-tour",
                    "challenge": "spoc-trappist-tour"}], fh)
    return len(xs)
