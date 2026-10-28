"""
Krahasimi i algoritmeve (analiza DAA):

  1. Random Search    - plane udhetimi krejt te rastit (pa asnje strategji)
  2. Greedy           - Beam Search me gjeresi 1 (merr gjithmone vetem me te mirin)
  3. Beam Search W=4  - mban 4 rruget me te mira ne cdo nivel
  4. Beam Search W=8  - mban 8 rruget me te mira (algoritmi yne kryesor)

Per secilin masim: hypervolume (cilesia), numrin e zgjidhjeve dhe kohen.
"""
import random
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import problem
import solver


def random_search(n=20000, seed=0):
    """Gjeneron n vektore krejt te rastit dhe mban ata qe jane te pranueshem."""
    rng = random.Random(seed)
    tours = []
    for _ in range(n):
        x = [rng.uniform(problem.LB[i], problem.UB[i]) for i in range(27)]
        x += [float(s) for s in rng.sample(range(7), 7)]     # radhe e rastit e planeteve
        dv, T, eq, ineq = problem.fitness(x)
        if eq == 0 and ineq <= 0:
            tours.append((dv, T, x))
    return tours


def _beam(args):
    tcap, width = args
    return solver.beam_search(tcap, width=width)


def run_beam(width, tcaps):
    """Ekzekuton Beam Search me gjeresine `width` per te gjithe kufijte kohore `tcaps`."""
    with Pool() as pool:
        results = pool.map(_beam, [(t, width) for t in tcaps])
    return [t for r in results for t in r]


def run_all(tcaps, beam8_tours=None, beam8_minutes=None):
    """Kthen nje liste rreshtash: (emri, hypervolume, zgjidhje ne front, minuta).
    Nese Beam W=8 eshte ekzekutuar me pare (hapi 5), jepi rezultatet qe te mos rillogaritet."""
    rows = []

    start = time.time()
    tours = random_search()
    rows.append(("Random Search", tours, (time.time() - start) / 60))
    print("Random Search: gati")

    for width, name in [(1, "Greedy (W=1)"), (4, "Beam Search (W=4)")]:
        start = time.time()
        tours = run_beam(width, tcaps)
        rows.append((name, tours, (time.time() - start) / 60))
        print(name + ": gati")

    if beam8_tours is None:
        start = time.time()
        beam8_tours = run_beam(8, tcaps)
        beam8_minutes = (time.time() - start) / 60
    rows.append(("Beam Search (W=8)", beam8_tours, beam8_minutes))

    table = []
    for name, tours, minutes in rows:
        front = solver.pareto_front(tours)
        table.append((name, round(solver.hypervolume(front)), len(front), round(minutes, 1)))
    return table


def print_table(table):
    print(f"{'Algoritmi':<20}{'Hypervolume':>14}{'Zgjidhje':>10}{'Koha (min)':>12}")
    print("-" * 56)
    for name, hv, n, minutes in table:
        print(f"{name:<20}{hv:>14,}{n:>10}{minutes:>12}")


def plot_table(table):
    names = [r[0] for r in table]
    hvs = [r[1] for r in table]
    times = [r[3] for r in table]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar(names, hvs, color="steelblue")
    ax1.set_title("Cilesia: hypervolume (me shume = me mire)")
    ax1.tick_params(axis="x", rotation=20)
    for i, v in enumerate(hvs):
        ax1.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)

    ax2.bar(names, times, color="darkorange")
    ax2.set_title("Koha e ekzekutimit (minuta)")
    ax2.tick_params(axis="x", rotation=20)
    for i, v in enumerate(times):
        ax2.text(i, v, str(v), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("krahasimi.png")
    plt.show()
