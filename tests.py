"""
Testet e projektit. Ekzekutohen me:   python tests.py
Nese te gjitha kalojne, shfaqet "Te gjitha testet kaluan".
"""
import math
import problem
import physics
import solver


def test_shembulli_esa():
    """Shembulli zyrtar i ESA-s duhet te jape DV = 2379 m/s dhe T = 3470 dite."""
    dv, T, eq, ineq = problem.fitness(problem.example())
    assert abs(dv - 2378.99) < 0.1, dv
    assert abs(T - 3469.83) < 0.1, T
    assert eq == 0 and ineq <= 0          # zgjidhja eshte e pranueshme
    print("OK  shembulli i ESA-s:", round(dv, 1), "m/s,", round(T, 1), "dite")


def test_lambert():
    """Anija qe niset me shpejtesine e Lambert-it duhet te arrije te planeti i synuar."""
    r1, _ = problem.PLANETS[0].eph(8000)          # planeti b
    r2, _ = problem.PLANETS[3].eph(8010)          # planeti e, 10 dite me vone
    v1, _ = physics.lambert(r1, r2, 10 * physics.DAY, problem.MU)
    r_arrives, _ = physics.propagate(r1, v1, 10 * physics.DAY, problem.MU)
    gabimi = physics.norm(physics.sub(r_arrives, r2))
    assert gabimi < 1000, gabimi                  # me pak se 1 km gabim
    print("OK  Lambert: gabimi ne mberritje =", round(gabimi, 3), "m")


def test_flyby():
    """Fly-by nuk e ndryshon madhesine e shpejtesise relative ndaj planetit."""
    p = problem.PLANETS[2]
    v_in, v_pla = (90e3, 5e3, 1e3), (80e3, 0.0, 0.0)
    v_out = physics.flyby(v_in, v_pla, 2 * p.radius, 1.0, p.mu_self)
    para = physics.norm(physics.sub(v_in, v_pla))
    pas = physics.norm(physics.sub(v_out, v_pla))
    assert abs(para - pas) < 1e-6
    print("OK  fly-by ruan shpejtesine relative:", round(para), "m/s")


def test_renditja_e_gabuar():
    """Nese nje planet perseritet, kufizimi eq duhet te jete i shkelur."""
    x = problem.example()
    x[27:34] = [0, 0, 2, 3, 4, 5, 6]              # planeti b dy here
    assert problem.fitness(x)[2] != 0
    print("OK  renditja e gabuar zbulohet")


def test_pareto():
    """Fronti Pareto mban vetem zgjidhjet qe nuk dominohen."""
    tours = [(1000, 3000, "A"), (2000, 2000, "B"), (1500, 3500, "C"), (900, 3900, "D")]
    front = solver.pareto_front(tours)
    emrat = [t[2] for t in front]
    assert emrat == ["D", "A", "B"], emrat       # C dominohet nga A
    print("OK  fronti Pareto:", emrat)


def test_hypervolume():
    """Hypervolume = siperfaqja e dominuar deri te pika referuese [2500, 4000]."""
    assert solver.hypervolume([(1500, 3000)]) == 1000 * 1000
    assert solver.hypervolume([(1000, 3000), (2000, 2000)]) == 1500 * 1000 + 500 * 1000
    assert solver.hypervolume([(3000, 1000)]) == 0          # jashte pikes referuese
    print("OK  hypervolume")


if __name__ == "__main__":
    test_shembulli_esa()
    test_lambert()
    test_flyby()
    test_renditja_e_gabuar()
    test_pareto()
    test_hypervolume()
    print("\nTe gjitha testet kaluan.")
