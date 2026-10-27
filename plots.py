"""
Grafiket e rezultateve (me matplotlib).
  plot_pareto(front) - zgjidhjet: karburanti kundrejt kohes
  plot_tour(x)       - radha e planeteve qe viziton anija
"""
import math
import matplotlib.pyplot as plt
import problem
import solver

AU = 149597870691.0          # 1 njesi astronomike ne metra


def plot_pareto(front):
    # marrim vetem zgjidhjet brenda pikes referuese [2500, 4000]
    karburanti = [dv for dv, T, x in front if dv < 2500 and T < 4000]
    koha = [T for dv, T, x in front if dv < 2500 and T < 4000]

    plt.figure(figsize=(8, 5))
    plt.scatter(karburanti, koha, color="blue", label="zgjidhjet")
    plt.scatter(2500, 4000, color="black", marker="x", label="pika referuese")
    plt.xlabel("Karburanti ΔV [m/s]")
    plt.ylabel("Koha e misionit [dite]")
    plt.title("Fronti Pareto, hypervolume = " + str(round(solver.hypervolume(front))))
    plt.legend()
    plt.grid(True)
    plt.savefig("pareto_front.png")
    plt.show()


def plot_tour(x):
    plt.figure(figsize=(7, 7))

    # 1) orbitat e planeteve si rrathe
    for p in problem.PLANETS:
        r = p.a / AU
        kendet = [2 * math.pi * k / 100 for k in range(101)]
        plt.plot([r * math.cos(k) for k in kendet], [r * math.sin(k) for k in kendet], color="lightgray")

    # 2) ylli ne qender
    plt.scatter(0, 0, color="orange", s=200, marker="*", label="ylli TRAPPIST-1")

    # 3) pozita e cdo planeti ne momentin kur anija arrin aty
    radha = [int(s) for s in x[27:34]]
    koha = problem.T_START
    pikat_x, pikat_y = [], []
    for i, planeti in enumerate(radha):
        koha += x[2 + 4 * i]                       # dita kur arrin te ky planet
        r, v = problem.PLANETS[planeti].eph(koha)  # pozita e planetit ate dite
        pikat_x.append(r[0] / AU)
        pikat_y.append(r[1] / AU)
        plt.text(r[0] / AU, r[1] / AU, f"  {i + 1}. {problem.NAMES[planeti]}")

    # 4) lidhim planetet sipas radhes se vizitave
    plt.plot(pikat_x, pikat_y, "o-", color="red", label="radha e vizitave")

    dv, T = problem.fitness(x)[:2]
    plt.title(f"ΔV = {dv:.0f} m/s, koha = {T:.0f} dite")
    plt.xlabel("x [AU]")
    plt.ylabel("y [AU]")
    plt.axis("equal")
    plt.legend()
    plt.savefig("tour.png")
    plt.show()
