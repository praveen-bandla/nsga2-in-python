import math
from importlib import import_module

import matplotlib.pyplot as plt
import numpy as np

from nsga2.individual import Individual
from nsga2.population import Population
from nsga2.problem import Problem
from nsga2.utils import NSGA2Utils
from portfolio.data import load_from_pipeline

try:
    fast_nondominated_sort_cy = import_module(
        "portfolio.cython_ops"
    ).fast_nondominated_sort_cy
except ModuleNotFoundError:  # pragma: no cover
    fast_nondominated_sort_cy = None

TRADING_DAYS = 252


def fast_nondominated_sort(population: Population) -> None:
    inds = population.population
    n = len(inds)
    if n == 0:
        population.fronts = [[]]
        return

    if fast_nondominated_sort_cy is None:
        NSGA2Utils(None).fast_nondominated_sort(population)
        return

    m = len(inds[0].objectives)
    objectives = np.empty((n, m), dtype=np.float64)
    for i, ind in enumerate(inds):
        objectives[i, :] = np.asarray(ind.objectives, dtype=np.float64)

    ranks = fast_nondominated_sort_cy(objectives)
    max_rank = int(ranks.max())
    population.fronts = [[] for _ in range(max_rank + 2)]
    for i, ind in enumerate(inds):
        ind.rank = int(ranks[i])
        population.fronts[ind.rank].append(ind)
    population.fronts.append([])


class ReturnVolProblem(Problem):
    def __init__(self, mean_returns_daily, cov_daily):
        self.mean_returns = np.asarray(mean_returns_daily, dtype=np.float64)
        self.cov = np.asarray(cov_daily, dtype=np.float64)
        self.num_assets = len(self.mean_returns)
        super().__init__(
            objectives=[lambda w: 0.0],
            num_of_variables=self.num_assets,
            variables_range=[(0.0, 1.0) for _ in range(self.num_assets)],
            expand=False,
            same_range=False,
        )

    def generate_individual(self):
        ind = Individual()
        w = np.random.random(self.num_assets)
        w /= w.sum()
        ind.features = w.astype(np.float64)
        return ind

    def calculate_objectives(self, individual):
        w = np.asarray(individual.features, dtype=np.float64)
        ret_d = float(w @ self.mean_returns)
        var_d = float(w @ (self.cov @ w))
        vol_d = math.sqrt(max(var_d, 0.0))
        ret_ann = ret_d * TRADING_DAYS
        vol_ann = vol_d * math.sqrt(TRADING_DAYS)
        individual.objectives = np.array([vol_ann, -ret_ann], dtype=np.float64)


def make_population(problem, size: int, seed: int) -> Population:
    np.random.seed(seed)
    pop = Population()
    for _ in range(size):
        ind = problem.generate_individual()
        problem.calculate_objectives(ind)
        pop.append(ind)
    return pop


def lou_downselect(problem, size: int, multiplier: int, seed: int) -> Population:
    utils = NSGA2Utils(problem, num_of_individuals=size)
    large = make_population(problem, size * multiplier, seed=seed)

    fast_nondominated_sort(large)
    for front in large.fronts:
        if front:
            utils.calculate_crowding_distance(front)

    selected = []
    front_num = 0
    while front_num < len(large.fronts) and large.fronts[front_num]:
        front = large.fronts[front_num]
        if len(selected) + len(front) <= size:
            selected.extend(front)
            front_num += 1
            continue
        front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
        selected.extend(front[: size - len(selected)])
        break

    out = Population()
    out.extend(selected)
    return out


def split_rank0(pop: Population):
    fast_nondominated_sort(pop)
    rank0 = list(pop.fronts[0]) if pop.fronts and pop.fronts[0] else []
    rank0_ids = {id(x) for x in rank0}
    rest = [ind for ind in pop.population if id(ind) not in rank0_ids]
    return rank0, rest


def xy_vol_ret(inds):
    obj = np.asarray([ind.objectives for ind in inds], dtype=float)
    return obj[:, 0], -obj[:, 1]


def nice_limits(x_all, y_all, pad=0.04):
    xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
    ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
    dx = xmax - xmin or 1.0
    dy = ymax - ymin or 1.0
    return (xmin - pad * dx, xmax + pad * dx), (ymin - pad * dy, ymax + pad * dy)


def draw(ax, title, blue_pop, rank0, rest):
    bx, by = xy_vol_ret(blue_pop.population)
    yx, yy = xy_vol_ret(rest)
    rx, ry = xy_vol_ret(rank0)

    h1 = ax.scatter(
        bx, by, s=6, c="tab:blue", alpha=0.20, linewidths=0, label="Random solutions"
    )
    h2 = ax.scatter(yx, yy, s=7, c="gold", alpha=0.45, linewidths=0, label="Population")
    h3 = ax.scatter(
        rx, ry, s=10, c="tab:red", alpha=0.95, linewidths=0, label="Pareto rank-0"
    )

    ax.set_title(title)
    ax.set_xlabel("Volatility (σ)")
    ax.set_ylabel("Expected return (μ)")
    ax.grid(True, alpha=0.35, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return h1, h2, h3


def main():
    size = 3000
    multiplier = 10

    mean_returns, cov_matrix, _, _ = load_from_pipeline()
    problem = ReturnVolProblem(mean_returns, cov_matrix)

    blue = make_population(problem, size, seed=1)
    std_pop = make_population(problem, size, seed=2)
    std_rank0, std_rest = split_rank0(std_pop)
    opt_pop = lou_downselect(problem, size, multiplier, seed=3)
    opt_rank0, opt_rest = split_rank0(opt_pop)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), dpi=180)
    handles = draw(
        axes[0], f"Standard initialization (N={size})", blue, std_rank0, std_rest
    )
    draw(
        axes[1],
        f"Optimized initialization (N={size}, pool={size * multiplier})",
        blue,
        opt_rank0,
        opt_rest,
    )

    x_all = np.concatenate(
        [
            xy_vol_ret(blue.population)[0],
            xy_vol_ret(std_rest)[0],
            xy_vol_ret(std_rank0)[0],
            xy_vol_ret(opt_rest)[0],
            xy_vol_ret(opt_rank0)[0],
        ]
    )
    y_all = np.concatenate(
        [
            xy_vol_ret(blue.population)[1],
            xy_vol_ret(std_rest)[1],
            xy_vol_ret(std_rank0)[1],
            xy_vol_ret(opt_rest)[1],
            xy_vol_ret(opt_rank0)[1],
        ]
    )
    xlim, ylim = nice_limits(x_all, y_all)
    for ax in axes:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.legend(
        handles,
        ["Random solutions", "Population", "Pareto rank-0"],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    out = "fig_return_vol_init_comparison.png"
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    main()
