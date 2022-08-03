from itertools import product

def plot_F_predictions_combined(benchmark: nmoo.Benchmark, n_samples: int):
    algorithms = benchmark._algorithms.keys()
    problems = benchmark._problems.keys()
    levels = ["1-knn_avg", "2-gaussian_noise", "3-wrapped_problem"]
    for a, p in product(algorithms, problems):
        df = pd.DataFrame()
        for l in levels:
            print(benchmark._output_dir_path / f"{p}.{a}.1.{l}.npz")
            history = np.load(benchmark._output_dir_path / f"{p}.{a}.1.{l}.npz")
            tmp = pd.DataFrame()
            tmp["_batch"] = history["_batch"]
            tmp["level"] = l
            for i in range(problem.n_obj):
                tmp[f"F{i}"] = history["F"][:, i]
            df = df.append(tmp, ignore_index=True)
        tmp = pd.DataFrame()
        tmp["_batch"] = history["_batch"]
        tmp["level"] = "true"
        gp = benchmark._problems[p]["problem"].ground_problem()
        out = {}
        gp._evaluate(history["X"], out)
        for i in range(problem.n_obj):
            tmp[f"F{i}"] = out["F"][:, i]
        df = df.append(tmp, ignore_index=True)

        n_samples = 50
        r = np.linspace(1, df._batch.max(), n_samples, dtype=int)
        grid = sns.FacetGrid(
            df[df._batch.isin(r)],
            col="_batch",
            col_wrap=int(sqrt(n_samples)),
        )
        grid.map_dataframe(sns.scatterplot, x="F0", y="F1", style="level", hue="level")
        grid.add_legend()
        pareto_front = gp.pareto_front()
        for ax in grid.axes:
            ax.plot(pareto_front[:, 0], pareto_front[:, 1], "--r")
        grid.savefig(benchmark._output_dir_path / f"{p}.{a}.1.jpg")


plot_F_predictions_combined(benchmark, 50)
