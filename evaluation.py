# Run evaluation and plot figures
import math
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

from algorithms import benchmark_using_algorithm
from network import QuantumNetwork

plt.rc('font', family='Times New Roman')  # Use the same font as the IEEE template
plt.rc('font', size=20)
default_cycler = (cycler(color=['#4daf4a', '#377eb8', '#e41a1c', '#984ea3', '#ff7f00', '#a65628']) +
                  cycler(marker=['s', 'v', 'o', 'x', '*', '+']) + cycler(linestyle=[':', '--', '-', '-.', '--', ':']))
plt.rc('axes', prop_cycle=default_cycler)


def generate_fidelity_list_avg_gap(path_num):
    result = []
    fidelity_max = 1
    fidelity_min = 0.9
    gap = (fidelity_max - fidelity_min) / path_num
    fidelity = fidelity_max
    for path in range(path_num):
        result.append(fidelity)
        fidelity -= gap
    assert len(result) == path_num
    return result


def generate_fidelity_list_fix_gap(path_num, gap, fidelity_max=1):
    result = []
    fidelity = fidelity_max
    for path in range(path_num):
        result.append(fidelity)
        fidelity -= gap
    assert len(result) == path_num
    return result


def generate_fidelity_list_random(path_num, alpha=0.95, beta=0.85, variance=0.1):
    '''Generate `path_num` links. The fidelity is determined as follows:
       u_1 = alpha, u_i = beta for all i = 2, 3, ..., n.
       Then, the fidelity of link i is a Gaussian random variable with mean u_i and variance `variance`.
    '''
    while True:
        mean = [alpha] + [beta] * (path_num - 1)
        result = []
        for i in range(path_num):
            mu = mean[i]
            # Sample a Gaussian random variable and make sure its value is in the valid range
            while True:
                r = np.random.normal(mu, variance)
                # Depolarizing noise and amplitude damping noise models require that fidelity >= 0.5
                # To be conservative, we require it >= 0.75
                if r >= 0.8 and r <= 1:
                    break
            result.append(r)
        assert len(result) == path_num
        sorted_res = sorted(result, reverse=True)
        # To guarantee the termination of algorithms, we require that the gap is large enough
        if sorted_res[0] - sorted_res[1] > 0.02:
            return result


def plot_cost_vs_path_num(path_num_list, algorithm_names, noise_model, repeat=10):
    file_name = f"plot_cost_vs_path_num_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))  # The path of the current script
    output_dir = os.path.join(root_dir, "outputs")
    file_path = os.path.join(output_dir, f"{file_name}.pickle")

    if os.path.exists(file_path):
        print("Pickle data exists, skip simulation and plot the data directly.")
        print("To rerun the simulation, delete the pickle file in `plots/outputs` directory.")
        with open(file_path, 'rb') as f:
            saved_data = pickle.load(f)
            # Handle both old and new format
            if isinstance(saved_data, dict) and 'metadata' in saved_data:
                results = saved_data
            else:
                # Old format compatibility
                results = {'metadata': {}, 'data': saved_data}
    else:
        bounces = [1, 2, 3, 4]
        sample_times = {}
        for i in bounces:
            sample_times[i] = 200

        results = {
            'metadata': {
                'noise_model': noise_model,
                'repeat': repeat,
                'bounces': bounces,
                'sample_times': sample_times,
                'gap': 0.04
            },
            'data': {}
        }

        for algo in algorithm_names:
            results['data'][algo] = {
                'path_num_list': path_num_list,
                'costs_per_path_num': [],
                'fidelity_lists': [],
                'raw_results': []
            }

        for path_num in path_num_list:
            path_list = list(range(1, path_num + 1))
            # fidelity_list = generate_fidelity_list_avg_gap(path_num)
            fidelity_list = generate_fidelity_list_fix_gap(path_num, 0.04)
            # print(
            #     f"Initializing network with {path_num} paths: {path_list}, true fidelities: {fidelity_list}, noise model: {noise_model}\n"
            # )
            # network = QuantumNetwork(path_num, fidelity_list, noise_model)

            for algorithm_name in algorithm_names:
                correct_rate = 0
                cost_list = []
                raw_results_list = []
                for i in range(repeat):  # Repeat several times and get average
                    print(f"Evaluating algorithm: {algorithm_name}, repeat: {i+1}/{repeat}...")

                    print(
                        f"Initializing network with {path_num} paths: {path_list}, true fidelities: {fidelity_list}, noise model: {noise_model}\n"
                    )
                    network = QuantumNetwork(path_num, fidelity_list, noise_model)
                    correctness, cost, estimated_fidelity = benchmark_using_algorithm(network, path_list, algorithm_name, bounces,
                                                                     sample_times)
                    print(f"Finish repeat {i+1}/{repeat}, correctness: {correctness}")
                    correct_rate += correctness
                    cost_list.append(cost)
                    raw_results_list.append({
                        'correctness': correctness,
                        'cost': cost,
                        'estimated_fidelity': estimated_fidelity
                    })
                correct_rate /= repeat
                print(f"Finish evaluating algorithm {algorithm_name}, correct rate: {correct_rate}\n")
                results['data'][algorithm_name]['costs_per_path_num'].append(cost_list)
                results['data'][algorithm_name]['fidelity_lists'].append(fidelity_list)
                results['data'][algorithm_name]['raw_results'].append(raw_results_list)

        # Store the results in file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)

    # Plot
    plt.rc('axes', prop_cycle=default_cycler)
    fig, ax = plt.subplots()

    # Handle both old and new format
    data_dict = results.get('data', results)

    for algorithm_name, algo_data in data_dict.items():
        # Handle both old and new format
        if isinstance(algo_data, tuple):
            # Old format: (path_num_list, costs_list)
            path_num_list_plot, costs_list = algo_data
        else:
            # New format: dict with keys
            path_num_list_plot = algo_data['path_num_list']
            costs_list = algo_data['costs_per_path_num']

        std_errs = []
        avg_costs = []
        max_costs = []
        min_costs = []
        for costs in costs_list:
            # print("Costs", costs)
            max_costs.append(max(costs))
            min_costs.append(min(costs))
            std_errs.append(np.std(costs))
            avg_costs.append(np.mean(costs))

        avg_costs = np.array(avg_costs)
        max_costs = np.array(max_costs)
        min_costs = np.array(min_costs)
        error_bar = np.stack((avg_costs - min_costs, max_costs - avg_costs))
        # print("STD ERR", std_errs)
        plt.fill_between(path_num_list_plot, min_costs, max_costs, interpolate=True, alpha=0.2)

        if algorithm_name == "Vanilla NB":
            algorithm_name = "VanillaNB"
        elif algorithm_name == "Succ. Elim. NB":
            algorithm_name = "SuccElimNB"

        ax.errorbar(path_num_list_plot,
                    avg_costs,
                    yerr=error_bar,
                    elinewidth=1.0,
                    capsize=3,
                    linewidth=2.0,
                    label=algorithm_name)
    plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    ax.set_xlabel('Number of Quantum Links')
    ax.set_ylabel('Average Number of Bounces')
    ax.grid(True)
    ax.legend(title="Algorithm", fontsize=14, title_fontsize=18)
    plt.tight_layout()
    pdf_name = f"{file_name}.pdf"
    plt.savefig(pdf_name)
    os.system(f"pdfcrop {pdf_name} {pdf_name}")  # Crop margins of PDF
    # plt.show()


def plot_cost_vs_gap(path_num, gap_list, algorithm_names, noise_model, repeat=5):
    file_name = f"plot_cost_vs_gap_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))  # The path of the current script
    output_dir = os.path.join(root_dir, "outputs")
    file_path = os.path.join(output_dir, f"{file_name}.pickle")

    if os.path.exists(file_path):
        print("Pickle data exists, skip simulation and plot the data directly.")
        print("To rerun the simulation, delete the pickle file in `plots/outputs` directory.")
        with open(file_path, 'rb') as f:
            saved_data = pickle.load(f)
            # Handle both old and new format
            if isinstance(saved_data, dict) and 'metadata' in saved_data:
                results = saved_data
            else:
                # Old format compatibility
                results = {'metadata': {}, 'data': saved_data}
    else:
        # bounces = list(range(1, 5))
        bounces = [1, 2, 3, 4]
        sample_times = {}
        for i in bounces:
            sample_times[i] = 200

        results = {
            'metadata': {
                'noise_model': noise_model,
                'repeat': repeat,
                'bounces': bounces,
                'sample_times': sample_times,
                'path_num': path_num
            },
            'data': {}
        }

        for algo in algorithm_names:
            results['data'][algo] = {
                'gap_list': gap_list,
                'costs_per_gap': [],
                'fidelity_lists': [],
                'raw_results': []
            }

        for gap in gap_list:
            path_list = list(range(1, path_num + 1))
            # fidelity_list = generate_fidelity_list_avg_gap(path_num)
            fidelity_list = generate_fidelity_list_fix_gap(path_num, gap)
            # print(
            #     f"Initializing network with {path_num} paths: {path_list}, gap: {gap}, true fidelities: {fidelity_list}, noise model: {noise_model}\n"
            # )
            # network = QuantumNetwork(path_num, fidelity_list, noise_model)

            for algorithm_name in algorithm_names:
                print(f"Using algorithm: {algorithm_name}, path_list: {path_list}")
                correct_rate = 0
                cost_list = []
                raw_results_list = []
                for i in range(repeat):  # Repeat several times and get average
                    print(f"Evaluating algorithm: {algorithm_name}, repeat: {i+1}/{repeat}...")

                    print(
                        f"Initializing network with {path_num} paths: {path_list}, gap: {gap}, true fidelities: {fidelity_list}, noise model: {noise_model}\n"
                    )
                    network = QuantumNetwork(path_num, fidelity_list, noise_model)

                    correctness, cost, estimated_fidelity = benchmark_using_algorithm(network, path_list, algorithm_name, bounces,
                                                                     sample_times)
                    print(f"Finish repeat {i+1}/{repeat}, correctness: {correctness}")
                    correct_rate += correctness
                    cost_list.append(cost)
                    raw_results_list.append({
                        'correctness': correctness,
                        'cost': cost,
                        'estimated_fidelity': estimated_fidelity
                    })
                correct_rate /= repeat
                print(f"Finish evaluating algorithm {algorithm_name}, correct rate: {correct_rate}\n")
                results['data'][algorithm_name]['costs_per_gap'].append(cost_list)
                results['data'][algorithm_name]['fidelity_lists'].append(fidelity_list)
                results['data'][algorithm_name]['raw_results'].append(raw_results_list)

        # Store the results in file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)

    # Plot
    plt.rc('axes', prop_cycle=default_cycler)
    fig, ax = plt.subplots()

    # Handle both old and new format
    data_dict = results.get('data', results)

    for algorithm_name, algo_data in data_dict.items():
        # Handle both old and new format
        if isinstance(algo_data, tuple):
            # Old format: (gap_list, costs_list)
            gap_list_plot, costs_list = algo_data
        else:
            # New format: dict with keys
            gap_list_plot = algo_data['gap_list']
            costs_list = algo_data['costs_per_gap']

        std_errs = []
        avg_costs = []
        max_costs = []
        min_costs = []
        for costs in costs_list:
            max_costs.append(max(costs))
            min_costs.append(min(costs))
            # error_bar.append((np.mean(costs) - min(costs), max(costs) - np.mean(costs)))
            std_errs.append(np.std(costs))
            avg_costs.append(np.mean(costs))
        avg_costs = np.array(avg_costs)
        max_costs = np.array(max_costs)
        min_costs = np.array(min_costs)
        error_bar = np.stack((avg_costs - min_costs, max_costs - avg_costs))

        if algorithm_name == "Vanilla NB":
            algorithm_name = "VanillaNB"
        elif algorithm_name == "Succ. Elim. NB":
            algorithm_name = "SuccElimNB"
        plt.fill_between(gap_list_plot, min_costs, max_costs, interpolate=False, alpha=0.2)
        ax.errorbar(gap_list_plot, avg_costs, yerr=error_bar, elinewidth=1.0, capsize=3, linewidth=2.0, label=algorithm_name)
        # ax.plot(gap_list, costs, linewidth=2.0, label=algorithm_name)
    plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    ax.set_xlabel('Gap')
    ax.set_ylabel('Average Number of Bounces')
    ax.grid(True)
    ax.legend(title="Algorithm", fontsize=14, title_fontsize=18)
    plt.tight_layout()
    pdf_name = f"{file_name}.pdf"
    plt.savefig(pdf_name)
    os.system(f"pdfcrop {pdf_name} {pdf_name}")  # Crop margins of PDF
    # plt.show()


def plot_error_vs_path_num(path_num_list, algorithm_names, noise_model, repeat=1):
    file_name = f"plot_error_vs_path_num_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))  # The path of the current script
    output_dir = os.path.join(root_dir, "outputs")
    file_path = os.path.join(output_dir, f"{file_name}.pickle")

    if os.path.exists(file_path):
        print("Pickle data exists, skip simulation and plot the data directly.")
        print("To rerun the simulation, delete the pickle file in `plots/outputs` directory.")
        with open(file_path, 'rb') as f:
            saved_data = pickle.load(f)
            # Handle both old and new format
            if isinstance(saved_data, dict) and 'metadata' in saved_data:
                results = saved_data
            else:
                # Old format compatibility
                results = {'metadata': {}, 'data': saved_data}
    else:
        bounces = [1, 2, 3, 4]
        sample_times = {}
        for i in bounces:
            sample_times[i] = 200

        results = {
            'metadata': {
                'noise_model': noise_model,
                'repeat': repeat,
                'bounces': bounces,
                'sample_times': sample_times
            },
            'data': {}
        }

        for algo in algorithm_names:
            results['data'][algo] = {
                'path_num_list': [],
                'errors_per_path_num': [],
                'fidelity_lists': [],
                'raw_results': []
            }

        for path_num in path_num_list:
            path_list = list(range(1, path_num + 1))
            # fidelity_list = generate_fidelity_list_random(path_num)
            # best_fidelity = max(fidelity_list)
            # network = QuantumNetwork(path_num, fidelity_list, noise_model)
            # print(
            #     f"Initializing network with {path_num} paths: {path_list}, true fidelities: {fidelity_list}, noise model: {noise_model}\n"
            # )

            for algorithm_name in algorithm_names:
                print(f"Using algorithm: {algorithm_name}, path_list: {path_list}")
                correct_rate = 0
                relative_error_list = []
                raw_results_list = []
                fidelity_lists_for_path_num = []
                for i in range(repeat):  # Repeat several times and get average
                    print(f"Evaluating algorithm: {algorithm_name}, repeat: {i+1}/{repeat}...")

                    fidelity_list = generate_fidelity_list_random(path_num)
                    best_fidelity = max(fidelity_list)
                    network = QuantumNetwork(path_num, fidelity_list, noise_model)
                    print(
                        f"Initializing network with {path_num} paths: {path_list}, true fidelities: {fidelity_list}, noise model: {noise_model}\n"
                    )

                    correctness, cost, estimated_fidelity = benchmark_using_algorithm(
                        network, path_list, algorithm_name, bounces, sample_times)
                    print(f"Finish repeat {i+1}/{repeat}, correctness: {correctness}")
                    correct_rate += correctness
                    relative_error = abs(estimated_fidelity - best_fidelity) / best_fidelity
                    relative_error_list.append(relative_error)
                    fidelity_lists_for_path_num.append(fidelity_list)
                    raw_results_list.append({
                        'correctness': correctness,
                        'cost': cost,
                        'estimated_fidelity': estimated_fidelity,
                        'best_fidelity': best_fidelity,
                        'relative_error': relative_error
                    })
                correct_rate /= repeat
                print(f"Finish evaluating algorithm {algorithm_name}, correct rate: {correct_rate}\n")
                # print(f"Estimated fidelity: {np.mean(estimated_fidelity_list)}\n")

                results['data'][algorithm_name]['path_num_list'].append(path_num)
                results['data'][algorithm_name]['errors_per_path_num'].append(relative_error_list)
                results['data'][algorithm_name]['fidelity_lists'].append(fidelity_lists_for_path_num)
                results['data'][algorithm_name]['raw_results'].append(raw_results_list)
                # results[algorithm_name] = list(error_probability.values())

        # Store the results in file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)

    # Plot
    plt.rc('axes', prop_cycle=default_cycler)
    fig, ax = plt.subplots()

    # Handle both old and new format
    data_dict = results.get('data', results)

    for algorithm_name, algo_data in data_dict.items():
        # Handle both old and new format
        if isinstance(algo_data, tuple):
            # Old format: (path_num_list, errors_list)
            path_num_list_plot, errors_list = algo_data
        else:
            # New format: dict with keys
            path_num_list_plot = algo_data['path_num_list']
            errors_list = algo_data['errors_per_path_num']

        std_errs = []
        avg_errors = []
        max_errors = []
        min_errors = []
        for errors in errors_list:
            max_errors.append(max(errors))
            min_errors.append(min(errors))
            std_errs.append(np.std(errors))
            avg_errors.append(np.mean(errors))
        # error_range = np.stack((, ymax-ymean))
        # plt.fill_between(path_num_list, min_errors, max_errors, interpolate=True, alpha=0.2)
        # ax.errorbar(path_num_list,
        #             avg_errors,
        #             yerr=std_errs,
        #             elinewidth=1.0,
        #             capsize=3,
        #             linewidth=2.0,
        #             label=algorithm_name)
        if algorithm_name == "Vanilla NB":
            algorithm_name = "VanillaNB"
        elif algorithm_name == "Succ. Elim. NB":
            algorithm_name = "SuccElimNB"
        ax.plot(path_num_list_plot, avg_errors, linewidth=2.0, label=algorithm_name)
    plt.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
    ax.set_xlabel('Number of Links')
    ax.set_ylabel('Estimated Error')
    ax.grid(True)
    ax.legend(title="Algorithm", fontsize=14, title_fontsize=18)
    plt.tight_layout()
    pdf_name = f"{file_name}.pdf"
    plt.savefig(pdf_name)
    os.system(f"pdfcrop {pdf_name} {pdf_name}")  # Crop margins of PDF
    # plt.show()


def plot_accuracy_vs_gap(path_num, gap_list, algorithm_names, noise_model, repeat=10):
    """
    Plot accuracy (mean ± 95% CI) vs. gap.
    - gap 定義は plot_cost_vs_gap と同じ：generate_fidelity_list_fix_gap(path_num, gap)
    - correctness (0/1) を repeat 回繰り返して平均を精度として採用
    - 95% CI は正規近似 p̂ ± 1.96 * sqrt(p̂(1-p̂)/n) を使用
    出力: outputs/plot_accuracy_vs_gap_{noise_model}.pickle と PDF
    """
    file_name = f"plot_accuracy_vs_gap_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(root_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{file_name}.pickle")

    if os.path.exists(file_path):
        print("Pickle data exists, skip simulation and plot the data directly.")
        print("To rerun the simulation, delete the pickle file in `plots/outputs` directory.")
        with open(file_path, "rb") as f:
            saved_data = pickle.load(f)
            # Handle both old and new format
            if isinstance(saved_data, dict) and 'metadata' in saved_data:
                results = saved_data
            else:
                # Old format compatibility
                results = {'metadata': {}, 'data': saved_data}
    else:
        # NBセット構成は他の関数と揃える
        bounces = [1, 2, 3, 4]
        sample_times = {i: 200 for i in bounces}

        results = {
            'metadata': {
                'noise_model': noise_model,
                'repeat': repeat,
                'bounces': bounces,
                'sample_times': sample_times,
                'path_num': path_num
            },
            'data': {}
        }

        for algo in algorithm_names:
            results['data'][algo] = {
                'gap_list': gap_list,
                'accs_per_gap': [],
                'fidelity_lists': [],
                'raw_results': []
            }

        path_list = list(range(1, path_num + 1))
        for gap in gap_list:
            fidelity_list = generate_fidelity_list_fix_gap(path_num, gap)
            print(f"[GAP {gap}] paths={path_list}, fids={fidelity_list}, noise={noise_model}")

            for algorithm_name in algorithm_names:
                print(f"  Evaluating {algorithm_name} ...")
                accs = []
                raw_results_list = []
                for r in range(repeat):
                    network = QuantumNetwork(path_num, fidelity_list, noise_model)
                    correctness, cost, estimated_fidelity = benchmark_using_algorithm(
                        network, path_list, algorithm_name, bounces, sample_times
                    )
                    accs.append(float(correctness))
                    raw_results_list.append({
                        'correctness': correctness,
                        'cost': cost,
                        'estimated_fidelity': estimated_fidelity
                    })
                results['data'][algorithm_name]['accs_per_gap'].append(accs)
                results['data'][algorithm_name]['fidelity_lists'].append(fidelity_list)
                results['data'][algorithm_name]['raw_results'].append(raw_results_list)

        with open(file_path, "wb") as f:
            pickle.dump(results, f)

    # ---- Plot ----
    plt.rc('axes', prop_cycle=default_cycler)
    fig, ax = plt.subplots()

    def _canon_name(name: str) -> str:
        if name == "Vanilla NB":
            return "VanillaNB"
        if name == "Succ. Elim. NB":
            return "SuccElimNB"
        return name

    # Handle both old and new format
    data_dict = results.get('data', results)

    for algorithm_name, algo_data in data_dict.items():
        # Handle both old and new format
        if isinstance(algo_data, tuple):
            # Old format: (gaps, acc_lists_per_gap)
            gaps, acc_lists_per_gap = algo_data
        else:
            # New format: dict with keys
            gaps = algo_data['gap_list']
            acc_lists_per_gap = algo_data['accs_per_gap']

        means = []
        lowers = []
        uppers = []

        for accs in acc_lists_per_gap:
            p = float(np.mean(accs))
            n = max(1, len(accs))
            # 95% CI (normal approximation for Bernoulli mean)
            ci = 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / n)
            means.append(p)
            lowers.append(max(0.0, p - ci))
            uppers.append(min(1.0, p + ci))

        means = np.array(means)

        uppers = np.array(uppers)

        ax.fill_between(gaps, lowers, uppers, alpha=0.2)
        ax.plot(gaps, means, linewidth=2.0, label=_canon_name(algorithm_name))

    ax.set_xlabel("Gap")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend(title="Algorithm", fontsize=14, title_fontsize=18)
    plt.tight_layout()
    pdf_name = f"{file_name}.pdf"
    plt.savefig(pdf_name)
    os.system(f"pdfcrop {pdf_name} {pdf_name}")  # 余白トリム
    # plt.show()


def plot_accuracy_vs_path_num(path_num_list, algorithm_names, noise_model, repeat=10):
    """
    横軸: path_num（リンク数）
    縦軸: accuracy（correctness の平均; 0/1 を repeat 回）
    - その他の構成は plot_cost_vs_path_num に極力そろえる
    """
    file_name = f"plot_accuracy_vs_path_num_{noise_model}"
    root_dir = os.path.dirname(os.path.abspath(__file__))  # The path of the current script
    output_dir = os.path.join(root_dir, "outputs")
    file_path = os.path.join(output_dir, f"{file_name}.pickle")

    if os.path.exists(file_path):
        print("Pickle data exists, skip simulation and plot the data directly.")
        print("To rerun the simulation, delete the pickle file in `plots/outputs` directory.")
        with open(file_path, 'rb') as f:
            saved_data = pickle.load(f)
            # Handle both old and new format
            if isinstance(saved_data, dict) and 'metadata' in saved_data:
                results = saved_data
            else:
                # Old format compatibility
                results = {'metadata': {}, 'data': saved_data}
    else:
        bounces = [1, 2, 3, 4]
        sample_times = {}
        for i in bounces:
            sample_times[i] = 200

        results = {
            'metadata': {
                'noise_model': noise_model,
                'repeat': repeat,
                'bounces': bounces,
                'sample_times': sample_times,
                'gap': 0.04
            },
            'data': {}
        }

        for algo in algorithm_names:
            results['data'][algo] = {
                'path_num_list': path_num_list,
                'accs_per_path_num': [],
                'fidelity_lists': [],
                'raw_results': []
            }

        for path_num in path_num_list:
            path_list = list(range(1, path_num + 1))
            # Use same fidelity generation method as plot_cost_vs_path_num
            fidelity_list = generate_fidelity_list_fix_gap(path_num, 0.04)

            for algorithm_name in algorithm_names:
                correct_rate = 0.0
                acc_list = []
                raw_results_list = []
                for i in range(repeat):  # Repeat several times and get average
                    print(f"Evaluating algorithm: {algorithm_name}, repeat: {i+1}/{repeat}...")

                    print(
                        f"Initializing network with {path_num} paths: {path_list}, true fidelities: {fidelity_list}, noise model: {noise_model}\n"
                    )
                    network = QuantumNetwork(path_num, fidelity_list, noise_model)
                    correctness, cost, estimated_fidelity = benchmark_using_algorithm(
                        network, path_list, algorithm_name, bounces, sample_times
                    )
                    print(f"Finish repeat {i+1}/{repeat}, correctness: {correctness}")
                    correct_rate += correctness
                    acc_list.append(float(correctness))
                    raw_results_list.append({
                        'correctness': correctness,
                        'cost': cost,
                        'estimated_fidelity': estimated_fidelity
                    })
                correct_rate /= repeat
                print(f"Finish evaluating algorithm {algorithm_name}, correct rate: {correct_rate}\n")
                results['data'][algorithm_name]['accs_per_path_num'].append(acc_list)
                results['data'][algorithm_name]['fidelity_lists'].append(fidelity_list)
                results['data'][algorithm_name]['raw_results'].append(raw_results_list)

        # Store the results in file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(file_path, 'wb') as f:
            pickle.dump(results, f)

    # Plot
    plt.rc('axes', prop_cycle=default_cycler)
    fig, ax = plt.subplots()

    # Handle both old and new format
    data_dict = results.get('data', results)

    for algorithm_name, algo_data in data_dict.items():
        # Handle both old and new format
        if isinstance(algo_data, tuple):
            # Old format: (pnums, acc_lists_per_pathnum)
            pnums, acc_lists_per_pathnum = algo_data
        else:
            # New format: dict with keys
            pnums = algo_data['path_num_list']
            acc_lists_per_pathnum = algo_data['accs_per_path_num']

        std_errs = []
        avg_accs = []
        max_accs = []
        min_accs = []
        for accs in acc_lists_per_pathnum:
            max_accs.append(max(accs))
            min_accs.append(min(accs))
            std_errs.append(np.std(accs))
            avg_accs.append(np.mean(accs))

        avg_accs = np.array(avg_accs)
        max_accs = np.array(max_accs)
        min_accs = np.array(min_accs)
        error_bar = np.stack((avg_accs - min_accs, max_accs - avg_accs))

        plt.fill_between(pnums, min_accs, max_accs, interpolate=True, alpha=0.2)

        name_for_legend = algorithm_name
        if name_for_legend == "Vanilla NB":
            name_for_legend = "VanillaNB"
        elif name_for_legend == "Succ. Elim. NB":
            name_for_legend = "SuccElimNB"

        ax.errorbar(pnums,
                    avg_accs,
                    yerr=error_bar,
                    elinewidth=1.0,
                    capsize=3,
                    linewidth=2.0,
                    label=name_for_legend)

    # ※縦軸のみ変更（他は極力オリジナル準拠）
    ax.set_xlabel('Number of Quantum Links')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0.0, 1.0)  # 精度なので 0〜1 に固定
    ax.grid(True)
    ax.legend(title="Algorithm", fontsize=14, title_fontsize=18)
    plt.tight_layout()
    pdf_name = f"{file_name}.pdf"
    plt.savefig(pdf_name)
    os.system(f"pdfcrop {pdf_name} {pdf_name}")  # Crop margins of PDF
    # plt.show()
