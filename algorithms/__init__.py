from .naive_nb import naive_network_benchmarking  # noqa: F401
from .naive1_nb import naive1_network_benchmarking  # noqa: F401
from .naive4_nb import naive4_network_benchmarking  # noqa: F401
from .naive5_nb import naive5_network_benchmarking  # noqa: F401
from .naive20_nb import naive20_network_benchmarking  # noqa: F401
from .naive50_nb import naive50_network_benchmarking  # noqa: F401
from .naive80_nb import naive80_network_benchmarking  # noqa: F401
from .naive100_nb import naive100_network_benchmarking  # noqa: F401
from .naive150_nb import naive150_network_benchmarking  # noqa: F401
from .online_nb import online_network_benchmarking  # noqa: F401
from .succ_elim_nb import \
    successive_elimination_network_benchmarking  # noqa: F401


def benchmark_using_algorithm(network, path_list, algorithm_name, bounces, sample_times):
    if algorithm_name == "Vanilla NB":
        return naive_network_benchmarking(network, path_list, bounces, sample_times)
    if algorithm_name == "Vanilla1 NB":
        return naive1_network_benchmarking(network, path_list, bounces, sample_times)
    if algorithm_name == "Vanilla4 NB":
        return naive4_network_benchmarking(network, path_list, bounces, sample_times)
    if algorithm_name == "Vanilla5 NB":
        return naive5_network_benchmarking(network, path_list, bounces, sample_times)
    if algorithm_name == "Vanilla20 NB":
        return naive20_network_benchmarking(network, path_list, bounces, sample_times)
    if algorithm_name == "Vanilla20 NB":
        return naive50_network_benchmarking(network, path_list, bounces, sample_times)
    if algorithm_name == "Vanilla50 NB":
        return naive80_network_benchmarking(network, path_list, bounces, sample_times)
    if algorithm_name == "Vanilla80 NB":
        return naive100_network_benchmarking(network, path_list, bounces, sample_times)
    if algorithm_name == "Vanilla100 NB":
        return naive100_network_benchmarking(network, path_list, bounces, sample_times)
    if algorithm_name == "Vanilla150 NB":
        return naive150_network_benchmarking(network, path_list, bounces, sample_times)
    elif algorithm_name == "LinkSelFiE":
        return online_network_benchmarking(network, path_list, bounces)
    elif algorithm_name == "Succ. Elim. NB":
        return successive_elimination_network_benchmarking(network, path_list, bounces)
    else:
        print("Error: Unknown algorithm name")
        exit(1)
