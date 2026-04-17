import numpy as np

from pyvrp.Cost import Cost

_INFEASIBLE_COST = Cost(int(np.iinfo(np.int32).max), int(np.iinfo(np.int64).max))


class FirstFeasible:
    """
    Terminates the search after a feasible solution has been observed.
    """

    def __call__(self, best_cost) -> bool:
        # This function is called with the output of CostEvaluator.cost on the
        # best solution, which is (INT32_MAX, INT64_MAX) when the best solution
        # is infeasible. Thus, when the cost is below that, we have at least
        # one feasible solution and we can terminate.
        return best_cost < _INFEASIBLE_COST
