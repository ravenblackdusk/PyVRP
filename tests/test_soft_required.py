"""
Tests for soft-required (``ClientRequired.SOFT``) clients: clients the solver
should visit whenever it feasibly can, before optimising monetary cost.
"""

import pytest
from numpy.testing import assert_, assert_equal

from pyvrp import (
    ClientRequired,
    CostEvaluator,
    Model,
    PenaltyManager,
    Solution,
)
from pyvrp.stop import MaxIterations


def _small_model(required: ClientRequired, prize: int = 0) -> Model:
    """
    One depot, one nearby required client, and one far away client with the
    given requirement. Visiting the far client is never worth it on cost
    alone.
    """
    model = Model()
    depot = model.add_depot(0, 0)
    near = model.add_client(1, 0, required=ClientRequired.HARD)
    far = model.add_client(100, 100, prize=prize, required=required)
    model.add_vehicle_type(num_available=2)

    dist = [[0, 1, 200], [1, 0, 199], [200, 199, 0]]
    for frm_idx, frm in enumerate([depot, near, far]):
        for to_idx, to in enumerate([depot, near, far]):
            model.add_edge(frm, to, dist[frm_idx][to_idx])

    return model


def test_num_missing_soft():
    """
    Tests that solutions count their missing soft-required clients, and that
    such clients are neither missing (HARD) nor make a solution incomplete.
    """
    data = _small_model(ClientRequired.SOFT).data()

    skips = Solution(data, [[1]])
    assert_equal(skips.num_missing_soft(), 1)
    assert_equal(skips.num_missing_clients(), 0)
    assert_(skips.is_complete())
    assert_(skips.is_feasible())

    visits = Solution(data, [[1], [2]])
    assert_equal(visits.num_missing_soft(), 0)


def test_missing_soft_enters_penalised_cost():
    """
    Tests that the penalised cost includes the missing soft-required penalty
    at its current value, while ``cost()`` uses the maximum value.
    """
    data = _small_model(ClientRequired.SOFT).data()
    skips = Solution(data, [[1]])
    visits = Solution(data, [[1], [2]])

    cost_eval = CostEvaluator(
        [], 0, 0, missing_soft_penalty=7, max_missing_soft_penalty=1_000
    )
    assert_equal(cost_eval.missing_soft_penalty(1), 7)
    assert_equal(cost_eval.missing_soft_penalty(3), 21)

    # Penalised cost weighs the missing client at the current value (7),
    # cost() at the maximum (1000).
    base = CostEvaluator([], 0, 0)
    assert_equal(
        cost_eval.penalised_cost(skips), base.penalised_cost(skips) + 7
    )
    assert_equal(cost_eval.cost(skips), base.cost(skips) + 1_000)

    # A solution visiting all soft clients is unaffected.
    assert_equal(cost_eval.penalised_cost(visits), base.penalised_cost(visits))
    assert_equal(cost_eval.cost(visits), base.cost(visits))


def test_cost_ranking_prefers_visiting_soft_clients():
    """
    Tests that with the maximum penalty at its instance-derived value, a
    solution that visits the SOFT client ranks better in ``cost()`` than a
    much cheaper one that skips it.
    """
    data = _small_model(ClientRequired.SOFT).data()
    cost_eval = PenaltyManager.init_from(data).cost_evaluator()

    skips = Solution(data, [[1]])
    visits = Solution(data, [[1], [2]])

    # Visiting is much more expensive in money, but the missing-soft weight
    # in cost() dominates any such saving.
    assert_(visits.distance() > skips.distance())
    assert_(cost_eval.cost(visits) < cost_eval.cost(skips))


def test_soft_client_is_visited_despite_cost():
    """
    Tests that a zero-prize SOFT client is visited even though the detour is
    expensive, whereas the same client with ``ClientRequired.NO`` is skipped.
    """
    res_soft = _small_model(ClientRequired.SOFT).solve(
        stop=MaxIterations(100), seed=1, display=False
    )
    assert_(res_soft.is_feasible())
    assert_equal(res_soft.best.num_missing_soft(), 0)
    assert_equal(res_soft.best.num_clients(), 2)

    res_opt = _small_model(ClientRequired.NO).solve(
        stop=MaxIterations(100), seed=1, display=False
    )
    assert_(res_opt.is_feasible())
    assert_equal(res_opt.best.num_clients(), 1)


def test_soft_clients_beyond_capacity_are_skipped():
    """
    Tests that when not all SOFT clients can be feasibly served (here due to
    a single vehicle with a tight shift), the solver serves as many as it can
    and skips the rest, returning a feasible solution.
    """
    model = Model()
    depot = model.add_depot(0, 0, tw_early=0, tw_late=100)
    clients = [
        model.add_client(
            i,
            0,
            service_duration=30,
            tw_early=0,
            tw_late=100,
            required=ClientRequired.SOFT,
        )
        for i in range(5)
    ]
    model.add_vehicle_type(num_available=1, tw_early=0, tw_late=100)

    locs = [depot, *clients]
    for frm in locs:
        for to in locs:
            if frm is not to:
                model.add_edge(frm, to, 1, 1)

    # Serving a client costs 30 service + at least 1 travel, and everything
    # must happen within 100 time units: at most 3 of 5 clients fit. The
    # penalty update cadence is shortened so the missing-soft and time warp
    # penalties can find their equilibrium within a small iteration budget.
    from pyvrp import PenaltyParams
    from pyvrp.solve import SolveParams

    params = SolveParams(penalty=PenaltyParams(solutions_between_updates=25))
    res = model.solve(
        stop=MaxIterations(1_000), seed=6, display=False, params=params
    )

    assert_(res.is_feasible())
    assert_equal(res.best.time_warp(), 0)
    assert_equal(res.best.num_clients(), 3)
    assert_equal(res.best.num_missing_soft(), 2)


@pytest.mark.parametrize("required", [ClientRequired.SOFT, ClientRequired.NO])
def test_missing_soft_penalty_registration(required: ClientRequired):
    """
    Tests that the penalty manager escalates the missing-soft penalty while
    solutions miss soft-required clients, and decays it otherwise.
    """
    from pyvrp import PenaltyParams

    data = _small_model(required).data()
    params = PenaltyParams(solutions_between_updates=1, penalty_increase=2)
    pm = PenaltyManager(([], 1, 1, 4), params, max_missing_soft_penalty=100)

    skips = Solution(data, [[1]])
    pm.register(skips)
    *_, soft = pm.penalties()

    if required == ClientRequired.SOFT:
        assert_equal(soft, 8)  # escalated: a soft client is missing
    else:
        assert_(soft < 4)  # decayed: nothing soft is missing
