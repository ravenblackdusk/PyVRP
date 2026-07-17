"""
Tests for soft-required (``ClientRequired.SOFT``) clients: clients the solver
should visit whenever possible, taking precedence over cost minimisation.
"""

import pytest
from numpy.testing import assert_, assert_equal

from pyvrp import (
    Client,
    ClientRequired,
    Cost,
    CostEvaluator,
    Model,
    ProblemData,
    Solution,
)
from pyvrp.stop import MaxIterations


def test_soft_client_prize_carries_missing_soft_required():
    """
    Tests that a SOFT client's prize gets a missing soft-required component,
    while HARD and NO clients keep a plain prize.
    """
    soft = Client(0, 0, prize=5, required=ClientRequired.SOFT)
    assert_equal(soft.prize, Cost(1, 5))
    assert_equal(soft.prize.missing_soft_required, 1)

    hard = Client(0, 0, prize=5, required=ClientRequired.HARD)
    assert_equal(hard.prize, Cost(5))
    assert_equal(hard.prize.missing_soft_required, 0)

    optional = Client(0, 0, prize=5, required=ClientRequired.NO)
    assert_equal(optional.prize, Cost(5))
    assert_equal(optional.prize.missing_soft_required, 0)


def _small_model(required: ClientRequired, prize: int = 0) -> Model:
    """
    One depot, one nearby optional-ish client, and one far away client with
    the given requirement. Visiting the far client is never worth it on cost
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


def test_soft_client_is_visited_despite_cost():
    """
    Tests that a zero-prize SOFT client is visited even though the detour is
    expensive, whereas the same client with ``ClientRequired.NO`` is skipped.
    """
    res_soft = _small_model(ClientRequired.SOFT).solve(
        stop=MaxIterations(50), seed=1, display=False
    )
    assert_(res_soft.is_feasible())
    assert_equal(res_soft.best.num_clients(), 2)

    res_opt = _small_model(ClientRequired.NO).solve(
        stop=MaxIterations(50), seed=1, display=False
    )
    assert_(res_opt.is_feasible())
    assert_equal(res_opt.best.num_clients(), 1)


def test_solution_cost_prefers_visiting_soft_clients():
    """
    Tests that a solution that visits the SOFT client is lexicographically
    better than a cheaper solution that skips it.
    """
    data = _small_model(ClientRequired.SOFT).data()
    cost_evaluator = CostEvaluator([], 0, 0)

    skips = Solution(data, [[1]])
    visits = Solution(data, [[1], [2]])

    # Missing a SOFT client does not make a solution infeasible; it is only
    # penalised in the cost's missing soft-required component.
    assert_(skips.is_feasible())
    assert_(visits.is_feasible())

    cost_skips = cost_evaluator.penalised_cost(skips)
    cost_visits = cost_evaluator.penalised_cost(visits)

    assert_equal(cost_skips.missing_soft_required, 1)
    assert_equal(cost_visits.missing_soft_required, 0)

    # Visiting is much more expensive in money, but still better overall.
    assert_(int(cost_visits) > int(cost_skips))
    assert_(cost_visits < cost_skips)


def test_soft_client_prize_is_returned_when_visited():
    """
    Tests that the collected prizes of a solution visiting a SOFT client
    include its missing soft-required component.
    """
    data = _small_model(ClientRequired.SOFT, prize=7).data()

    visits = Solution(data, [[1], [2]])
    assert_equal(visits.prizes().missing_soft_required, 1)
    assert_equal(visits.uncollected_prizes(), Cost(0))

    skips = Solution(data, [[1]])
    assert_equal(skips.prizes().missing_soft_required, 0)
    assert_equal(skips.uncollected_prizes(), Cost(1, 7))


@pytest.mark.parametrize("required", [ClientRequired.SOFT, ClientRequired.NO])
def test_soft_client_never_counts_as_missing(required: ClientRequired):
    """
    Tests that skipping a SOFT (or NO) client does not mark the solution as
    incomplete: only HARD clients are mandatory.
    """
    data = _small_model(required).data()
    solution = Solution(data, [[1]])
    assert_equal(solution.num_missing_clients(), 0)
    assert_(solution.is_complete())


def _sparse_model(required: ClientRequired) -> Model:
    """
    One depot and two clients. The depot and ``reachable`` are fully
    connected; ``unreachable`` has no edges at all, so it can only be visited
    through edges missing from the underlying network.
    """
    model = Model()
    depot = model.add_depot(0, 0, tw_early=0, tw_late=1_000)
    reachable = model.add_client(
        1, 0, tw_early=0, tw_late=1_000, required=required
    )
    model.add_client(50, 50, tw_early=0, tw_late=1_000, required=required)
    model.add_vehicle_type(num_available=2, tw_early=0, tw_late=1_000)

    model.add_edge(depot, reachable, 1, 1)
    model.add_edge(reachable, depot, 1, 1)

    return model


def test_model_data_edge_exists():
    """
    Tests that Model.data() marks exactly the explicitly added edges (and self
    loops) as existing.
    """
    data = _sparse_model(ClientRequired.SOFT).data()

    assert_(data.edge_exists(0, 0, 1))  # depot -> reachable
    assert_(data.edge_exists(0, 1, 0))  # reachable -> depot
    assert_(data.edge_exists(0, 0, 0))  # self loops exist
    assert_(not data.edge_exists(0, 0, 2))  # depot -> unreachable
    assert_(not data.edge_exists(0, 2, 0))
    assert_(not data.edge_exists(0, 1, 2))

    exists = data.edge_exists_matrices()
    assert_equal(len(exists), 1)
    assert_equal(exists[0].sum(), 5)  # 3 self loops + 2 explicit edges


def test_edge_exists_default_and_validation():
    """
    Tests that edge existence defaults to "all edges exist", and that
    inconsistent matrices are rejected.
    """
    import numpy as np

    data = _small_model(ClientRequired.SOFT).data()
    dense = ProblemData(
        data.clients(),
        data.depots(),
        data.vehicle_types(),
        data.distance_matrices(),
        data.duration_matrices(),
    )
    assert_equal(len(dense.edge_exists_matrices()), 0)
    assert_(dense.edge_exists(0, 0, 2))  # empty means all edges exist

    with pytest.raises(ValueError):
        ProblemData(
            data.clients(),
            data.depots(),
            data.vehicle_types(),
            data.distance_matrices(),
            data.duration_matrices(),
            [],
            [np.ones((2, 2), dtype=bool)],  # wrong shape
        )


def test_unreachable_soft_client_is_skipped():
    """
    Tests that a SOFT client only reachable via missing edges is left out of
    the solution, which remains feasible and still visits reachable clients.
    This used to wedge the search in an infeasible state.
    """
    res = _sparse_model(ClientRequired.SOFT).solve(
        stop=MaxIterations(50), seed=2, display=False
    )

    assert_(res.is_feasible())
    assert_equal(res.best.num_clients(), 1)  # reachable only

    visits = [client for route in res.best.routes() for client in route]
    assert_equal(visits, [1])


def test_solution_never_uses_missing_edges_for_soft_clients():
    """
    Tests on a larger sparse instance that the solution's routes only use
    edges that exist.
    """
    model = Model()
    depot = model.add_depot(0, 0)
    clients = [
        model.add_client(i, i, required=ClientRequired.SOFT) for i in range(8)
    ]
    model.add_vehicle_type(num_available=3)

    # Chain structure: depot <-> client 0 <-> client 1 <-> ... <-> client 4.
    # Clients 5..7 are disconnected.
    chain = [depot] + clients[:5]
    for frm, to in zip(chain, chain[1:]):
        model.add_edge(frm, to, 1)
        model.add_edge(to, frm, 1)
    for client in clients[:5]:  # direct return legs to the depot
        model.add_edge(client, depot, 10)

    data = model.data()
    res = model.solve(stop=MaxIterations(100), seed=3, display=False)

    assert_(res.is_feasible())
    assert_equal(res.best.num_clients(), 5)

    for route in res.best.routes():
        stops = [0, *route, 0]
        for frm, to in zip(stops, stops[1:]):
            assert_(data.edge_exists(0, frm, to))


def test_data_edge_exists_roundtrips():
    """
    Tests that edge existence survives pickling and ``replace()``.
    """
    import pickle

    data = _sparse_model(ClientRequired.SOFT).data()

    copy = pickle.loads(pickle.dumps(data))
    assert_equal(copy, data)
    assert_(not copy.edge_exists(0, 0, 2))

    replaced = data.replace()
    assert_(not replaced.edge_exists(0, 0, 2))
