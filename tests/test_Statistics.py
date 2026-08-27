import pytest
from numpy.testing import assert_, assert_equal

from pyvrp import CostEvaluator, Solution, Statistics


def test_csv_serialises_correctly(ok_small, tmp_path):
    """
    Tests that writing a CSV of a ``Statistics`` object and then reading that
    CSV again returns the same object.
    """
    sol = Solution(ok_small, [[1, 2], [3, 4]])
    cost_eval = CostEvaluator([20], 6, 6)

    collected_stats = Statistics(ok_small)
    for idx in range(10):  # populate the statistics object
        collected_stats.collect(sol, sol, sol, cost_eval)

    csv_path = tmp_path / "test.csv"
    assert_(not csv_path.exists())

    # Write the collected statistics to the CSV file location, and check that
    # the file now does exist.
    collected_stats.to_csv(csv_path)
    assert_(csv_path.exists())

    # Now read them back in, and check that the newly read object is the same
    # as the previously written object.
    read_stats = Statistics.from_csv(csv_path)
    assert_equal(collected_stats, read_stats)


@pytest.mark.parametrize("num_iterations", [0, 5])
def test_collect_a_data_point_per_iteration(ok_small, num_iterations: int):
    """
    Tests that the statistics object collects solution statistics every time
    ``collect`` is called.
    """
    stats = Statistics(ok_small)
    assert_(stats.is_collecting())

    sol = Solution(ok_small, [[1, 2], [3, 4]])
    cost_eval = CostEvaluator([20], 6, 6)

    for _ in range(num_iterations):  # populate the statistics object
        stats.collect(sol, sol, sol, cost_eval)

    assert_equal(len(stats.data), num_iterations)


def test_data_point_matches_collect(ok_small):
    """
    Tests that the data point collected matches with the values passed to
    ``collect()``.
    """
    curr = Solution(ok_small, [[1, 2, 3, 4]])
    cand = Solution(ok_small, [[2, 1], [4, 3]])
    best = Solution(ok_small, [[1, 2], [3, 4]])
    cost_eval = CostEvaluator([20], 6, 6)

    stats = Statistics(ok_small)
    stats.collect(curr, cand, best, cost_eval)

    datum = stats.data[0]

    assert_equal(datum.current_cost, 28408)
    assert_(not datum.current_feas)
    assert_equal(
        datum.current_route_durations,
        tuple(r.duration() for r in curr.routes()),
    )

    assert_equal(datum.candidate_cost, 10012)
    assert_(datum.candidate_feas)

    assert_equal(datum.best_cost, 9725)
    assert_(datum.best_feas)
    assert_equal(
        datum.best_route_durations,
        tuple(r.duration() for r in best.routes()),
    )
    assert_equal(datum.best_num_missing, 0)
    assert_equal(datum.best_distance, best.distance())


def test_best_num_missing_counts_a_group_once_when_any_member_is_visited(
    ok_small_mutually_exclusive_groups,
):
    """
    Tests that a client group counts as served, contributing nothing to
    best_num_missing, as soon as any one of its members is visited - even
    though the other members remain unvisited.
    """
    data = ok_small_mutually_exclusive_groups
    cost_eval = CostEvaluator([20], 6, 6)

    # Clients 1, 2, 3 are in the group; client 4 is not. Visiting just one
    # group member (1) should satisfy the whole group.
    sol = Solution(data, [[1, 4]])
    stats = Statistics(data)
    stats.collect(sol, sol, sol, cost_eval)
    assert_equal(stats.data[0].best_num_missing, 0)

    # Now nobody in the group is visited, so the group as a whole is missing
    # - but that is one missing stop, not three.
    sol = Solution(data, [[4]])
    stats = Statistics(data)
    stats.collect(sol, sol, sol, cost_eval)
    assert_equal(stats.data[0].best_num_missing, 1)


@pytest.mark.parametrize("num_iterations", [0, 1, 10])
def test_eq(ok_small, num_iterations: int):
    """
    Tests the equality operator.
    """
    sol = Solution(ok_small, [[1, 2], [3, 4]])
    cost_eval = CostEvaluator([20], 6, 6)

    stats = Statistics(ok_small)
    for _ in range(num_iterations):  # populate the statistics object
        stats.collect(sol, sol, sol, cost_eval)

    assert_equal(stats, stats)
    assert_(stats != "test")

    if num_iterations > 0:
        assert_(stats != Statistics(ok_small))


def test_more_eq(ok_small):
    """
    Tests the equality operator for the same search trajectory.
    """
    sol = Solution(ok_small, [[1, 2], [3, 4]])
    cost_eval = CostEvaluator([20], 6, 6)

    stats1 = Statistics(ok_small)
    stats2 = Statistics(ok_small)

    assert_equal(stats1, stats2)
    assert_(stats1 != "str")

    stats1.collect(sol, sol, sol, cost_eval)
    assert_(stats1 != stats2)

    # Now do the same thing for stats2, so they have the seen the exact same
    # search trajectory. That, however, is not enough: the runtimes are
    # still slightly different.
    stats2.collect(sol, sol, sol, cost_eval)
    assert_(stats1 != stats2)

    # But once we fix that the two should be the exact same again.
    stats2.runtimes = stats1.runtimes
    stats2.data = stats1.data
    assert_equal(stats1, stats2)


def test_iterating_over_statistics_returns_data(ok_small):
    """
    Tests that iterating over a Statistics object yields the correct data.
    """
    stats = Statistics(ok_small)
    assert_equal(list(stats), [])

    sol = Solution(ok_small, [[1, 2], [3, 4]])
    cost_eval = CostEvaluator([20], 6, 6)
    cost = cost_eval.penalised_cost(sol)
    route_durations = tuple(r.duration() for r in sol.routes())

    stats.collect(sol, sol, sol, cost_eval)
    stats.collect(sol, sol, sol, cost_eval)

    assert_equal(len(list(stats)), 2)

    for datum in stats:
        assert_(datum.elapsed >= 0)
        assert_equal(datum.current_cost, cost)
        assert_equal(datum.candidate_cost, cost)
        assert_equal(datum.best_cost, cost)
        assert_equal(datum.current_route_durations, route_durations)
        assert_equal(datum.best_route_durations, route_durations)
        assert_equal(datum.best_num_missing, 0)
        assert_equal(datum.best_distance, sol.distance())
        assert_(datum.current_feas)
        assert_(datum.candidate_feas)
        assert_(datum.best_feas)


def test_not_collecting(ok_small):
    """
    Tests that calling collect_from() on a Statistics object that is not
    collecting is a no-op.
    """
    sol = Solution(ok_small, [[1, 2], [3, 4]])
    cost_eval = CostEvaluator([20], 6, 6)

    stats = Statistics(ok_small, collect_stats=False)
    assert_(not stats.is_collecting())

    stats.collect(sol, sol, sol, cost_eval)
    stats.collect(sol, sol, sol, cost_eval)

    assert_equal(stats, Statistics(ok_small, collect_stats=False))
