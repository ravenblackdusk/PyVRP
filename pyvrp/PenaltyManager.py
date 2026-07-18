from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from warnings import warn

import numpy as np

from pyvrp._pyvrp import CostEvaluator, ProblemData, Solution
from pyvrp.constants import MAX_VALUE
from pyvrp.exceptions import PenaltyBoundWarning


@dataclass
class PenaltyParams:
    """
    The penalty manager parameters.

    Parameters
    ----------
    solutions_between_updates
        Number of feasibility registrations between penalty value updates. The
        penalty manager updates the penalty terms every once in a while based
        on recent feasibility registrations. This parameter controls how often
        such updating occurs.
    penalty_increase
        Amount :math:`p_i \\ge 1` by which the current penalties are
        increased when insufficient feasible solutions (see
        ``target_feasible``) have been found amongst the most recent
        registrations. The penalty values :math:`v` are updated as
        :math:`v \\gets p_i v`.
    penalty_decrease
        Amount :math:`p_d \\in [0, 1]` by which the current penalties are
        decreased when sufficient feasible solutions (see ``target_feasible``)
        have been found amongst the most recent registrations. The penalty
        values :math:`v` are updated as :math:`v \\gets p_d v`.
    target_feasible
        Target percentage :math:`p_f \\in [0, 1]` of feasible registrations
        in the last ``solutions_between_updates`` registrations. This
        percentage is used to update the penalty terms: when insufficient
        feasible solutions have been registered, the penalties are increased;
        similarly, when too many feasible solutions have been registered, the
        penalty terms are decreased. This ensures a balanced search, with a
        fraction :math:`p_f` feasible and a fraction :math:`1 - p_f` infeasible
        solutions.
    feas_tolerance
        Deviation tolerance (in :math:`[0, 1]`) between actual and target
        percentage of feasible solutions between updates. If the deviation is
        smaller than this tolerance, the penalty terms are not updated. See
        also ``target_feasible`` and ``solutions_between_updates``.
    min_penalty
        Minimum penalty term value. Must not be negative.
    max_penalty
        Maximum penalty term value. Must not be negative.

        .. warning::
           Setting a (too) large maximum penalty value may cause integer
           overflow in PyVRP's native extensions.
    min_missing_soft_penalty
        Minimum value of the missing soft-required client penalty. The
        adaptive penalty never drops below this value, and also starts no
        lower. Set this (possibly equal to ``max_missing_soft_penalty``) to
        keep constant pressure to serve soft-required clients when the
        iteration budget is too small for the adaptive ramp-up.
    max_missing_soft_penalty
        Maximum value of the missing soft-required client penalty. This value
        determines how strongly the solver insists on serving soft-required
        clients: at this value, a client is only dropped when keeping it
        costs more than this in penalised terms (for example, time warp at
        the maximum time warp penalty). When not provided, a value is derived
        from the problem data as an upper bound on the money a solution could
        save by dropping one client.

    Attributes
    ----------
    solutions_between_updates
        Number of feasibility registrations between penalty value updates.
    penalty_increase
        Amount :math:`p_i \\ge 1` by which the current penalties are
        increased when insufficient feasible solutions (see
        ``target_feasible``) have been found amongst the most recent
        registrations.
    penalty_decrease
        Amount :math:`p_d \\in [0, 1]` by which the current penalties are
        decreased when sufficient feasible solutions (see ``target_feasible``)
        have been found amongst the most recent registrations.
    target_feasible
        Target percentage :math:`p_f \\in [0, 1]` of feasible registrations
        in the last ``solutions_between_updates`` registrations.
    feas_tolerance
        Deviation tolerance for ``target_feasible``.
    min_penalty
        Minimum penalty term value.
    max_penalty
        Maximum penalty term value.
    """

    solutions_between_updates: int = 500
    penalty_increase: float = 1.50
    penalty_decrease: float = 0.95
    target_feasible: float = 0.65
    feas_tolerance: float = 0.05
    min_penalty: float = 0.1
    max_penalty: float = 100_000.0
    min_missing_soft_penalty: float | None = None
    max_missing_soft_penalty: float | None = None

    def __post_init__(self):
        if not self.solutions_between_updates >= 1:
            raise ValueError("Expected solutions_between_updates >= 1.")

        if not self.penalty_increase >= 1.0:
            raise ValueError("Expected penalty_increase >= 1.")

        if not (0.0 <= self.penalty_decrease <= 1.0):
            raise ValueError("Expected penalty_decrease in [0, 1].")

        if not (0.0 <= self.target_feasible <= 1.0):
            raise ValueError("Expected target_feasible in [0, 1].")

        if not (0.0 <= self.feas_tolerance <= 1.0):
            raise ValueError("Expected feas_tolerance in [0, 1].")

        if self.min_penalty < 0:
            raise ValueError("Expected min_penalty >= 0.")

        if self.max_penalty < self.min_penalty:
            raise ValueError("Expected max_penalty >= min_penalty.")

        min_soft = (
            self.min_missing_soft_penalty
            if self.min_missing_soft_penalty is not None
            else self.min_penalty
        )
        if min_soft < 0:
            raise ValueError("Expected min_missing_soft_penalty >= 0.")

        if (
            self.max_missing_soft_penalty is not None
            and self.max_missing_soft_penalty < min_soft
        ):
            raise ValueError(
                "Expected max_missing_soft_penalty >= its minimum."
            )


class PenaltyManager:
    """
    Creates a PenaltyManager instance.

    This class manages time warp and load penalties, and provides penalty terms
    for given time warp and load values. It updates these penalties based on
    recent history.

    .. note::

       Consider initialising using :meth:`~init_from` to compute initial
       penalty values that are scaled according to the data instance.

    Parameters
    ----------
    initial_penalties
        Initial penalty values for units of load (idx 0), duration (1), and
        distance (2) violations, and for each missing soft-required client
        (3). These values are clipped to the range
        [:attr:`~pyvrp.PenaltyManager.PenaltyParams.min_penalty`,
        :attr:`~pyvrp.PenaltyManager.PenaltyParams.max_penalty`], where the
        missing soft-required penalty uses ``max_missing_soft_penalty`` as
        its upper bound instead.
    params
        PenaltyManager parameters. If not provided, a default will be used.
    max_missing_soft_penalty
        Maximum value of the missing soft-required penalty. This should be
        set well above the monetary cost a solution could save by dropping a
        single client, so that a missing soft-required client is never
        preferred over such savings in the final solution ranking. When not
        provided, ``params.max_penalty`` is used.
    """

    def __init__(
        self,
        initial_penalties: tuple[list[float], float, float, float],
        params: PenaltyParams = PenaltyParams(),
        max_missing_soft_penalty: float | None = None,
    ):
        self._params = params

        if max_missing_soft_penalty is None:
            max_missing_soft_penalty = params.max_penalty

        self._max_missing_soft_penalty = max_missing_soft_penalty
        self._min_missing_soft_penalty = (
            params.min_missing_soft_penalty
            if params.min_missing_soft_penalty is not None
            else params.min_penalty
        )

        *loads, tw, dist, soft = np.asarray(
            initial_penalties[0] + list(initial_penalties[1:])
        )
        self._penalties = np.array(
            [
                *np.clip(loads, params.min_penalty, params.max_penalty),
                np.clip(tw, params.min_penalty, params.max_penalty),
                np.clip(dist, params.min_penalty, params.max_penalty),
                np.clip(
                    soft,
                    self._min_missing_soft_penalty,
                    max_missing_soft_penalty,
                ),
            ]
        )

        # Tracks recent feasibilities for each penalty dimension.
        self._feas_lists: list[list[bool]] = [
            [] for _ in range(len(self._penalties))
        ]

    def penalties(self) -> tuple[list[float], float, float, float]:
        """
        Returns the current penalty values.
        """
        return (
            self._penalties[:-3].tolist(),  # loads
            self._penalties[-3],  # duration
            self._penalties[-2],  # distance
            self._penalties[-1],  # missing soft-required
        )

    @classmethod
    def init_from(
        cls,
        data: ProblemData,
        params: PenaltyParams = PenaltyParams(),
    ) -> PenaltyManager:
        """
        Initialises from the given data instance and parameter object. The
        initial penalty values are computed from the problem data.

        Parameters
        ----------
        data
            Data instance to use when computing penalty values.
        params
            PenaltyManager parameters. If not provided, a default will be used.
        """
        distances = data.distance_matrices()
        durations = data.duration_matrices()

        # We first determine the elementwise minimum cost across all vehicle
        # types. This is the cheapest way any edge can be traversed.
        unique_edge_costs = {
            (
                veh_type.unit_distance_cost,
                veh_type.unit_duration_cost,
                veh_type.profile,
            )
            for veh_type in data.vehicle_types()
        }

        first, *rest = unique_edge_costs
        unit_dist, unit_dur, prof = first
        edge_costs = unit_dist * distances[prof] + unit_dur * durations[prof]
        for unit_dist, unit_dur, prof in rest:
            mat = unit_dist * distances[prof] + unit_dur * durations[prof]
            np.minimum(edge_costs, mat, out=edge_costs)

        # Best edge cost/distance/duration over all vehicle types and profiles,
        # and then average that for the entire matrix to obtain an "average
        # best" edge cost/distance/duration.
        avg_cost = edge_costs.mean()
        avg_distance = np.minimum.reduce(distances).mean()
        avg_duration = np.minimum.reduce(durations).mean()

        avg_load = np.zeros((data.num_load_dimensions,))
        if data.num_clients != 0 and data.num_load_dimensions != 0:
            pickups = np.array([c.pickup for c in data.clients()])
            deliveries = np.array([c.delivery for c in data.clients()])
            avg_load = np.maximum(pickups, deliveries).mean(axis=0)

        # Initial penalty parameters are meant to weigh an average increase
        # in the relevant value by the same amount as the average edge cost.
        init_load = avg_cost / np.maximum(avg_load, 1)
        init_tw = avg_cost / max(avg_duration, 1)
        init_dist = avg_cost / max(avg_distance, 1)

        # A missing soft-required client initially weighs about as much as
        # the average detour of serving a client: two average edges. Its
        # maximum weight must exceed any monetary amount a solution could
        # save by dropping a single client, which is bounded by the worst
        # possible detour plus the largest fixed vehicle cost. Both are
        # derived from real edges only: missing edges carry MAX_VALUE-sized
        # placeholder values that would inflate these weights so much that
        # dropping an unservable soft client could never pay off.
        real_edges = edge_costs[edge_costs < MAX_VALUE]
        avg_real = real_edges.mean() if real_edges.size else avg_cost
        max_real = real_edges.max() if real_edges.size else edge_costs.max()

        init_soft = 2 * avg_real
        max_fixed = max((v.fixed_cost for v in data.vehicle_types()), default=0)
        max_soft = params.max_missing_soft_penalty
        if max_soft is None:
            max_soft = 2 * (max_real + 1) + max_fixed

        return cls(
            (init_load.tolist(), init_tw, init_dist, min(init_soft, max_soft)),
            params,
            max_missing_soft_penalty=max_soft,
        )  # the constructor clips the initial value to the soft bounds

    def _compute(
        self,
        penalty: float,
        feas_percentage: float,
        min_penalty: float,
        max_penalty: float,
    ) -> float:
        # Computes and returns the new penalty value, given the current value
        # and the percentage of feasible solutions since the last update.
        diff = self._params.target_feasible - feas_percentage

        if abs(diff) < self._params.feas_tolerance:
            return penalty

        if diff > 0:
            new_penalty = self._params.penalty_increase * penalty
        else:
            new_penalty = self._params.penalty_decrease * penalty

        if new_penalty >= max_penalty:
            msg = """
            A penalty parameter has reached its maximum value. This means PyVRP
            struggles to find a feasible solution for this instance, either
            because the instance has no feasible solution, or it is hard to
            find one - possibly due to large data scaling differences. Check
            the instance carefully to determine if a feasible solution exists.
            """
            warn(msg, PenaltyBoundWarning)

        return np.clip(new_penalty, min_penalty, max_penalty)

    def _register(
        self,
        feas_list: list[bool],
        penalty: float,
        is_feas: bool,
        min_penalty: float,
        max_penalty: float,
    ):
        feas_list.append(is_feas)

        if len(feas_list) != self._params.solutions_between_updates:
            return penalty

        avg = fmean(feas_list)
        feas_list.clear()
        return self._compute(penalty, avg, min_penalty, max_penalty)

    def register(self, sol: Solution):
        """
        Registers the feasibility dimensions of the given solution.
        """
        is_feasible = [
            *[excess == 0 for excess in sol.excess_load()],
            not sol.has_time_warp(),
            not sol.has_excess_distance(),
            sol.num_missing_soft() == 0,
        ]

        for idx, is_feas in enumerate(is_feasible):
            feas_list = self._feas_lists[idx]
            penalty = self._penalties[idx]
            is_soft = idx == len(is_feasible) - 1
            min_penalty = (
                self._min_missing_soft_penalty
                if is_soft
                else self._params.min_penalty
            )
            max_penalty = (
                self._max_missing_soft_penalty
                if is_soft
                else self._params.max_penalty
            )
            self._penalties[idx] = self._register(
                feas_list, penalty, is_feas, min_penalty, max_penalty
            )

    def cost_evaluator(self) -> CostEvaluator:
        """
        Get a cost evaluator using the current penalty values.
        """
        *loads, tw, dist, soft = self._penalties
        return CostEvaluator(
            loads, tw, dist, soft, self._max_missing_soft_penalty
        )

    def max_cost_evaluator(self) -> CostEvaluator:
        """
        Get a cost evaluator using the maximum penalty value.
        """
        penalties = np.full_like(self._penalties, self._params.max_penalty)
        *loads, tw, dist, _ = penalties
        soft = self._max_missing_soft_penalty
        return CostEvaluator(loads, tw, dist, soft, soft)
