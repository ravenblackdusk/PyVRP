#include "Solution.h"

#include "primitives.h"

#include <algorithm>
#include <cassert>
#include <iterator>

using pyvrp::search::Solution;

Solution::Solution(ProblemData const &data) : data_(data)
{
    nodes.reserve(data.numLocations());
    for (size_t loc = 0; loc != data.numLocations(); ++loc)
        nodes.emplace_back(loc);

    routes.reserve(data.numVehicles());
    size_t rIdx = 0;
    for (size_t vehType = 0; vehType != data.numVehicleTypes(); ++vehType)
    {
        auto const numAvailable = data.vehicleType(vehType).numAvailable;
        for (size_t vehicle = 0; vehicle != numAvailable; ++vehicle)
            routes.emplace_back(data, rIdx++, vehType);
    }
}

void Solution::load(pyvrp::Solution const &solution)
{
    // Determine offsets for vehicle types.
    std::vector<size_t> vehicleOffset(data_.numVehicleTypes(), 0);
    for (size_t vehType = 1; vehType < data_.numVehicleTypes(); vehType++)
    {
        auto const prevAvail = data_.vehicleType(vehType - 1).numAvailable;
        vehicleOffset[vehType] = vehicleOffset[vehType - 1] + prevAvail;
    }

    for (auto const &solRoute : solution.routes())
    {
        // Determine index of next route of this type to load, where we rely
        // on solution to be valid to not exceed the number of vehicles per
        // vehicle type.
        auto const idx = vehicleOffset[solRoute.vehicleType()]++;
        auto &route = routes[idx];

        if (route == solRoute)  // then the current route is still OK and we
            continue;           // can skip inserting and updating

        // Else we need to clear the route and insert the updated route from
        // the solution.
        route.clear();

        // Routes use a representation with nodes for each client, reload depot
        // (one per trip), and start/end depots. The start depot doubles as the
        // reload depot for the first trip.
        route.reserve(solRoute.size() + solRoute.numTrips() + 1);

        for (size_t tripIdx = 0; tripIdx != solRoute.numTrips(); ++tripIdx)
        {
            auto const &trip = solRoute.trip(tripIdx);

            if (tripIdx != 0)  // then we first insert a trip delimiter.
            {
                Route::Node depot = {trip.startDepot()};
                route.push_back(&depot);
            }

            for (auto const client : trip)
                route.push_back(&nodes[client]);
        }

        route.update();
    }

    // Finally, we clear any routes that we have not re-used or inserted from
    // the solution.
    size_t firstOfType = 0;
    for (size_t vehType = 0; vehType != data_.numVehicleTypes(); ++vehType)
    {
        auto const numAvailable = data_.vehicleType(vehType).numAvailable;
        auto const firstOfNextType = firstOfType + numAvailable;
        for (size_t idx = vehicleOffset[vehType]; idx != firstOfNextType; ++idx)
            routes[idx].clear();

        firstOfType = firstOfNextType;
    }
}

pyvrp::Solution Solution::unload() const
{
    std::vector<pyvrp::Route> solRoutes;
    solRoutes.reserve(data_.numVehicles());

    std::vector<size_t> visits;

    for (auto const &route : routes)
    {
        if (route.empty())
            continue;

        std::vector<Trip> trips;
        trips.reserve(route.numTrips());

        visits.clear();
        visits.reserve(route.numClients());

        auto const *prevDepot = route[0];
        for (size_t idx = 1; idx != route.size(); ++idx)
        {
            auto const *node = route[idx];

            if (!node->isDepot())
            {
                visits.push_back(node->client());
                continue;
            }

            trips.emplace_back(data_,
                               visits,
                               route.vehicleType(),
                               prevDepot->client(),
                               node->client());

            visits.clear();
            prevDepot = node;
        }

        assert(trips.size() == route.numTrips());
        solRoutes.emplace_back(data_, std::move(trips), route.vehicleType());
    }

    return {data_, std::move(solRoutes)};
}

bool Solution::insert(Route::Node *U,
                      SearchSpace const &searchSpace,
                      CostEvaluator const &costEvaluator,
                      bool required)
{
    assert(size_t(std::distance(nodes.data(), U)) < nodes.size());

    // Positions that would use edges missing from the underlying network are
    // not eligible. We track the best eligible position, and - separately -
    // the overall best position as a last resort for required clients.
    Route::Node *UAfter = nullptr;
    Cost bestCost = std::numeric_limits<Cost>::max();
    Route::Node *UAfterAny = routes[0][0];  // fallback option
    Cost bestCostAny = insertCost(U, UAfterAny, data_, costEvaluator);

    // SOFT clients are always inserted when eligible (the compound cost
    // makes insertion always improving), so eligibility must guarantee the
    // insertion does not make the route less feasible. Other clients only
    // need to avoid missing edges; regular cost penalties handle the rest.
    ProblemData::Client const &client = data_.location(U->client());
    auto const isSoft = client.required == pyvrp::ClientRequired::SOFT;

    auto const consider = [&](Route::Node *V)
    {
        auto const cost = insertCost(U, V, data_, costEvaluator);

        if (cost < bestCostAny)
        {
            bestCostAny = cost;
            UAfterAny = V;
        }

        if (cost >= bestCost)
            return;

        auto const eligible = isSoft ? !insertAddsViolation(U, V, data_)
                                     : !insertAddsMissingEdges(U, V, data_);
        if (eligible)
        {
            bestCost = cost;
            UAfter = V;
        }
    };

    consider(routes[0][0]);

    // First attempt a neighbourhood search to place U into routes that are
    // already in use.
    for (auto const vClient : searchSpace.neighboursOf(U->client()))
    {
        auto *V = &nodes[vClient];

        if (V->route())
            consider(V);
    }

    // Next consider empty routes, of each vehicle type. We consider the first
    // empty route of each vehicle type.
    for (auto const &[vehType, offset] : searchSpace.vehTypeOrder())
    {
        auto const begin = routes.begin() + offset;
        auto const end = begin + data_.vehicleType(vehType).numAvailable;
        auto const pred = [](auto const &route) { return route.empty(); };
        auto empty = std::find_if(begin, end, pred);

        if (empty != end)
            consider((*empty)[0]);
    }

    if (UAfter && bestCost < 0)
    {
        auto *route = UAfter->route();
        route->insert(UAfter->idx() + 1, U);
        return true;
    }

    if (required)
    {
        // Required nodes must be inserted. We prefer an eligible position,
        // but fall back to the overall best position if there is none.
        auto *where = UAfter ? UAfter : UAfterAny;
        auto *route = where->route();
        route->insert(where->idx() + 1, U);
        return true;
    }

    return false;
}
