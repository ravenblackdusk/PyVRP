#include "CostEvaluator.h"

#include <algorithm>
#include <stdexcept>

using pyvrp::CostEvaluator;

CostEvaluator::CostEvaluator(std::vector<double> loadPenalties,
                             double twPenalty,
                             double distPenalty,
                             double missingSoftPenalty,
                             double maxMissingSoftPenalty)
    : loadPenalties_(std::move(loadPenalties)),
      twPenalty_(twPenalty),
      distPenalty_(distPenalty),
      missingSoftPenalty_(missingSoftPenalty),
      maxMissingSoftPenalty_(std::max(missingSoftPenalty, maxMissingSoftPenalty))
{
    for (auto const penalty : loadPenalties_)
        if (penalty < 0)
            throw std::invalid_argument("load_penalties must be >= 0.");

    if (twPenalty_ < 0)
        throw std::invalid_argument("tw_penalty must be >= 0.");

    if (distPenalty_ < 0)
        throw std::invalid_argument("dist_penalty must be >= 0.");

    if (missingSoftPenalty_ < 0)
        throw std::invalid_argument("missing_soft_penalty must be >= 0.");
}
