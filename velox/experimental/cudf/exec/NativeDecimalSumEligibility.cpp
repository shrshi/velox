/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/CudfAggregation.h"
#include "velox/experimental/cudf/exec/NativeDecimalSumEligibility.h"

#include "velox/exec/HashPartitionFunction.h"

#include <unordered_map>
#include <unordered_set>

namespace facebook::velox::cudf_velox {
namespace {

using Parents = std::
    unordered_map<const core::PlanNode*, std::vector<const core::PlanNode*>>;

void collectParents(
    const core::PlanNode& node,
    Parents& parents,
    std::unordered_set<const core::PlanNode*>& visited) {
  if (!visited.insert(&node).second) {
    return;
  }
  for (const auto& source : node.sources()) {
    parents[source.get()].push_back(&node);
    collectParents(*source, parents, visited);
  }
}

const core::PlanNode* onlyParent(
    const core::PlanNode& node,
    const Parents& parents) {
  auto it = parents.find(&node);
  return it != parents.end() && it->second.size() == 1 ? it->second[0]
                                                       : nullptr;
}

bool compatibleLayout(
    const core::AggregationNode& partial,
    const core::LocalPartitionNode& exchange,
    const core::AggregationNode& final) {
  using Step = core::AggregationNode::Step;
  const auto numKeys = partial.groupingKeys().size();
  if (partial.step() != Step::kPartial || final.step() != Step::kFinal ||
      numKeys == 0 || final.groupingKeys().size() != numKeys ||
      partial.aggregates().empty() ||
      partial.aggregates().size() != final.aggregates().size() ||
      exchange.type() != core::LocalPartitionNode::Type::kRepartition ||
      exchange.scaleWriter()) {
    return false;
  }

  const auto* hash = dynamic_cast<const exec::HashPartitionFunctionSpec*>(
      &exchange.partitionFunctionSpec());
  if (!hash) {
    return false;
  }
  // HashPartitionFunctionSpec exposes channels only through serialization.
  // Inspect channel numbers rather than parsing field names from toString().
  const auto spec = hash->serialize();
  if (spec["keyChannels"].empty()) {
    return false;
  }
  for (const auto& channel : spec["keyChannels"]) {
    if (channel.asInt() < 0 ||
        channel.asInt() >= static_cast<int64_t>(numKeys)) {
      return false;
    }
  }

  const auto& stateType = partial.outputType();
  for (size_t i = 0; i < numKeys; ++i) {
    const auto& key = final.groupingKeys()[i];
    if (!key->isInputColumn() || key->name() != stateType->nameOf(i) ||
        !key->type()->equivalent(*stateType->childAt(i))) {
      return false;
    }
  }

  const auto sumName = CudfConfig::getInstance().functionNamePrefix + "sum";
  for (size_t i = 0; i < partial.aggregates().size(); ++i) {
    const auto& producer = partial.aggregates()[i];
    const auto& consumer = final.aggregates()[i];
    for (const auto* aggregate : {&producer, &consumer}) {
      if (aggregate->call->name() != sumName ||
          aggregate->rawInputTypes.size() != 1 ||
          !aggregate->rawInputTypes[0]->isShortDecimal() ||
          aggregate->call->inputs().size() != 1 || aggregate->mask ||
          aggregate->distinct || !aggregate->sortingKeys.empty()) {
        return false;
      }
    }
    const auto& rawType = producer.rawInputTypes[0];
    const auto* raw = dynamic_cast<const core::FieldAccessTypedExpr*>(
        producer.call->inputs()[0].get());
    const auto* state = dynamic_cast<const core::FieldAccessTypedExpr*>(
        consumer.call->inputs()[0].get());
    const auto channel = numKeys + i;
    const auto resultType =
        DECIMAL(38, getDecimalPrecisionScale(*rawType).second);
    if (!raw || !raw->isInputColumn() || !state || !state->isInputColumn() ||
        !raw->type()->equivalent(*rawType) ||
        !consumer.rawInputTypes[0]->equivalent(*rawType) ||
        state->name() != stateType->nameOf(channel) ||
        !state->type()->isVarbinary() ||
        !stateType->childAt(channel)->isVarbinary() ||
        !producer.call->type()->isVarbinary() ||
        !consumer.call->type()->equivalent(*resultType) ||
        !final.outputType()->childAt(channel)->equivalent(*resultType)) {
      return false;
    }
  }
  return true;
}

} // namespace

std::vector<bool> nativeDecimalSumEligibility(
    const core::AggregationNode& aggregation,
    const core::PlanNode& planRoot,
    core::QueryCtx* queryCtx,
    memory::MemoryPool* pool) {
  std::vector<bool> eligible(aggregation.aggregates().size(), false);
  using Step = core::AggregationNode::Step;
  if (aggregation.step() != Step::kPartial &&
      aggregation.step() != Step::kFinal) {
    return eligible;
  }

  Parents parents;
  std::unordered_set<const core::PlanNode*> visited;
  collectParents(planRoot, parents, visited);
  if (!visited.count(&aggregation)) {
    return eligible;
  }

  const auto* partial = &aggregation;
  const core::AggregationNode* final = nullptr;
  const core::LocalPartitionNode* exchange = nullptr;
  if (aggregation.step() == Step::kPartial) {
    exchange = dynamic_cast<const core::LocalPartitionNode*>(
        onlyParent(aggregation, parents));
    if (exchange) {
      final = dynamic_cast<const core::AggregationNode*>(
          onlyParent(*exchange, parents));
    }
  } else {
    final = &aggregation;
    if (final->sources().size() == 1) {
      exchange = dynamic_cast<const core::LocalPartitionNode*>(
          final->sources()[0].get());
    }
    if (exchange && exchange->sources().size() == 1) {
      partial = dynamic_cast<const core::AggregationNode*>(
          exchange->sources()[0].get());
    }
  }
  if (!partial || !exchange || !final || exchange->sources().size() != 1 ||
      exchange->sources()[0].get() != partial || final->sources().size() != 1 ||
      final->sources()[0].get() != exchange ||
      onlyParent(*partial, parents) != exchange ||
      onlyParent(*exchange, parents) != final ||
      !compatibleLayout(*partial, *exchange, *final) ||
      !canBeEvaluatedByCudf(*partial, queryCtx, pool) ||
      !canBeEvaluatedByCudf(*final, queryCtx, pool)) {
    return eligible;
  }
  return std::vector<bool>(aggregation.aggregates().size(), true);
}

} // namespace facebook::velox::cudf_velox
