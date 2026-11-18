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
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/utils/CudfFuzzerUtil.h"

#include "velox/exec/Task.h"

#include <folly/Synchronized.h>
#include <glog/logging.h>

#include <map>
#include <set>

namespace facebook::velox::cudf_velox::test {
namespace {

// Counts, for each operator type, the tasks that finished successfully with
// that operator. Tasks may finish on any thread.
class OperatorCounter : public exec::TaskListener {
 public:
  void onTaskCompletion(
      const std::string& /*taskUuid*/,
      const std::string& /*taskId*/,
      exec::TaskState state,
      std::exception_ptr /*error*/,
      exec::TaskStats stats) override {
    if (state != exec::TaskState::kFinished) {
      return;
    }
    std::set<std::string> operatorTypes;
    for (const auto& pipeline : stats.pipelineStats) {
      for (const auto& op : pipeline.operatorStats) {
        operatorTypes.insert(op.operatorType);
      }
    }
    numTasksByOperator_.withWLock([&](auto& numTasksByOperator) {
      for (const auto& operatorType : operatorTypes) {
        ++numTasksByOperator[operatorType];
      }
    });
  }

  void logCounts() const {
    LOG(INFO) << "Number of tasks that ran each operator:";
    numTasksByOperator_.withRLock([](const auto& numTasksByOperator) {
      for (const auto& [operatorType, numTasks] : numTasksByOperator) {
        LOG(INFO) << "  " << operatorType << ": " << numTasks;
      }
    });
  }

 private:
  folly::Synchronized<std::map<std::string, size_t>> numTasksByOperator_;
};

std::shared_ptr<OperatorCounter> operatorCounter;

} // namespace

void registerCudfAndCountOperators() {
  CudfConfig::getInstance().allowCpuFallback = true;
  registerCudf();
  operatorCounter = std::make_shared<OperatorCounter>();
  VELOX_CHECK(exec::registerTaskListener(operatorCounter));
}

void logOperatorCountsAndUnregisterCudf() {
  VELOX_CHECK(exec::unregisterTaskListener(operatorCounter));
  operatorCounter->logCounts();
  operatorCounter.reset();
  unregisterCudf();
}

} // namespace facebook::velox::cudf_velox::test
