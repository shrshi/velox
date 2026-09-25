# Q18 Final Group-by: Further Memory Reduction Ideas

## Goal

Reduce Q18 SF1000 final group-by peak allocation while retaining two drivers and exact decimal semantics. These are investigation and implementation proposals, not measured improvements.

## Measured baseline after native decimal state compaction

Profile in the sibling `velox-testing` repository:

```text
presto/scripts/q18_managed_memory_compact_decimal_updated_profile/profiles/tpch/Q18.nsys-rep
```

Velox revision: `53bc297a6`.

Configuration:

```properties
task.max-drivers-per-task=2
cudf.memory_resource=managed
cudf.concat_optimization_enabled=true
cudf.batch_size_min_threshold=100000000
cudf.batch_size_max_threshold=500000000
```

| Iteration | Previous serialized-state peak | Native-state peak | Native-state runtime |
|---|---:|---:|---:|
| 1 | 274.82 GiB | 136.97 GiB | 148.047 s |
| 2 | 281.37 GiB | 145.61 GiB | 149.491 s |

Both new peaks occur during overlapping `CudfGroupbyFINAL::addInput [9]` calls. Decimal state pack, unpack, and offset-generation kernels have disappeared. The largest individual allocations are now approximately 11.18 GiB rather than 22.35 GiB.

These are live CUDA allocation totals, not physical GPU residency. They include concurrent operators, not just memory exclusively owned by node 9.

Q18's important aggregation is:

```sql
SELECT l_orderkey
FROM lineitem
GROUP BY l_orderkey
HAVING sum(l_quantity) > 300
```

Runtime cardinalities:

```text
Final input:  1,500,000,000 partial rows
Final output: 1,500,000,000 groups
HAVING output:       63,430 rows
```

The native key/state payload is approximately:

```text
1.5B × (8-byte key + 16-byte sum) = 36 GB = 33.53 GiB
```

Null masks and algorithmic workspace are additional. The gap between this payload and the 145.61 GiB process peak motivates reducing overlapping copies and grouping workspace.

## 1. Release obsolete inputs after concatenation

### Current behavior

`CudfGroupby::computeFinalGroupbyIncrementally()` in `CudfGroupby.cpp` concatenates the buffered result and incoming batch, runs group-by, and only then replaces `bufferedResult_`.

```text
old buffered state + incoming batch
+ concatenated table
+ group-by workspace
+ replacement output
```

The old buffered table stays alive throughout the group-by even though concatenation has already copied its contents.

### Proposed experiment

Release obsolete owners after concatenation has finished reading their buffers, before allocating the next group-by workspace/output:

```text
old buffered state + incoming batch
    -> concatenate
    -> release obsolete inputs
    -> group-by on concatenated table
```

At about 750M groups per driver, one key/sum table is approximately 16.76 GiB. Removing two large old buffered tables from the overlap could save tens of GiB, depending on ownership and allocation timing.

### Safety and validation

- Establish CUDA stream ordering before releasing buffers read asynchronously by concatenation.
- Trace all shared owners: resetting one local pointer may not free storage. In the current call chain, `doAddInput()` passes `cudfInput` without moving it and its `input` parameter is another shared owner, so releasing only `bufferedResult_` and the callee's `tbl` is insufficient.
- Prefer stream-ordered destruction or rebinding over a full stream synchronization, which would reduce overlap.
- Account for retained views and aggregator-owned columns.
- Preserve error cleanup behavior.
- Compare allocation lifetimes at the same late final-aggregation updates, not just overall peak.

This is the first change to investigate because it may require only a localized lifetime adjustment.

## 2. Avoid the gathered Decimal128 value temporary

The sort-based SUM path in `cudf/cpp/src/groupby/sort/aggregate.cpp` calls:

```cpp
detail::group_sum(
    get_grouped_values(),
    helper.num_groups(stream),
    helper.group_labels(stream),
    stream,
    mr);
```

Investigate replacing materialized reordered values with indirect reads through the grouping permutation:

```text
Current:
original values -> gathered values -> segmented SUM

Proposed:
original values + grouping permutation -> segmented SUM
```

One Decimal128 column at 750M rows occupies approximately 11.18 GiB. This matches several observed allocations, but allocation-site attribution is required before claiming those allocations are gathered-value buffers.

Potential benefit: avoid a full-sized value copy per active driver.

Trade-offs and requirements:

- Indirect loads may reduce memory-access efficiency.
- Null masks, slices, and grouping order must remain correct.
- Existing algorithms may require contiguous grouped input; a new iterator-aware reduction path may be needed.
- This is a cuDF implementation change, not a configuration change.

## 3. Exact unique-key fast path

Q18's measured final input row count equals its final output group count. For this execution, the final stage performs no cardinality reduction.

After exact grouping/uniqueness detection, if every retained key occurs once, no SUM merge is necessary:

```text
output sum = corresponding input partial sum
```

A specialized path could skip value gathering and reduction output allocation, forwarding or moving columns when ordering and ownership permit.

Correctness requirements:

- Prove uniqueness over the accumulated input, not just within one batch.
- Do not infer uniqueness from estimates or assume it from TPC-H data generation.
- Future batches can repeat keys; retain enough state to handle subsequent merges.
- Preserve null-key policy, all-null state, and final decimal range validation.
- Account for any required output ordering or key/value permutation.

This does not eliminate the cost of establishing uniqueness, but may avoid large subsequent value-processing allocations. If global uniqueness can instead be proven by a valid planner property, a broader aggregation-elimination optimization may be possible; the profile alone is not such a proof.

## 4. Partitioned final aggregation with incremental output

Replace a single approximately 750M-group working set per driver with bounded hash buckets:

```text
partial states
    -> partition by grouping key into N buckets
    -> aggregate a bounded number of buckets concurrently
    -> emit completed results incrementally
```

Rows for the same key must always enter the same bucket. A bucket can be finalized only after no more contributions can arrive.

Memory then consists of retained bucket inputs plus workspace for active buckets, rather than a full-state concatenation and full-state group-by working set.

Important limitations:

- Bucketing alone does not remove retained input memory.
- A bounded-device-memory design needs inactive buckets in host memory/spill, or an explicit residency strategy.
- Skew requires handling; equal hash ranges do not guarantee equal bucket sizes.
- Avoid constructing all bucket outputs concurrently.
- Do not concatenate all completed output buckets into another giant table.
- Apply backpressure to bound active work across both drivers.

This is the strongest general bounded-memory direction, but is a larger operator change.

## 5. Filter completed groups before retaining full output

Only 63,430 of Q18's 1.5B groups pass HAVING. Combine bucketed/chunked finalization with filtering:

```text
finalize bucket
    -> apply SUM(quantity) > 300
    -> retain qualifying order keys
    -> release completed bucket state
```

This can avoid keeping a huge completed result while processing later buckets.

Do not discard incomplete groups because their current sum is below 300. The predicate is safe only once a group is complete. General decimal SUM also allows negative contributions.

HAVING fusion alone does not remove the observed `addInput` peak. It is most useful together with incremental finalization.

## 6. Memory-budgeted concurrency without reducing driver count

Keep two drivers but gate the largest group-by allocation phases through a shared memory budget:

```text
Driver A: large final group-by update
Driver B: waits before allocating large workspace
```

Scans and other work can remain parallel while avoiding simultaneous peak workspaces.

Requirements:

- Use scheduler-aware blocking, not blocking GPU worker threads indefinitely.
- Avoid deadlocks involving exchange backpressure and retained input buffers.
- Bound actual workspace or use a conservative estimate.
- Measure throughput loss against memory savings.

This addresses the observed overlap directly without changing the query-wide driver count.

Admission should happen before accepting or materializing another large input where possible. Otherwise, the waiting driver can retain a large batch and limit the peak reduction.

## 7. Merge sorted aggregated runs instead of regrouping all prior state

If each input batch is reduced to a key-sorted aggregated run and `bufferedResult_` is also sorted, update the state with a merge-reduction:

```text
sorted accumulated state
+ sorted newly aggregated batch
    -> merge by key
    -> add states only for matching keys
    -> new sorted accumulated state
```

This avoids concatenating and sorting all previously accumulated groups again on every update. For Q18, where the measured final input and output cardinalities are equal, most of the work may reduce to merging disjoint keys. Unlike a unique-key forwarding path, it also handles duplicates in later batches naturally.

Requirements and limitations:

- Confirm that the selected group-by path provides a documented key order, or explicitly sort each run.
- A simple merge still retains both inputs while allocating its output; use chunked merging or spill to bound that overlap.
- Preserve decimal overflow validation, null-key policy, and state encodings while combining matching rows.
- Do not concatenate all merge chunks into another full-sized table.
- Compare total bytes processed as well as peak memory: this should avoid repeatedly sorting and gathering the largest prior run.

This direction may subsume much of the exact unique-key fast path because disjoint sorted ranges can be copied or forwarded while duplicate keys are reduced.

## 8. Maintain size-tiered aggregated runs

Instead of merging every batch immediately into one ever-growing result, retain a small number of sorted aggregated runs organized by size and merge only runs of similar size:

```text
level 0: batch-sized runs
level 1: approximately 2x runs
level 2: approximately 4x runs
...
```

This LSM-style approach prevents every small batch from rewriting the entire accumulated state. Each row participates in logarithmically many merges rather than every subsequent update.

It still requires a memory budget and spill policy because several runs can coexist. Limit the number of resident runs, compact one pair at a time, and perform the final multiway merge incrementally so the optimization does not end by materializing all runs together.

## 9. Use a thresholded state when non-negativity is proven

Q18 outputs the order key, not the exact sum, and only tests `sum(l_quantity) > 300`. If the planner can prove that all contributions are nonnegative, the state can saturate at the first value above the threshold:

```text
state = min(state + quantity, 301)
```

After a group crosses the threshold, its exact sum and later contributions are irrelevant to this query. A narrow saturating state could replace Decimal128 state and avoid decimal overflow while evaluating this predicate.

This optimization is invalid based only on knowledge of the TPC-H data generator. Non-negativity must follow from a trusted column constraint, predicate, or planner property. Without that proof, later negative contributions can bring a sum back below the threshold. The general decimal SUM path must remain available as a fallback.

## 10. Store narrow exact state with checked widening

Exact decimal semantics do not necessarily require every intermediate group state to occupy 128 bits. Keep a run or partition in Decimal64 while checked additions prove that all values remain representable, and transactionally widen the whole run or partition to Decimal128 before an unsafe merge.

This could reduce the value payload from 16 to 8 bytes for groups that never need widening. Per-run or per-partition widening is likely simpler than a mixed-width representation per row.

Correctness requirements:

- Detect overflow before destructive updates.
- Preserve scale and final DECIMAL(38) range validation.
- Make widening transactional so allocation failure leaves the old state valid.
- Measure widening overlap; holding both narrow and wide copies can temporarily offset the savings.

## 11. Propagate planner and exchange key properties

Investigate whether execution can provide mechanically guaranteed properties such as:

- keys are unique within an input run;
- incoming key ranges are disjoint from retained ranges;
- input is ordered by grouping key;
- all rows for a hash or range partition have arrived.

These properties could permit aggregation elimination, append instead of merge, or early partition finalization. They must be guaranteed by partitioning and exchange protocols rather than inferred from observed cardinalities or estimates.

## 12. Adapt input batch size as retained state grows

Reducing incoming batch size near the memory limit can lower overlap between the incoming table, concatenated input, and group-by workspace. Couple this with admission control and derive the batch budget from retained bytes plus estimated temporary and output bytes.

This is only a guardrail. With the current concatenate-and-regroup algorithm, smaller batches can increase the number of times the full accumulated state is processed and make runtime worse. It is more suitable alongside merge-based updates or partitioned aggregation.

## Why enabling streaming group-by is not an immediate solution

The current Velox streaming aggregator factory rejects decimal aggregates. libcudf's streaming group-by also rejects DECIMAL128 SUM because its hash-update path requires unsupported 128-bit atomic accumulation. The compact state's physical type is DECIMAL128 even though its raw input was DECIMAL64.

Supporting this path requires a correct Decimal128 update mechanism and native-state-aware Velox integration. Removing the eligibility guard alone is insufficient.

Persistent streaming group-by could avoid repeated concatenation/regrouping, but its memory must still be evaluated:

- approximately 750M logical groups per driver;
- hash-table capacity headroom and roughly 0.5 load factor;
- old and replacement tables coexisting during growth;
- persistent state and output coexisting during finalization.

Also, fewer drivers do not imply proportionally smaller final state: one driver would own all 1.5B groups. Memory comparisons must distinguish per-driver partition size from overlap of temporary workspaces. The existing streaming adapter's safe capacity is approximately 1.074B logical keys, so a single streaming instance cannot simply hold this full Q18 cardinality.

## Investigation order

1. Attribute the remaining 11.18 GiB allocations to exact cuDF allocation sites and owners. Record logical live bytes, allocator-reserved bytes, and managed-memory residency separately.
2. Transfer ownership correctly and shorten the old buffered table/input lifetimes after concatenation using stream-ordered destruction.
3. Evaluate memory-budgeted admission before another large input is retained or materialized.
4. Determine whether grouped runs can be maintained in a documented sorted order, then prototype merge-reduction instead of concatenate-and-regroup.
5. If repeated updates remain necessary, evaluate size-tiered aggregated runs.
6. Prototype indirect or fused reduction to remove the gathered Decimal128 temporary.
7. Evaluate an exact unique-key fast path only where uniqueness or key disjointness can be established before paying ordinary grouping costs.
8. Evaluate checked narrow decimal state with per-run or per-partition widening.
9. For robust bounded memory, design bucketed aggregation with incremental output and HAVING filtering.
10. Investigate planner-proven non-negativity, uniqueness, ordering, and key-domain properties for specialized paths.
11. Use adaptive batch sizing as a guardrail after accounting for its additional regrouping cost.

For each experiment, preserve the two-driver managed-memory baseline and report both iteration peaks, managed allocation separately, allocator-reserved and resident memory when available, query runtime, total bytes processed, and result validation. Only test device-only allocation after the measured working set has adequate headroom below physical VRAM.
