# Native GPU State for Decimal64 SUM

## Goal

Avoid materializing the 32-byte `VARBINARY` decimal SUM state between compatible GPU partial and final aggregations.

This first version is deliberately narrow:

- aggregate: decimal `SUM` only;
- raw input: `DECIMAL64` only;
- producer: `CudfGroupbyPARTIAL`;
- transport: GPU `CudfLocalPartition`;
- consumer: `CudfGroupbyFINAL`;
- fallback: materialize the existing CPU-compatible `VARBINARY` state at every unsupported boundary.

The motivating Q18 pipeline is:

```text
CudfGroupbyPARTIAL(l_orderkey, sum(l_quantity DECIMAL64))
    -> CudfLocalPartition HASH(l_orderkey)
    -> CudfGroupbyFINAL(l_orderkey, sum(partial state))
```

The plan's logical intermediate type remains `VARBINARY`. Only its physical representation inside a compatible GPU pipeline changes.

## Why the existing representation is expensive

The existing intermediate state mirrors `LongDecimalWithOverflowState`:

```text
count:     INT64       8 bytes
overflow:  INT64       8 bytes
sum:       INT128     16 bytes
                       --------
payload:               32 bytes per group
```

On GPU this payload is represented as a cuDF STRING column, which also needs string offsets and a validity mask. Q18 emits 1.5 billion partial groups, so both the payload and the temporary decode/encode columns are large.

The final incremental group-by may repeatedly:

1. decode the incoming STRING into sum and count columns;
2. concatenate it with buffered state;
3. group the combined state;
4. encode the result back into STRING.

The proposed representation avoids these conversions while the state remains in the GPU pipeline.

## Native representation

Use one nullable `DECIMAL128` cuDF column:

```text
physical data:  DECIMAL128 partial sum
physical null:  group has no non-null input
logical type:   VARBINARY
encoding tag:   NativeDecimal64SumState
```

No empty or placeholder `VARBINARY` values are produced. The native column is the physical value. A real `VARBINARY` column is created only when an unsupported consumer requires the standard intermediate representation.

A STRUCT column is unnecessary for this first version because the state has only one value plus validity.

## Why sum plus validity is sufficient

For a SUM whose raw input type is `DECIMAL64`:

- the exact input count is not needed to merge sums;
- validity distinguishes an empty/all-null group from a valid sum of zero;
- a `DECIMAL64` value widens exactly to signed 128 bits;
- the aggregate's global row count is represented by signed 64-bit counters;
- therefore, even a conservative bound using an `INT64` value and `INT64_MAX` rows is below the signed 128-bit range.

In particular:

```text
(2^63 - 1) * (2^63 - 1) < 2^126 < 2^127
```

Actual DECIMAL64 values have decimal precision at most 18 and are more tightly bounded. Combining any valid set of DECIMAL64 inputs therefore does not require the long-decimal overflow counter used for DECIMAL128 raw inputs.

The final result must still perform the normal `DECIMAL(38, scale)` range validation. This optimization does not relax SQL overflow checks.

This reasoning does not apply to:

- raw `DECIMAL128` inputs;
- AVG, which needs the count;
- a state whose raw input width cannot be proven to be DECIMAL64;
- arbitrary CPU-produced `VARBINARY` states, which may contain an overflow count.

Those cases continue to use the existing representation.

## Encoding metadata

Do not infer native state solely from a mismatch between the logical Velox type and cuDF type. Add explicit per-column physical-encoding metadata to `CudfVector`, for example:

```cpp
enum class CudfPhysicalEncoding {
  kDefault,
  kNativeDecimal64SumState,
};
```

The metadata must:

- have one entry per logical top-level column;
- survive selection, slicing, splitting, concatenation, and local partitioning;
- be rejected or materialized by operators that do not understand it;
- participate in validation and debugging output;
- not change the logical `RowType` seen by the plan.

For `kNativeDecimal64SumState`, validate that the physical column is nullable `DECIMAL128` with the expected scale.

## Producer behavior

When `CudfGroupbyPARTIAL` handles decimal SUM with raw `DECIMAL64` input and the downstream path supports native state:

1. aggregate into a nullable `DECIMAL128` result per group;
2. use result validity to represent whether the group observed a non-null value;
3. emit that column in the logical `VARBINARY` output position;
4. mark the column `kNativeDecimal64SumState`;
5. do not call `serializeDecimalSumState`.

The initial implementation should enable this only when compatibility of the immediate path is known. Otherwise, retain the existing `VARBINARY` output.

## Local exchange behavior

`CudfLocalPartition` already passes `CudfVector` objects through local exchange queues and uses `cudf::hash_partition` on their physical tables. A nullable fixed-width `DECIMAL128` column can be partitioned along with its grouping key without serialization.

Required changes:

1. preserve physical-encoding metadata when constructing each partition vector;
2. keep the native state column out of the partition-key set unless it is logically a key;
3. account for its actual retained GPU bytes;
4. ensure queueing, splitting, and stream ordering preserve the column and its null mask.

There is no HTTP or page serialization in this Q18 local-exchange path.

## Consumer behavior

When `CudfGroupbyFINAL` receives a column tagged `kNativeDecimal64SumState`:

1. validate that the aggregate is decimal SUM whose raw input type was DECIMAL64;
2. consume the physical `DECIMAL128` column directly;
3. group and sum it using null exclusion;
4. preserve the resulting validity;
5. retain the same native encoding for any internal incremental buffered result;
6. cast or validate the final result as `DECIMAL(38, scale)` when producing final query output.

The final operator must not call `deserializeDecimalSumState` for native input.

For incremental final aggregation, both the incoming and buffered state should remain native. Concatenation then operates directly on nullable fixed-width columns instead of STRING payloads.

## Materialization boundary

Provide one explicit conversion:

```text
NativeDecimal64SumState -> standard 32-byte VARBINARY state
```

Materialization is required before:

- CPU execution or fallback;
- remote/HTTP exchange;
- generic spill or row serialization that does not preserve the encoding;
- conversion to an ordinary Velox vector;
- an expression or operator that treats the value as actual VARBINARY;
- mixing with an untagged or CPU-produced decimal SUM state;
- any unsupported local-exchange implementation.

To produce the existing state for each valid native row:

```text
count = 1
overflow = 0
sum = native DECIMAL128 sum
```

For SUM, `count = 1` is sufficient: downstream merging only needs zero versus nonzero to preserve NULL semantics. A null native row becomes a null intermediate state. This conversion must be verified against CPU partial, intermediate, and final aggregate paths.

The reverse conversion from standard `VARBINARY` to native state is safe only after validating that the state originated from DECIMAL64 semantics and has no nonzero overflow. It is not required for the first version.

## Eligibility and fallback

Use native state only if all of the following are true:

- function is decimal SUM;
- original raw input is `DECIMAL64`;
- aggregation step is partial or an internal intermediate merge;
- immediate consumers preserve or consume the encoding;
- no remote exchange, spill boundary, CPU conversion, or unsupported operator intervenes.

If eligibility cannot be proven, produce the existing standard `VARBINARY` state. Correct fallback is more important than maximizing coverage.

A mixed batch sequence must not silently combine native and serialized representations. Either materialize native batches before mixing or keep the entire edge on one representation.

## Implementation assumptions to recheck

The implementation currently relies on the following assumptions. These are recorded explicitly so they can be challenged before broadening the optimization.

### Activation and plan assumptions

- Native eligibility is per aggregate and does not require PARTIAL, exchange, and FINAL adjacency.
- Only grouped PARTIAL and FINAL aggregation steps participate; global reduction remains outside this implementation.
- The aggregate is the configured ordinary `SUM` function, not a companion function.
- The aggregate has one direct field argument, one raw input type, no mask, no `DISTINCT`, and no ordering.
- The raw semantic input type is a short decimal (`DECIMAL64`). A physical `DECIMAL128` state alone is not proof of DECIMAL64 provenance.
- Aggregate result columns appear after grouping-key output columns in aggregate order. Input channels may be renamed or reordered and are resolved by the aggregation input-channel mapping.
- Eligibility is per aggregate: unsupported sibling aggregates do not disable eligible SUM columns.
- Built-in cuDF operator replacement remains independently responsible for deciding whether an operator runs on GPU. Native metadata must not be interpreted as proof that a custom adapter preserved it.
- Shared plans and multiple drivers are valid as long as every runtime branch preserves the encoding or materializes before representations mix.
- Hash, gather, and round-robin transport may carry native state as payload. If the native state is itself a partition key, it is materialized first.

### Mathematical and SQL-semantic assumptions

- Sum plus validity is sufficient for SUM: validity distinguishes no non-null input from a valid zero, and exact count is not needed when merging SUM states.
- The total represented raw input contribution count is bounded by signed 64-bit row-count semantics, so valid DECIMAL64 inputs cannot overflow signed 128-bit accumulation.
- Operators that duplicate already-aggregated state rows, such as many joins, may invalidate that contribution-count argument. Until a sufficient bound or checked accumulation exists, such operators remain materialization boundaries.
- Final native output is separately validated against the unscaled `DECIMAL(38)` range. That validation does not detect an earlier signed-128 overflow if the contribution-count assumption is violated.
- Materializing a valid native SUM with `count = 1` and `overflow = 0` preserves SUM merge and null semantics. It is not valid for AVG or another count-bearing state.
- cuDF SUM excludes nulls and returns a null result for an all-null group.
- Raw DECIMAL128 SUM, AVG, arbitrary CPU-produced VARBINARY state, and state with a nonzero overflow counter are not native-compatible.

### Representation and operator assumptions

- Encoding metadata applies to logical top-level columns only; nested native state is unsupported.
- Metadata carries the encoding family and decimal scale, but not full plan lineage or a represented-contribution bound.
- A valid native column is physical `DECIMAL128` at the recorded scale. All-valid data need not allocate a null mask.
- A forwarding operator must preserve the physical column, validity, and metadata together. Identity projection, rename, reordering, dropping other columns, filtering on other columns, slicing, limiting, sorting, and partitioning are safe when metadata is mapped to the resulting channels.
- An expression, filter, grouping key, partition key, comparison, hash, or sort that reads the logical VARBINARY value must materialize that column first unless it explicitly implements native semantics.
- `CudfFilterProject` discovers top-level field use recursively. Identity outputs preserve metadata; computed outputs use default encoding; state columns read by expressions are selectively materialized.
- Concatenation requires one representation per logical column. If batches disagree, that column is normalized to serialized VARBINARY in all native batches; unrelated native columns stay native.
- A FINAL aggregate validates the incoming encoding and scale per aggregate. It may consume native and serialized sibling SUM states in the same batch.
- If a column changes from native to serialized after FINAL has buffered native state, the corresponding buffered column is materialized before concatenation. Serialized state is not converted back to native.
- CPU conversion, remote/HTTP exchange, generic spill or row serialization, unsupported operators, and unknown expression use remain materialization boundaries.
- Stream ordering and GPU buffer lifetime remain governed by existing cuDF operations; metadata adds no synchronization.
- Retained-byte accounting uses physical GPU storage rather than the logical VARBINARY type.

### Current operator support

The following operators preserve native state or handle it selectively:

- `CudfGroupby`;
- `CudfFilterProject`;
- `CudfLimit`;
- `CudfLocalPartition`; and
- `CudfBatchConcat`.

Even these operators materialize individual columns when required:

- `CudfFilterProject` materializes a state read by a filter or computed expression;
- `CudfLocalPartition` materializes a state used as a partition key;
- `CudfGroupby` materializes an ineligible state, a state with mismatched scale or encoding, or buffered state when an input column switches to serialized representation; and
- concatenation materializes a logical column when input batches disagree on its physical encoding, while preserving unrelated native columns.

Every `CudfOperatorBase` subclass that does not opt in through `acceptsNativeDecimalSumState()` materializes all incoming native-state columns before `doAddInput()`. The current unsupported operators are:

- `CudfAssignUniqueId`;
- `CudfDistinct`;
- `CudfEnforceSingleRow`;
- `CudfGroupId`;
- `CudfHashJoinProbe`;
- `CudfJoinBuild`;
- `CudfMarkDistinct`;
- `CudfNestedLoopJoinProbe`;
- `CudfOrderBy`;
- `CudfReduce`;
- `CudfTopN`;
- `CudfTopNRowNumber`;
- `CudfWindow`; and
- `CudfToVelox`, before conversion to ordinary Velox vectors.

`CudfFromVelox` also inherits the default behavior, but normally receives ordinary Velox vectors and therefore cannot receive native GPU state.

Some unsupported operators could transport native state after adding channel-aware metadata propagation. Joins require separate analysis because duplicating an already-aggregated state can invalidate the current signed-128 contribution bound.

## Q18 implementation gap found during profiling

The first implementation has the producer, consumer, metadata, materialization, and local-partition plumbing described above, but it does not activate for Q18. A two-iteration managed-memory profile therefore retained the previous peak of approximately 281.37 GiB. The profile still contained decimal state packing and unpacking kernels in the critical partial and final group-bys.

Targeted logs confirmed that eligibility failed before native state was produced:

```text
node=1717, step=PARTIAL, eligible=0
node=9,    step=FINAL,   eligible=0
```

The partial output and final input consequently remained in the default encoding:

```text
node=1717, encoding=Default, Default
node=9, matches=0, nativeInput=0, inputEncoding=Default, Default
```

### Why eligibility rejects Q18

The initial eligibility analysis requires exact logical adjacency:

```text
Aggregate(PARTIAL)
    -> LocalPartition
    -> Aggregate(FINAL)
```

It verifies this with direct parent relationships. The actual Q18 GPU pipeline contains a transparent filter/project operator between the partial aggregation and local exchange:

```text
CudfBatchConcat[1717]
    -> CudfGroupbyPARTIAL[1717]
    -> CudfFilterProject[1719.0]
    -> CudfLocalPartition
    -> CudfGroupbyFINAL[9]
```

For this query the intervening `CudfFilterProject` preserves the partial output needed by the exchange, but its presence means that the partial aggregation's direct parent is not the `LocalPartition`. The strict path matcher therefore marks both the partial and final aggregations ineligible.

This is why the initial compact-state profile did not measure the optimization: it measured the existing serialized `VARBINARY` path.

### Why relaxing eligibility alone is insufficient

`CudfFilterProject` is not currently a transparent carrier of native physical encodings:

1. it does not opt in through `acceptsNativeDecimalSumState()`;
2. `CudfOperatorBase::addInput()` therefore materializes native state before passing it to the operator;
3. it releases and rebuilds the cuDF table; and
4. its output `CudfVector` is constructed without propagated physical-encoding metadata.

Simply allowing the plan-path matcher to traverse this node would cause the producer to emit native state only for the operator boundary to convert it immediately back to `VARBINARY` or lose its encoding metadata.

### Implemented direction

Eligibility is now independent of adjacency. Operators decide locally whether they preserve an opaque native column or selectively materialize it. The following original proposal records how the gap was identified; transport is no longer limited to discovering transparent nodes through plan traversal.

#### 1. Traverse transparent nodes during eligibility analysis

Change the partial-to-final path analysis to recognize:

```text
Aggregate(PARTIAL)
    -> zero or more transparent nodes
    -> LocalPartition
    -> zero or more transparent nodes
    -> Aggregate(FINAL)
```

A node is transparent for a native state column only when it preserves that logical field as an identity projection. Eligibility must be checked per aggregate column rather than assuming that every column passing through the node is safe.

Do not traverse a filter/project when it:

- computes an expression from the state;
- uses the state in a filter;
- casts or otherwise transforms the state;
- drops the state required by the final aggregate;
- changes its logical type;
- routes it through an unsupported CPU operator; or
- creates an ambiguous mapping that cannot be tracked safely.

The traversal must return the input-to-output channel mapping so downstream encoding metadata is attached to the correct output channel even when columns are reordered.

#### 2. Preserve metadata in `CudfFilterProject`

For supported identity projections, `CudfFilterProject` should:

- override `acceptsNativeDecimalSumState()`;
- capture input physical encodings before releasing the input vector;
- propagate each identity-projected column's encoding to its output channel;
- preserve scale metadata;
- construct the output `CudfVector` with the resulting encoding vector; and
- leave computed output columns at the default encoding.

If a native state column is filtered, the row operation itself is safe only if cuDF applies the same retention mask to the native data and its validity mask. If the filter predicate reads the logical `VARBINARY` state, materialize before evaluation. The narrowest first fix should support the Q18 projection-only bridge and continue to materialize for all other cases.

#### 3. Keep materialization as the fallback

If any bridge cannot prove that it preserves the native column exactly, use the existing native-to-`VARBINARY` conversion before that operator. Correctness must not depend on physical metadata surviving a generic operator accidentally.

#### 4. Verify activation before profiling again

A successful Q18 run should log:

```text
node=1717, step=PARTIAL, eligible=1
node=1717, encoding=Default, NativeDecimal64SumState(scale=2)
node=9,    step=FINAL,   eligible=1
node=9,    matches=1, nativeInput=1,
           inputEncoding=Default, NativeDecimal64SumState(scale=2)
```

It should also report native rows produced and consumed, with no materialized values on this edge. Only after these conditions hold is another Nsight peak comparison meaningful.

## Initial implementation stages

### Stage 1: Native state plumbing

- Add physical-encoding metadata to `CudfVector`.
- Preserve it through the required table and local-exchange operations.
- Add strict physical/logical validation.
- Add explicit native-to-VARBINARY materialization.

### Stage 2: Decimal SUM producer and consumer

- Emit nullable `DECIMAL128` from eligible partial group-by.
- Consume it directly in final group-by.
- Keep incremental buffered state native.
- Preserve the existing path as fallback.

### Stage 3: Plan-path eligibility

- Enable native output only for a fully compatible GPU edge.
- Materialize before unsupported boundaries.
- Add runtime counters for native production, consumption, and materialization.

### Stage 4: Validation and profiling

Test:

- positive, negative, and zero sums;
- cancellation to zero;
- null and all-null inputs;
- mixed null/non-null inputs;
- maximum and minimum DECIMAL64 values;
- final DECIMAL(38) range checks;
- multiple partial batches and drivers;
- hash local exchange;
- forced materialization followed by CPU final aggregation;
- unsupported-path fallback;
- metadata preservation through split, concatenate, and partition.

Profile Q18 with the same two-driver managed-memory configuration and compare:

- partial output retained bytes;
- final `addInput` peak by iteration;
- STRING payload and offset allocations;
- decode/encode temporary allocations;
- runtime and Unified Memory migration behavior.

## Expected Q18 impact

The native state uses approximately:

```text
DECIMAL128 data: 16 bytes per group
validity:         1 bit per group
```

It replaces a 32-byte packed payload plus STRING offsets and avoids repeated decode/encode materialization. At 1.5 billion groups, eliminating 16 payload bytes alone represents about 24 GB decimal (22.35 GiB) per full materialized copy, with additional savings from removed offsets and temporary columns.

The exact peak reduction may be larger because the current incremental final group-by can retain incoming, buffered, concatenated, decoded, and output state concurrently. Hash-table and grouping-key allocations remain and must be measured separately.

## Non-goals for the first version

- native state for DECIMAL128 raw input;
- native decimal AVG state;
- changing the planner-visible intermediate type;
- remote exchange support for native state;
- generic native state support for all aggregates;
- replacing the existing CPU-compatible serialized format.
