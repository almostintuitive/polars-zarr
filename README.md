# Polars IO plugin for reading Zarr v3 arrays

This plugin provides a way to read [Zarr v3](https://zarr.dev/) arrays and groups as [Polars](https://pola.rs/) DataFrames.

[Zarr](https://zarr.dev/) is a format for storing chunked, compressed, N-dimensional arrays, designed for use in parallel computing. This plugin enables reading Zarr data directly into Polars LazyFrames, supporting lazy evaluation with predicate and projection pushdown.

## Features

- **Lazy reading**: Uses Polars' `register_io_source` for lazy evaluation with query optimization
- **Chunked streaming**: Reads data in chunks matching Zarr's chunk structure for memory efficiency
- **Multiple array types**:
  - 1D arrays: Read as single-column DataFrames
  - 2D arrays: Read as DataFrames where rows correspond to the first dimension
  - Groups with multiple 1D arrays: Each array becomes a column
- **Predicate pushdown**: Filter operations are pushed down to reduce memory usage
- **Projection pushdown**: Only requested columns are read from disk

## Requirements

- Python >= 3.11
- Polars >= 1.22.0
- Zarr >= 3.0.8
- NumPy

## Installation

```bash
pip install polars-zarr
```

Or install from source:

```bash
pip install git+https://github.com/ghuls/polars-zarr.git
```

## Usage

### Reading a 2D array

```python
import numpy as np
import polars_zarr
import zarr

# Create a sample zarr array
z = zarr.create_array(
    store="my_data.zarr",
    shape=(1000, 5),
    chunks=(100, 5),
    dtype="float64",
)
z[:] = np.random.randn(1000, 5)

# Read lazily with polars_zarr
lf = polars_zarr.scan_zarr("my_data.zarr")

# Apply transformations - filter and projection are pushed down
result = (
    lf
    .select(["0", "2"])  # Only read columns 0 and 2
    .filter(pl.col("0") > 0)  # Filter rows where column 0 is positive
    .head(100)
    .collect()
)
```

### Reading with custom column names

```python
# Read with custom column names
df = polars_zarr.scan_zarr(
    "my_data.zarr",
    column_names=["a", "b", "c", "d", "e"]
).collect()
```

### Reading a zarr group with multiple arrays

```python
import zarr
import numpy as np

# Create a zarr group with multiple arrays
root = zarr.open_group("my_group.zarr", mode="w")
root.create_array("id", data=np.arange(100), chunks=(20,))
root.create_array("value", data=np.random.randn(100), chunks=(20,))
root.create_array("label", data=np.random.randint(0, 10, size=100), chunks=(20,))

# Read the group - each array becomes a column
df = polars_zarr.scan_zarr("my_group.zarr").collect()
print(df.columns)  # ['id', 'value', 'label']
```

### Limiting rows

```python
# Read only the first 50 rows
df = polars_zarr.scan_zarr("my_data.zarr", n_rows=50).collect()
```

## API Reference

### `scan_zarr`

```python
def scan_zarr(
    source: str | Path,
    *,
    column_names: Sequence[str] | None = None,
    rechunk: bool = False,
    n_rows: int | None = None,
) -> pl.LazyFrame:
```

**Parameters:**

- `source`: Path to a Zarr array or group
- `column_names`: Optional list of column names. For 2D arrays, specifies names for each column. For groups, this is ignored (array names are used)
- `rechunk`: Reallocate to contiguous memory when all chunks are parsed
- `n_rows`: Stop reading after `n_rows` rows

**Returns:** A Polars LazyFrame

## Supported Data Types

| NumPy dtype | Polars dtype |
|-------------|--------------|
| int8        | Int8         |
| int16       | Int16        |
| int32       | Int32        |
| int64       | Int64        |
| uint8       | UInt8        |
| uint16      | UInt16       |
| uint32      | UInt32       |
| uint64      | UInt64       |
| float16     | Float32      |
| float32     | Float32      |
| float64     | Float64      |
| bool        | Boolean      |
| datetime64  | Datetime     |
| timedelta64 | Duration     |
| str/bytes   | String       |

## License

MIT
