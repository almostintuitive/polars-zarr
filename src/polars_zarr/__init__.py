"""Polars IO plugin for reading Zarr v3 arrays as DataFrames."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import zarr

from polars.io.plugins import register_io_source

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from polars._typing import PolarsDataType, SchemaDict


# Mapping from numpy dtypes to polars dtypes
NUMPY_TO_POLARS_DTYPE: dict[np.dtype, PolarsDataType] = {
    np.dtype("int8"): pl.Int8,
    np.dtype("int16"): pl.Int16,
    np.dtype("int32"): pl.Int32,
    np.dtype("int64"): pl.Int64,
    np.dtype("uint8"): pl.UInt8,
    np.dtype("uint16"): pl.UInt16,
    np.dtype("uint32"): pl.UInt32,
    np.dtype("uint64"): pl.UInt64,
    np.dtype("float16"): pl.Float32,
    np.dtype("float32"): pl.Float32,
    np.dtype("float64"): pl.Float64,
    np.dtype("bool"): pl.Boolean,
}


def _numpy_dtype_to_polars(dtype: np.dtype) -> PolarsDataType:
    """Convert numpy dtype to polars dtype."""
    if dtype in NUMPY_TO_POLARS_DTYPE:
        return NUMPY_TO_POLARS_DTYPE[dtype]

    # Handle string dtypes
    if np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.bytes_):
        return pl.String

    # Handle complex types - store as string representation
    if np.issubdtype(dtype, np.complexfloating):
        return pl.String

    # Handle datetime types
    if np.issubdtype(dtype, np.datetime64):
        return pl.Datetime

    # Handle timedelta types
    if np.issubdtype(dtype, np.timedelta64):
        return pl.Duration

    # Default to string for unknown types
    return pl.String


def _is_zarr_group(store_path: str | Path) -> bool:
    """Check if the zarr store is a group with multiple arrays."""
    store_path = Path(store_path)
    zarr_json = store_path / "zarr.json"

    if zarr_json.exists():
        import json

        with open(zarr_json) as f:
            metadata = json.load(f)
            return metadata.get("node_type") == "group"

    return False


def _get_group_arrays(store_path: str | Path) -> dict[str, zarr.Array]:
    """Get all arrays from a zarr group."""
    group = zarr.open_group(str(store_path), mode="r")
    arrays = {}

    for name, member in group.members():
        if isinstance(member, zarr.Array):
            arrays[name] = member

    return arrays


def scan_zarr(
    source: str | Path,
    *,
    column_names: Sequence[str] | None = None,
    rechunk: bool = False,
    n_rows: int | None = None,
) -> pl.LazyFrame:
    """
    Lazily read from a Zarr v3 array or group.

    Lazy reading allows the query optimizer to push down predicates
    and projections to the scan level, thereby potentially reducing
    memory overhead.

    For single arrays:
    - 1D arrays become a single-column DataFrame
    - 2D arrays become a DataFrame where each row corresponds to the first
      dimension

    For groups with multiple 1D arrays of the same length:
    - Each array becomes a column in the DataFrame

    Parameters
    ----------
    source
        Path to a Zarr array or group.
    column_names
        Optional list of column names. For 2D arrays, specifies names for
        each column. For groups, this is ignored (array names are used).
    rechunk
        Reallocate to contiguous memory when all chunks are parsed.
    n_rows
        Stop reading after `n_rows` rows.

    Returns
    -------
    LazyFrame

    Examples
    --------
    >>> import numpy as np
    >>> import polars_zarr
    >>> import zarr
    >>>
    >>> # Create a test zarr array
    >>> z = zarr.create_array(
    ...     store="my_data.zarr",
    ...     shape=(1000, 5),
    ...     chunks=(100, 5),
    ...     dtype="float64",
    ... )
    >>> z[:] = np.random.randn(1000, 5)
    >>>
    >>> # Read lazily with polars_zarr
    >>> lf = polars_zarr.scan_zarr("my_data.zarr")
    >>> df = lf.head(10).collect()  # doctest: +SKIP

    """
    source_path = Path(source) if isinstance(source, str) else source

    def get_schema() -> SchemaDict:
        """Get the schema from the zarr array or group."""
        if _is_zarr_group(source_path):
            # Handle group with multiple arrays
            arrays = _get_group_arrays(source_path)

            if not arrays:
                msg = f"No arrays found in zarr group: {source_path}"
                raise ValueError(msg)

            schema: SchemaDict = {}
            for name, arr in arrays.items():
                if arr.ndim != 1:
                    msg = f"Array '{name}' in group must be 1D, got {arr.ndim}D"
                    raise ValueError(msg)
                schema[name] = _numpy_dtype_to_polars(arr.dtype)

            return schema

        # Handle single array
        arr = zarr.open_array(str(source_path), mode="r")

        if arr.ndim == 1:
            # 1D array: single column
            col_name = column_names[0] if column_names else "0"
            return {col_name: _numpy_dtype_to_polars(arr.dtype)}

        if arr.ndim == 2:
            # 2D array: columns are the second dimension
            n_cols = arr.shape[1]
            if column_names:
                if len(column_names) != n_cols:
                    msg = (
                        f"Number of column names ({len(column_names)}) does not "
                        f"match number of columns ({n_cols})"
                    )
                    raise ValueError(msg)
                names = list(column_names)
            else:
                names = [str(i) for i in range(n_cols)]

            return {name: _numpy_dtype_to_polars(arr.dtype) for name in names}

        msg = f"Only 1D and 2D arrays are supported, got {arr.ndim}D"
        raise ValueError(msg)

    def zarr_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows_limit: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        """Generate DataFrames from zarr chunks."""
        n_rows_remaining = n_rows_limit if n_rows_limit is not None else n_rows

        if _is_zarr_group(source_path):
            # Handle group with multiple arrays
            arrays = _get_group_arrays(source_path)

            if not arrays:
                return

            # Determine which columns to read
            cols_to_read = with_columns if with_columns else list(arrays.keys())

            # Get the length from the first array
            first_arr = next(iter(arrays.values()))
            total_rows = first_arr.shape[0]
            chunk_size = first_arr.chunks[0]

            # Verify all arrays have the same length
            for name, arr in arrays.items():
                if arr.shape[0] != total_rows:
                    msg = (
                        f"Array '{name}' has length {arr.shape[0]}, "
                        f"expected {total_rows}"
                    )
                    raise ValueError(msg)

            # Read in chunks
            offset = 0
            while offset < total_rows:
                if n_rows_remaining is not None and n_rows_remaining <= 0:
                    break

                # Determine chunk bounds
                end = min(offset + chunk_size, total_rows)
                if n_rows_remaining is not None:
                    end = min(end, offset + n_rows_remaining)

                # Read data for each column
                data_dict = {}
                for name in cols_to_read:
                    if name in arrays:
                        data_dict[name] = arrays[name][offset:end]

                df = pl.DataFrame(data_dict)

                # Apply predicate filter
                if predicate is not None:
                    df = df.filter(predicate)

                if df.height > 0:
                    yield df

                    if n_rows_remaining is not None:
                        n_rows_remaining -= df.height

                offset = end

        else:
            # Handle single array
            arr = zarr.open_array(str(source_path), mode="r")

            if arr.ndim == 1:
                col_name = column_names[0] if column_names else "0"
                cols_to_read = with_columns if with_columns else [col_name]

                total_rows = arr.shape[0]
                chunk_size = arr.chunks[0]

                offset = 0
                while offset < total_rows:
                    if n_rows_remaining is not None and n_rows_remaining <= 0:
                        break

                    end = min(offset + chunk_size, total_rows)
                    if n_rows_remaining is not None:
                        end = min(end, offset + n_rows_remaining)

                    chunk_data = arr[offset:end]

                    if col_name in cols_to_read:
                        df = pl.DataFrame({col_name: chunk_data})
                    else:
                        # Column not requested, skip
                        offset = end
                        continue

                    if predicate is not None:
                        df = df.filter(predicate)

                    if df.height > 0:
                        yield df

                        if n_rows_remaining is not None:
                            n_rows_remaining -= df.height

                    offset = end

            elif arr.ndim == 2:
                n_cols = arr.shape[1]
                if column_names:
                    names = list(column_names)
                else:
                    names = [str(i) for i in range(n_cols)]

                cols_to_read = with_columns if with_columns else names

                total_rows = arr.shape[0]
                # Use the chunk size along the first dimension (rows)
                chunk_size = arr.chunks[0]

                offset = 0
                while offset < total_rows:
                    if n_rows_remaining is not None and n_rows_remaining <= 0:
                        break

                    end = min(offset + chunk_size, total_rows)
                    if n_rows_remaining is not None:
                        end = min(end, offset + n_rows_remaining)

                    # Read the chunk
                    chunk_data = arr[offset:end, :]

                    # Build dataframe with requested columns
                    data_dict = {}
                    for i, name in enumerate(names):
                        if name in cols_to_read:
                            data_dict[name] = chunk_data[:, i]

                    if data_dict:
                        df = pl.DataFrame(data_dict)

                        if predicate is not None:
                            df = df.filter(predicate)

                        if df.height > 0:
                            yield df

                            if n_rows_remaining is not None:
                                n_rows_remaining -= df.height

                    offset = end

            else:
                msg = f"Only 1D and 2D arrays are supported, got {arr.ndim}D"
                raise ValueError(msg)

    # Get schema from zarr
    schema = get_schema()

    return register_io_source(io_source=zarr_generator, schema=schema)
