"""Tests for polars-zarr plugin."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import zarr

import polars_zarr


class TestScanZarr1DArray:
    """Test reading a simple 1D zarr array."""

    def test_read_1d_int_array(self, tmp_path: Path) -> None:
        """Test reading a 1D integer array from zarr."""
        # Create test data
        data = np.arange(100, dtype=np.int64)

        # Save to zarr
        zarr_path = tmp_path / "test_1d.zarr"
        z = zarr.create_array(
            store=str(zarr_path),
            shape=data.shape,
            chunks=(10,),
            dtype=data.dtype,
        )
        z[:] = data

        # Read with polars_zarr
        df = polars_zarr.scan_zarr(str(zarr_path)).collect()

        # Verify
        assert df.shape == (100, 1)
        assert df.columns == ["0"]
        assert df["0"].to_list() == list(range(100))

    def test_read_1d_float_array(self, tmp_path: Path) -> None:
        """Test reading a 1D float array from zarr."""
        # Create test data
        data = np.linspace(0.0, 10.0, 50, dtype=np.float64)

        # Save to zarr
        zarr_path = tmp_path / "test_1d_float.zarr"
        z = zarr.create_array(
            store=str(zarr_path),
            shape=data.shape,
            chunks=(10,),
            dtype=data.dtype,
        )
        z[:] = data

        # Read with polars_zarr
        df = polars_zarr.scan_zarr(str(zarr_path)).collect()

        # Verify
        assert df.shape == (50, 1)
        np.testing.assert_array_almost_equal(
            df["0"].to_numpy(), data, decimal=10
        )


class TestScanZarr2DArray:
    """Test reading 2D zarr arrays."""

    def test_read_2d_array(self, tmp_path: Path) -> None:
        """Test reading a 2D array from zarr - rows become records."""
        # Create test data: 20 rows, 5 columns
        data = np.arange(100, dtype=np.int32).reshape(20, 5)

        # Save to zarr
        zarr_path = tmp_path / "test_2d.zarr"
        z = zarr.create_array(
            store=str(zarr_path),
            shape=data.shape,
            chunks=(5, 5),
            dtype=data.dtype,
        )
        z[:] = data

        # Read with polars_zarr
        df = polars_zarr.scan_zarr(str(zarr_path)).collect()

        # Verify: 20 rows, 5 columns
        assert df.shape == (20, 5)
        assert df.columns == ["0", "1", "2", "3", "4"]

        # Check first row
        assert df.row(0) == (0, 1, 2, 3, 4)
        # Check last row
        assert df.row(19) == (95, 96, 97, 98, 99)

    def test_read_2d_array_with_column_names(self, tmp_path: Path) -> None:
        """Test reading a 2D array with custom column names."""
        # Create test data: 10 rows, 3 columns
        data = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0],
            [16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0],
            [22.0, 23.0, 24.0],
            [25.0, 26.0, 27.0],
            [28.0, 29.0, 30.0],
        ], dtype=np.float64)

        # Save to zarr
        zarr_path = tmp_path / "test_2d_named.zarr"
        z = zarr.create_array(
            store=str(zarr_path),
            shape=data.shape,
            chunks=(5, 3),
            dtype=data.dtype,
        )
        z[:] = data

        # Read with polars_zarr with custom column names
        df = polars_zarr.scan_zarr(
            str(zarr_path),
            column_names=["x", "y", "z"]
        ).collect()

        # Verify
        assert df.shape == (10, 3)
        assert df.columns == ["x", "y", "z"]
        assert df["x"].to_list() == [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0]


class TestScanZarrGroup:
    """Test reading zarr groups with multiple arrays."""

    def test_read_group_arrays(self, tmp_path: Path) -> None:
        """Test reading multiple arrays from a zarr group as columns."""
        zarr_path = tmp_path / "test_group.zarr"

        # Create a zarr group with multiple 1D arrays
        root = zarr.open_group(str(zarr_path), mode="w")

        # Create arrays with same length
        ids = np.arange(100, dtype=np.int64)
        values = np.random.randn(100).astype(np.float64)
        flags = np.random.randint(0, 2, size=100, dtype=np.int8)

        root.create_array("id", data=ids, chunks=(20,))
        root.create_array("value", data=values, chunks=(20,))
        root.create_array("flag", data=flags, chunks=(20,))

        # Read with polars_zarr - should combine arrays into columns
        df = polars_zarr.scan_zarr(str(zarr_path)).collect()

        # Verify: 100 rows, 3 columns (id, value, flag)
        assert df.shape == (100, 3)
        assert set(df.columns) == {"id", "value", "flag"}

        # Check values
        np.testing.assert_array_equal(df["id"].to_numpy(), ids)
        np.testing.assert_array_almost_equal(df["value"].to_numpy(), values)
        np.testing.assert_array_equal(df["flag"].to_numpy(), flags)
