//! Polars plugin for reading and writing zipped Zarr files.
//!
//! This crate provides functions to read Zarr arrays from zip files
//! and write Polars DataFrames to zipped Zarr format.

use polars::prelude::*;
use std::path::Path;
use thiserror::Error;

pub mod io;

pub use io::WriteOptions;

#[derive(Error, Debug)]
pub enum ZarrError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Zip error: {0}")]
    Zip(#[from] ::zip::result::ZipError),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Polars error: {0}")]
    Polars(#[from] PolarsError),
    #[error("Zarr error: {0}")]
    Zarr(String),
    #[error("Unsupported data type: {0}")]
    UnsupportedDataType(String),
}

pub type Result<T> = std::result::Result<T, ZarrError>;

/// Read a zipped Zarr file into a Polars DataFrame.
///
/// The Zarr file should contain either:
/// - A single 2D array (rows become DataFrame rows, columns become DataFrame columns)
/// - A group with multiple 1D arrays of the same length (each array becomes a column)
///
/// # Arguments
/// * `path` - Path to the zipped Zarr file (.zarr.zip)
/// * `column_names` - Optional custom column names
///
/// # Returns
/// A Polars DataFrame
pub fn read_zarr_zip(path: impl AsRef<Path>, column_names: Option<Vec<String>>) -> Result<DataFrame> {
    io::read_zarr_zip(path, column_names)
}

/// Write a Polars DataFrame to a zipped Zarr file.
///
/// Creates a Zarr group where each DataFrame column becomes a 1D array.
///
/// # Arguments
/// * `df` - The DataFrame to write
/// * `path` - Path to the output zipped Zarr file (.zarr.zip)
///
/// # Returns
/// Ok(()) on success
pub fn write_zarr_zip(df: &DataFrame, path: impl AsRef<Path>) -> Result<()> {
    io::write_zarr_zip(df, path)
}

/// Write a Polars DataFrame to a zipped Zarr file with options.
///
/// Creates a Zarr group where each DataFrame column becomes a 1D array.
/// Allows specifying which string columns should use dictionary encoding.
///
/// # Arguments
/// * `df` - The DataFrame to write
/// * `path` - Path to the output zipped Zarr file (.zarr.zip)
/// * `options` - Write options including dictionary encoding settings
///
/// # Returns
/// Ok(()) on success
pub fn write_zarr_zip_with_options(
    df: &DataFrame,
    path: impl AsRef<Path>,
    options: &WriteOptions,
) -> Result<()> {
    io::write_zarr_zip_with_options(df, path, options)
}

/// Scan a zipped Zarr file lazily.
///
/// Returns a LazyFrame for deferred execution.
pub fn scan_zarr_zip(path: impl AsRef<Path>, column_names: Option<Vec<String>>) -> Result<LazyFrame> {
    let df = read_zarr_zip(path, column_names)?;
    Ok(df.lazy())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Test 1: Write and read a simple 1-column DataFrame
    #[test]
    fn test_write_read_single_column() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_single.zarr.zip");

        // Create a simple DataFrame with one column
        let df = df! {
            "values" => &[1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
        .unwrap();

        // Write to zarr zip
        write_zarr_zip(&df, &path).unwrap();

        // Read back
        let read_df = read_zarr_zip(&path, None).unwrap();

        // Verify
        assert_eq!(read_df.shape(), (10, 1));
        assert_eq!(read_df.get_column_names(), vec!["values"]);

        let values: Vec<i64> = read_df
            .column("values")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(values, vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    }

    /// Test 2: Write and read a multi-column DataFrame with different types
    #[test]
    fn test_write_read_multi_column() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_multi.zarr.zip");

        // Create a DataFrame with multiple columns
        let df = df! {
            "id" => &[1i64, 2, 3, 4, 5],
            "value" => &[1.5f64, 2.5, 3.5, 4.5, 5.5],
            "count" => &[10i32, 20, 30, 40, 50]
        }
        .unwrap();

        // Write to zarr zip
        write_zarr_zip(&df, &path).unwrap();

        // Read back
        let read_df = read_zarr_zip(&path, None).unwrap();

        // Verify shape
        assert_eq!(read_df.shape(), (5, 3));

        // Verify columns exist (order may vary due to HashMap)
        let cols: Vec<String> = read_df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(cols.contains(&"id".to_string()));
        assert!(cols.contains(&"value".to_string()));
        assert!(cols.contains(&"count".to_string()));

        // Verify values
        let ids: Vec<i64> = read_df
            .column("id")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(ids, vec![1, 2, 3, 4, 5]);

        let values: Vec<f64> = read_df
            .column("value")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert!((values[0] - 1.5).abs() < 1e-10);
        assert!((values[4] - 5.5).abs() < 1e-10);
    }

    /// Test 3: Test scan_zarr_zip returns a LazyFrame that can be collected
    #[test]
    fn test_scan_zarr_lazy() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_lazy.zarr.zip");

        // Create test data
        let df = df! {
            "x" => &[1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y" => &[10i64, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        }
        .unwrap();

        write_zarr_zip(&df, &path).unwrap();

        // Scan lazily
        let lazy = scan_zarr_zip(&path, None).unwrap();

        // Apply filter and collect
        let filtered = lazy
            .filter(col("x").gt(lit(5)))
            .collect()
            .unwrap();

        // Should have 5 rows (x > 5 means x in [6,7,8,9,10])
        assert_eq!(filtered.height(), 5);
    }

    /// Test 4: Write and read string columns with variable-length encoding
    #[test]
    fn test_write_read_string_varlen() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_string_varlen.zarr.zip");

        // Create a DataFrame with string column
        let df = df! {
            "id" => &[1i64, 2, 3, 4, 5],
            "name" => &["Alice", "Bob", "Charlie", "David", "Eve"]
        }
        .unwrap();

        // Write to zarr zip (default: variable-length encoding)
        write_zarr_zip(&df, &path).unwrap();

        // Read back
        let read_df = read_zarr_zip(&path, None).unwrap();

        // Verify shape
        assert_eq!(read_df.shape(), (5, 2));

        // Verify string values
        let names: Vec<String> = read_df
            .column("name")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(names, vec!["Alice", "Bob", "Charlie", "David", "Eve"]);
    }

    /// Test 5: Write and read string columns with dictionary encoding
    #[test]
    fn test_write_read_string_dictionary() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_string_dict.zarr.zip");

        // Create a DataFrame with string column with repeated values (good for dictionary)
        let df = df! {
            "id" => &[1i64, 2, 3, 4, 5, 6, 7, 8],
            "category" => &["A", "B", "A", "C", "B", "A", "C", "B"]
        }
        .unwrap();

        // Write with dictionary encoding for the category column
        let options = WriteOptions {
            dictionary_columns: vec!["category".to_string()],
        };
        write_zarr_zip_with_options(&df, &path, &options).unwrap();

        // Read back
        let read_df = read_zarr_zip(&path, None).unwrap();

        // Verify shape
        assert_eq!(read_df.shape(), (8, 2));

        // Verify string values
        let categories: Vec<String> = read_df
            .column("category")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(categories, vec!["A", "B", "A", "C", "B", "A", "C", "B"]);
    }

    /// Test 6: Write and read mixed numeric and string columns
    #[test]
    fn test_write_read_mixed_types() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_mixed.zarr.zip");

        // Create a DataFrame with mixed types
        let df = df! {
            "int_col" => &[1i64, 2, 3],
            "float_col" => &[1.5f64, 2.5, 3.5],
            "string_col" => &["hello", "world", "test"]
        }
        .unwrap();

        write_zarr_zip(&df, &path).unwrap();

        // Read back
        let read_df = read_zarr_zip(&path, None).unwrap();

        // Verify shape
        assert_eq!(read_df.shape(), (3, 3));

        // Verify int column
        let ints: Vec<i64> = read_df
            .column("int_col")
            .unwrap()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        assert_eq!(ints, vec![1, 2, 3]);

        // Verify string column
        let strings: Vec<String> = read_df
            .column("string_col")
            .unwrap()
            .str()
            .unwrap()
            .into_no_null_iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(strings, vec!["hello", "world", "test"]);
    }
}
