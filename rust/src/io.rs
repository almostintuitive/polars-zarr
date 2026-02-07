//! IO operations for reading and writing zipped Zarr files.

use crate::{Result, ZarrError};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

// Use fully qualified path to avoid conflict with polars::prelude::zip
use ::zip::read::ZipArchive;
use ::zip::write::{FileOptions, ZipWriter};
use ::zip::CompressionMethod;

/// Zarr v3 array metadata
#[derive(Debug, Serialize, Deserialize)]
struct ZarrArrayMetadata {
    zarr_format: u8,
    node_type: String,
    shape: Vec<usize>,
    data_type: String,
    chunk_grid: ChunkGrid,
    chunk_key_encoding: ChunkKeyEncoding,
    fill_value: serde_json::Value,
    codecs: Vec<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChunkGrid {
    name: String,
    configuration: ChunkGridConfig,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChunkGridConfig {
    chunk_shape: Vec<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ChunkKeyEncoding {
    name: String,
}

/// Zarr v3 group metadata
#[derive(Debug, Serialize, Deserialize)]
struct ZarrGroupMetadata {
    zarr_format: u8,
    node_type: String,
}

fn polars_dtype_to_zarr(dtype: &DataType) -> Result<&'static str> {
    match dtype {
        DataType::Int8 => Ok("int8"),
        DataType::Int16 => Ok("int16"),
        DataType::Int32 => Ok("int32"),
        DataType::Int64 => Ok("int64"),
        DataType::UInt8 => Ok("uint8"),
        DataType::UInt16 => Ok("uint16"),
        DataType::UInt32 => Ok("uint32"),
        DataType::UInt64 => Ok("uint64"),
        DataType::Float32 => Ok("float32"),
        DataType::Float64 => Ok("float64"),
        DataType::Boolean => Ok("bool"),
        _ => Err(ZarrError::UnsupportedDataType(format!("{:?}", dtype))),
    }
}

fn zarr_dtype_to_polars(dtype: &str) -> Result<DataType> {
    match dtype {
        "int8" | "<i1" | "|i1" => Ok(DataType::Int8),
        "int16" | "<i2" => Ok(DataType::Int16),
        "int32" | "<i4" => Ok(DataType::Int32),
        "int64" | "<i8" => Ok(DataType::Int64),
        "uint8" | "<u1" | "|u1" => Ok(DataType::UInt8),
        "uint16" | "<u2" => Ok(DataType::UInt16),
        "uint32" | "<u4" => Ok(DataType::UInt32),
        "uint64" | "<u8" => Ok(DataType::UInt64),
        "float32" | "<f4" => Ok(DataType::Float32),
        "float64" | "<f8" => Ok(DataType::Float64),
        "bool" | "|b1" => Ok(DataType::Boolean),
        _ => Err(ZarrError::UnsupportedDataType(dtype.to_string())),
    }
}

fn dtype_byte_size(dtype: &DataType) -> usize {
    match dtype {
        DataType::Int8 | DataType::UInt8 | DataType::Boolean => 1,
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int32 | DataType::UInt32 | DataType::Float32 => 4,
        DataType::Int64 | DataType::UInt64 | DataType::Float64 => 8,
        _ => 8, // Default
    }
}

/// Read a zipped Zarr file into a Polars DataFrame.
pub fn read_zarr_zip(path: impl AsRef<Path>, column_names: Option<Vec<String>>) -> Result<DataFrame> {
    let file = File::open(path.as_ref())?;
    let mut archive = ZipArchive::new(file)?;

    // Find all zarr.json files to identify arrays
    let mut array_paths: Vec<String> = Vec::new();

    for i in 0..archive.len() {
        let file = archive.by_index(i)?;
        let name = file.name().to_string();
        if name.ends_with("/zarr.json") || name == "zarr.json" {
            array_paths.push(name);
        }
    }

    // Check if root is a group or an array
    let root_metadata: serde_json::Value = {
        let mut root_file = archive.by_name("zarr.json")?;
        let mut contents = String::new();
        root_file.read_to_string(&mut contents)?;
        serde_json::from_str(&contents)?
    };

    let node_type = root_metadata
        .get("node_type")
        .and_then(|v| v.as_str())
        .unwrap_or("array");

    if node_type == "group" {
        // Read group with multiple arrays
        read_zarr_group(&mut archive, column_names)
    } else {
        // Read single array
        read_zarr_array(&mut archive, "", column_names)
    }
}

fn read_zarr_group<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    _column_names: Option<Vec<String>>,
) -> Result<DataFrame> {
    // Find all array directories
    let mut arrays: HashMap<String, (ZarrArrayMetadata, Vec<u8>)> = HashMap::new();

    // First pass: find all zarr.json files
    let mut array_names: Vec<String> = Vec::new();
    for i in 0..archive.len() {
        let file = archive.by_index(i)?;
        let name = file.name().to_string();
        // Look for paths like "array_name/zarr.json"
        if name.ends_with("/zarr.json") && name.matches('/').count() == 1 {
            let array_name = name.trim_end_matches("/zarr.json").to_string();
            array_names.push(array_name);
        }
    }

    // Read each array's metadata and data
    for array_name in &array_names {
        let metadata_path = format!("{}/zarr.json", array_name);
        let metadata: ZarrArrayMetadata = {
            let mut file = archive.by_name(&metadata_path)?;
            let mut contents = String::new();
            file.read_to_string(&mut contents)?;
            serde_json::from_str(&contents)?
        };

        // Read chunk data (assuming single chunk c/0)
        let chunk_path = format!("{}/c/0", array_name);
        let chunk_path_alt = format!("{}/c0", array_name);

        // Try primary path first
        let primary_result: std::result::Result<Vec<u8>, _> = {
            match archive.by_name(&chunk_path) {
                Ok(mut file) => {
                    let mut data = Vec::new();
                    file.read_to_end(&mut data)?;
                    Ok(data)
                }
                Err(e) => Err(e),
            }
        };

        let chunk_data = match primary_result {
            Ok(data) => data,
            Err(_) => {
                // Try alternate path c0
                let mut file = archive.by_name(&chunk_path_alt)?;
                let mut data = Vec::new();
                file.read_to_end(&mut data)?;
                data
            }
        };

        arrays.insert(array_name.clone(), (metadata, chunk_data));
    }

    // Convert to DataFrame
    let mut columns: Vec<Column> = Vec::new();

    for (name, (metadata, data)) in arrays {
        let dtype = zarr_dtype_to_polars(&metadata.data_type)?;
        let len = metadata.shape[0];

        let series = bytes_to_series(&name, &data, &dtype, len)?;
        columns.push(series.into());
    }

    DataFrame::new(columns).map_err(ZarrError::from)
}

fn read_zarr_array<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    prefix: &str,
    column_names: Option<Vec<String>>,
) -> Result<DataFrame> {
    let metadata_path = if prefix.is_empty() {
        "zarr.json".to_string()
    } else {
        format!("{}/zarr.json", prefix)
    };

    let metadata: ZarrArrayMetadata = {
        let mut file = archive.by_name(&metadata_path)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        serde_json::from_str(&contents)?
    };

    let dtype = zarr_dtype_to_polars(&metadata.data_type)?;

    if metadata.shape.len() == 1 {
        // 1D array
        let chunk_path = if prefix.is_empty() {
            "c/0".to_string()
        } else {
            format!("{}/c/0", prefix)
        };

        let chunk_data = {
            let mut file = archive.by_name(&chunk_path)?;
            let mut data = Vec::new();
            file.read_to_end(&mut data)?;
            data
        };

        let col_name = column_names
            .and_then(|names| names.into_iter().next())
            .unwrap_or_else(|| "0".to_string());

        let series = bytes_to_series(&col_name, &chunk_data, &dtype, metadata.shape[0])?;
        DataFrame::new(vec![series.into()]).map_err(ZarrError::from)
    } else if metadata.shape.len() == 2 {
        // 2D array
        let n_rows = metadata.shape[0];
        let n_cols = metadata.shape[1];

        // Read all chunks
        let chunk_path = if prefix.is_empty() {
            "c/0/0".to_string()
        } else {
            format!("{}/c/0/0", prefix)
        };

        let chunk_data = {
            let mut file = archive.by_name(&chunk_path)?;
            let mut data = Vec::new();
            file.read_to_end(&mut data)?;
            data
        };

        let names: Vec<String> = column_names.unwrap_or_else(|| {
            (0..n_cols).map(|i| i.to_string()).collect()
        });

        let byte_size = dtype_byte_size(&dtype);
        let mut columns: Vec<Column> = Vec::new();

        for col_idx in 0..n_cols {
            let col_data: Vec<u8> = (0..n_rows)
                .flat_map(|row_idx| {
                    let offset = (row_idx * n_cols + col_idx) * byte_size;
                    chunk_data[offset..offset + byte_size].to_vec()
                })
                .collect();

            let series = bytes_to_series(&names[col_idx], &col_data, &dtype, n_rows)?;
            columns.push(series.into());
        }

        DataFrame::new(columns).map_err(ZarrError::from)
    } else {
        Err(ZarrError::Zarr(format!(
            "Only 1D and 2D arrays are supported, got {}D",
            metadata.shape.len()
        )))
    }
}

fn bytes_to_series(name: &str, data: &[u8], dtype: &DataType, _len: usize) -> Result<Series> {
    let series = match dtype {
        DataType::Int8 => {
            let values: Vec<i8> = data.iter().map(|&b| b as i8).collect();
            Series::new(name.into(), &values)
        }
        DataType::Int16 => {
            let values: Vec<i16> = data
                .chunks(2)
                .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
                .collect();
            Series::new(name.into(), &values)
        }
        DataType::Int32 => {
            let values: Vec<i32> = data
                .chunks(4)
                .map(|chunk| i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Series::new(name.into(), &values)
        }
        DataType::Int64 => {
            let values: Vec<i64> = data
                .chunks(8)
                .map(|chunk| {
                    i64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect();
            Series::new(name.into(), &values)
        }
        DataType::UInt8 => {
            // Store as UInt32 to avoid NamedFrom limitations
            let values: Vec<u32> = data.iter().map(|&v| v as u32).collect();
            Series::new(name.into(), &values).cast(&DataType::UInt8)?
        }
        DataType::UInt16 => {
            // Store as UInt32 to avoid NamedFrom limitations
            let values: Vec<u32> = data
                .chunks(2)
                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]) as u32)
                .collect();
            Series::new(name.into(), &values).cast(&DataType::UInt16)?
        }
        DataType::UInt32 => {
            let values: Vec<u32> = data
                .chunks(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Series::new(name.into(), &values)
        }
        DataType::UInt64 => {
            let values: Vec<u64> = data
                .chunks(8)
                .map(|chunk| {
                    u64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect();
            Series::new(name.into(), &values)
        }
        DataType::Float32 => {
            let values: Vec<f32> = data
                .chunks(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            Series::new(name.into(), &values)
        }
        DataType::Float64 => {
            let values: Vec<f64> = data
                .chunks(8)
                .map(|chunk| {
                    f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ])
                })
                .collect();
            Series::new(name.into(), &values)
        }
        DataType::Boolean => {
            let values: Vec<bool> = data.iter().map(|&b| b != 0).collect();
            Series::new(name.into(), &values)
        }
        _ => return Err(ZarrError::UnsupportedDataType(format!("{:?}", dtype))),
    };

    Ok(series)
}

fn series_to_bytes(series: &Series) -> Result<Vec<u8>> {
    let dtype = series.dtype();

    let bytes = match dtype {
        DataType::Int8 => series
            .i8()?
            .into_no_null_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DataType::Int16 => series
            .i16()?
            .into_no_null_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DataType::Int32 => series
            .i32()?
            .into_no_null_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DataType::Int64 => series
            .i64()?
            .into_no_null_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DataType::UInt8 => series.u8()?.into_no_null_iter().collect(),
        DataType::UInt16 => series
            .u16()?
            .into_no_null_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DataType::UInt32 => series
            .u32()?
            .into_no_null_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DataType::UInt64 => series
            .u64()?
            .into_no_null_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DataType::Float32 => series
            .f32()?
            .into_no_null_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DataType::Float64 => series
            .f64()?
            .into_no_null_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        DataType::Boolean => series
            .bool()?
            .into_no_null_iter()
            .map(|v| if v { 1u8 } else { 0u8 })
            .collect(),
        _ => return Err(ZarrError::UnsupportedDataType(format!("{:?}", dtype))),
    };

    Ok(bytes)
}

/// Write a Polars DataFrame to a zipped Zarr file.
pub fn write_zarr_zip(df: &DataFrame, path: impl AsRef<Path>) -> Result<()> {
    let file = File::create(path.as_ref())?;
    let mut zip = ZipWriter::new(file);

    let options: FileOptions<()> = FileOptions::default()
        .compression_method(CompressionMethod::Stored);

    // Write root group metadata
    let group_metadata = ZarrGroupMetadata {
        zarr_format: 3,
        node_type: "group".to_string(),
    };

    zip.start_file("zarr.json", options.clone())?;
    zip.write_all(serde_json::to_string_pretty(&group_metadata)?.as_bytes())?;

    // Write each column as a 1D array
    for col in df.get_columns() {
        let series = col.as_materialized_series();
        let name = series.name().to_string();
        let len = series.len();
        let dtype = series.dtype();
        let zarr_dtype = polars_dtype_to_zarr(dtype)?;

        // Create array metadata
        let array_metadata = ZarrArrayMetadata {
            zarr_format: 3,
            node_type: "array".to_string(),
            shape: vec![len],
            data_type: zarr_dtype.to_string(),
            chunk_grid: ChunkGrid {
                name: "regular".to_string(),
                configuration: ChunkGridConfig {
                    chunk_shape: vec![len],
                },
            },
            chunk_key_encoding: ChunkKeyEncoding {
                name: "default".to_string(),
            },
            fill_value: serde_json::Value::Number(0.into()),
            codecs: vec![serde_json::json!({
                "name": "bytes",
                "configuration": {
                    "endian": "little"
                }
            })],
        };

        // Write array metadata
        zip.start_file(format!("{}/zarr.json", name), options.clone())?;
        zip.write_all(serde_json::to_string_pretty(&array_metadata)?.as_bytes())?;

        // Write chunk data
        let chunk_data = series_to_bytes(series)?;
        zip.start_file(format!("{}/c/0", name), options.clone())?;
        zip.write_all(&chunk_data)?;
    }

    zip.finish()?;
    Ok(())
}
