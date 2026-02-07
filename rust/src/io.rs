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

/// Options for writing string columns
#[derive(Debug, Clone, Default)]
pub struct WriteOptions {
    /// Columns to encode using dictionary encoding
    pub dictionary_columns: Vec<String>,
}

/// Zarr v3 array metadata
#[derive(Debug, Serialize, Deserialize)]
struct ZarrArrayMetadata {
    zarr_format: u8,
    node_type: String,
    shape: Vec<usize>,
    data_type: serde_json::Value,
    chunk_grid: ChunkGrid,
    chunk_key_encoding: ChunkKeyEncoding,
    fill_value: serde_json::Value,
    codecs: Vec<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attributes: Option<serde_json::Value>,
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

/// String encoding format indicator
#[derive(Debug, Clone, PartialEq)]
enum StringEncoding {
    /// Variable-length strings stored as length-prefixed UTF-8
    VarLen,
    /// Dictionary-encoded strings
    Dictionary,
}

fn polars_dtype_to_zarr(dtype: &DataType, use_dictionary: bool) -> Result<serde_json::Value> {
    match dtype {
        DataType::Int8 => Ok(serde_json::json!("int8")),
        DataType::Int16 => Ok(serde_json::json!("int16")),
        DataType::Int32 => Ok(serde_json::json!("int32")),
        DataType::Int64 => Ok(serde_json::json!("int64")),
        DataType::UInt8 => Ok(serde_json::json!("uint8")),
        DataType::UInt16 => Ok(serde_json::json!("uint16")),
        DataType::UInt32 => Ok(serde_json::json!("uint32")),
        DataType::UInt64 => Ok(serde_json::json!("uint64")),
        DataType::Float32 => Ok(serde_json::json!("float32")),
        DataType::Float64 => Ok(serde_json::json!("float64")),
        DataType::Boolean => Ok(serde_json::json!("bool")),
        DataType::String => {
            if use_dictionary {
                // Dictionary encoding: store as uint32 indices
                Ok(serde_json::json!({
                    "name": "polars.dictionary",
                    "configuration": {
                        "index_type": "uint32"
                    }
                }))
            } else {
                // Variable-length string encoding
                Ok(serde_json::json!({
                    "name": "polars.vlen_utf8"
                }))
            }
        }
        DataType::Categorical(_, _) => {
            // Categorical is stored as dictionary encoding
            Ok(serde_json::json!({
                "name": "polars.dictionary",
                "configuration": {
                    "index_type": "uint32"
                }
            }))
        }
        _ => Err(ZarrError::UnsupportedDataType(format!("{:?}", dtype))),
    }
}

fn zarr_dtype_to_polars(dtype: &serde_json::Value) -> Result<(DataType, StringEncoding)> {
    // Handle string dtype
    if let Some(s) = dtype.as_str() {
        let dt = match s {
            "int8" | "<i1" | "|i1" => DataType::Int8,
            "int16" | "<i2" => DataType::Int16,
            "int32" | "<i4" => DataType::Int32,
            "int64" | "<i8" => DataType::Int64,
            "uint8" | "<u1" | "|u1" => DataType::UInt8,
            "uint16" | "<u2" => DataType::UInt16,
            "uint32" | "<u4" => DataType::UInt32,
            "uint64" | "<u8" => DataType::UInt64,
            "float32" | "<f4" => DataType::Float32,
            "float64" | "<f8" => DataType::Float64,
            "bool" | "|b1" => DataType::Boolean,
            _ => return Err(ZarrError::UnsupportedDataType(s.to_string())),
        };
        return Ok((dt, StringEncoding::VarLen));
    }

    // Handle object dtype (for extension types)
    if let Some(obj) = dtype.as_object() {
        if let Some(name) = obj.get("name").and_then(|v| v.as_str()) {
            match name {
                "polars.vlen_utf8" => return Ok((DataType::String, StringEncoding::VarLen)),
                "polars.dictionary" => return Ok((DataType::String, StringEncoding::Dictionary)),
                _ => return Err(ZarrError::UnsupportedDataType(name.to_string())),
            }
        }
    }

    Err(ZarrError::UnsupportedDataType(format!("{:?}", dtype)))
}

fn dtype_byte_size(dtype: &DataType) -> Option<usize> {
    match dtype {
        DataType::Int8 | DataType::UInt8 | DataType::Boolean => Some(1),
        DataType::Int16 | DataType::UInt16 => Some(2),
        DataType::Int32 | DataType::UInt32 | DataType::Float32 => Some(4),
        DataType::Int64 | DataType::UInt64 | DataType::Float64 => Some(8),
        DataType::String => None, // Variable length
        _ => None,
    }
}

/// Read a zipped Zarr file into a Polars DataFrame.
pub fn read_zarr_zip(
    path: impl AsRef<Path>,
    column_names: Option<Vec<String>>,
) -> Result<DataFrame> {
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
    let mut arrays: HashMap<String, (ZarrArrayMetadata, Vec<u8>, Option<Vec<String>>)> =
        HashMap::new();

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

        // Check for dictionary data
        let dictionary = {
            let dict_path = format!("{}/dictionary.json", array_name);
            match archive.by_name(&dict_path) {
                Ok(mut file) => {
                    let mut contents = String::new();
                    file.read_to_string(&mut contents)?;
                    let dict: Vec<String> = serde_json::from_str(&contents)?;
                    Some(dict)
                }
                Err(_) => None,
            }
        };

        arrays.insert(array_name.clone(), (metadata, chunk_data, dictionary));
    }

    // Convert to DataFrame
    let mut columns: Vec<Column> = Vec::new();

    for (name, (metadata, data, dictionary)) in arrays {
        let (dtype, encoding) = zarr_dtype_to_polars(&metadata.data_type)?;
        let len = metadata.shape[0];

        let series = bytes_to_series(&name, &data, &dtype, len, encoding, dictionary.as_ref())?;
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

    let (dtype, encoding) = zarr_dtype_to_polars(&metadata.data_type)?;

    // Check for dictionary
    let dictionary = {
        let dict_path = if prefix.is_empty() {
            "dictionary.json".to_string()
        } else {
            format!("{}/dictionary.json", prefix)
        };
        match archive.by_name(&dict_path) {
            Ok(mut file) => {
                let mut contents = String::new();
                file.read_to_string(&mut contents)?;
                let dict: Vec<String> = serde_json::from_str(&contents)?;
                Some(dict)
            }
            Err(_) => None,
        }
    };

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

        let series = bytes_to_series(
            &col_name,
            &chunk_data,
            &dtype,
            metadata.shape[0],
            encoding,
            dictionary.as_ref(),
        )?;
        DataFrame::new(vec![series.into()]).map_err(ZarrError::from)
    } else if metadata.shape.len() == 2 {
        // 2D array - only for numeric types
        if dtype == DataType::String {
            return Err(ZarrError::Zarr(
                "2D string arrays are not supported".to_string(),
            ));
        }

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

        let byte_size = dtype_byte_size(&dtype).ok_or_else(|| {
            ZarrError::Zarr("Cannot determine byte size for dtype".to_string())
        })?;
        let mut columns: Vec<Column> = Vec::new();

        for col_idx in 0..n_cols {
            let col_data: Vec<u8> = (0..n_rows)
                .flat_map(|row_idx| {
                    let offset = (row_idx * n_cols + col_idx) * byte_size;
                    chunk_data[offset..offset + byte_size].to_vec()
                })
                .collect();

            let series = bytes_to_series(
                &names[col_idx],
                &col_data,
                &dtype,
                n_rows,
                StringEncoding::VarLen,
                None,
            )?;
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

fn bytes_to_series(
    name: &str,
    data: &[u8],
    dtype: &DataType,
    len: usize,
    encoding: StringEncoding,
    dictionary: Option<&Vec<String>>,
) -> Result<Series> {
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
        DataType::String => {
            match encoding {
                StringEncoding::VarLen => {
                    // Variable-length strings: each string is 4-byte length + UTF-8 bytes
                    let mut strings: Vec<String> = Vec::with_capacity(len);
                    let mut offset = 0;

                    while offset < data.len() && strings.len() < len {
                        if offset + 4 > data.len() {
                            break;
                        }
                        let str_len = u32::from_le_bytes([
                            data[offset],
                            data[offset + 1],
                            data[offset + 2],
                            data[offset + 3],
                        ]) as usize;
                        offset += 4;

                        if offset + str_len > data.len() {
                            break;
                        }
                        let s = String::from_utf8_lossy(&data[offset..offset + str_len]).to_string();
                        strings.push(s);
                        offset += str_len;
                    }

                    Series::new(name.into(), &strings)
                }
                StringEncoding::Dictionary => {
                    // Dictionary-encoded: indices stored as u32, look up in dictionary
                    let dict = dictionary.ok_or_else(|| {
                        ZarrError::Zarr("Dictionary data missing for dictionary-encoded column".to_string())
                    })?;

                    let indices: Vec<u32> = data
                        .chunks(4)
                        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();

                    let strings: Vec<String> = indices
                        .iter()
                        .map(|&idx| {
                            dict.get(idx as usize)
                                .cloned()
                                .unwrap_or_else(|| "".to_string())
                        })
                        .collect();

                    Series::new(name.into(), &strings)
                }
            }
        }
        _ => return Err(ZarrError::UnsupportedDataType(format!("{:?}", dtype))),
    };

    Ok(series)
}

fn series_to_bytes(series: &Series, use_dictionary: bool) -> Result<(Vec<u8>, Option<Vec<String>>)> {
    let dtype = series.dtype();

    match dtype {
        DataType::Int8 => {
            let bytes: Vec<u8> = series
                .i8()?
                .into_no_null_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok((bytes, None))
        }
        DataType::Int16 => {
            let bytes: Vec<u8> = series
                .i16()?
                .into_no_null_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok((bytes, None))
        }
        DataType::Int32 => {
            let bytes: Vec<u8> = series
                .i32()?
                .into_no_null_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok((bytes, None))
        }
        DataType::Int64 => {
            let bytes: Vec<u8> = series
                .i64()?
                .into_no_null_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok((bytes, None))
        }
        DataType::UInt8 => {
            let bytes: Vec<u8> = series.u8()?.into_no_null_iter().collect();
            Ok((bytes, None))
        }
        DataType::UInt16 => {
            let bytes: Vec<u8> = series
                .u16()?
                .into_no_null_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok((bytes, None))
        }
        DataType::UInt32 => {
            let bytes: Vec<u8> = series
                .u32()?
                .into_no_null_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok((bytes, None))
        }
        DataType::UInt64 => {
            let bytes: Vec<u8> = series
                .u64()?
                .into_no_null_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok((bytes, None))
        }
        DataType::Float32 => {
            let bytes: Vec<u8> = series
                .f32()?
                .into_no_null_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok((bytes, None))
        }
        DataType::Float64 => {
            let bytes: Vec<u8> = series
                .f64()?
                .into_no_null_iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            Ok((bytes, None))
        }
        DataType::Boolean => {
            let bytes: Vec<u8> = series
                .bool()?
                .into_no_null_iter()
                .map(|v| if v { 1u8 } else { 0u8 })
                .collect();
            Ok((bytes, None))
        }
        DataType::String => {
            let str_ca = series.str()?;

            if use_dictionary {
                // Build dictionary
                let mut dict_map: HashMap<String, u32> = HashMap::new();
                let mut dictionary: Vec<String> = Vec::new();
                let mut indices: Vec<u32> = Vec::new();

                for opt_s in str_ca.into_iter() {
                    let s = opt_s.unwrap_or("");
                    let idx = if let Some(&existing_idx) = dict_map.get(s) {
                        existing_idx
                    } else {
                        let new_idx = dictionary.len() as u32;
                        dict_map.insert(s.to_string(), new_idx);
                        dictionary.push(s.to_string());
                        new_idx
                    };
                    indices.push(idx);
                }

                let bytes: Vec<u8> = indices
                    .iter()
                    .flat_map(|&idx| idx.to_le_bytes())
                    .collect();

                Ok((bytes, Some(dictionary)))
            } else {
                // Variable-length encoding: 4-byte length + UTF-8 bytes
                let mut bytes: Vec<u8> = Vec::new();

                for opt_s in str_ca.into_iter() {
                    let s = opt_s.unwrap_or("");
                    let s_bytes = s.as_bytes();
                    let len = s_bytes.len() as u32;
                    bytes.extend_from_slice(&len.to_le_bytes());
                    bytes.extend_from_slice(s_bytes);
                }

                Ok((bytes, None))
            }
        }
        DataType::Categorical(_, _) => {
            // Convert categorical to string and use dictionary encoding
            let string_series = series.cast(&DataType::String)?;
            series_to_bytes(&string_series, true)
        }
        _ => Err(ZarrError::UnsupportedDataType(format!("{:?}", dtype))),
    }
}

/// Write a Polars DataFrame to a zipped Zarr file.
pub fn write_zarr_zip(df: &DataFrame, path: impl AsRef<Path>) -> Result<()> {
    write_zarr_zip_with_options(df, path, &WriteOptions::default())
}

/// Write a Polars DataFrame to a zipped Zarr file with options.
pub fn write_zarr_zip_with_options(
    df: &DataFrame,
    path: impl AsRef<Path>,
    options: &WriteOptions,
) -> Result<()> {
    let file = File::create(path.as_ref())?;
    let mut zip = ZipWriter::new(file);

    let zip_options: FileOptions<()> =
        FileOptions::default().compression_method(CompressionMethod::Stored);

    // Write root group metadata
    let group_metadata = ZarrGroupMetadata {
        zarr_format: 3,
        node_type: "group".to_string(),
    };

    zip.start_file("zarr.json", zip_options.clone())?;
    zip.write_all(serde_json::to_string_pretty(&group_metadata)?.as_bytes())?;

    // Write each column as a 1D array
    for col in df.get_columns() {
        let series = col.as_materialized_series();
        let name = series.name().to_string();
        let len = series.len();
        let dtype = series.dtype();

        // Check if this column should use dictionary encoding
        let use_dictionary = options.dictionary_columns.contains(&name)
            || matches!(dtype, DataType::Categorical(_, _));

        let zarr_dtype = polars_dtype_to_zarr(dtype, use_dictionary)?;

        // Create array metadata
        let fill_value = if matches!(dtype, DataType::String) {
            serde_json::json!("")
        } else {
            serde_json::json!(0)
        };

        let array_metadata = ZarrArrayMetadata {
            zarr_format: 3,
            node_type: "array".to_string(),
            shape: vec![len],
            data_type: zarr_dtype,
            chunk_grid: ChunkGrid {
                name: "regular".to_string(),
                configuration: ChunkGridConfig {
                    chunk_shape: vec![len],
                },
            },
            chunk_key_encoding: ChunkKeyEncoding {
                name: "default".to_string(),
            },
            fill_value,
            codecs: vec![serde_json::json!({
                "name": "bytes",
                "configuration": {
                    "endian": "little"
                }
            })],
            attributes: None,
        };

        // Write array metadata
        zip.start_file(format!("{}/zarr.json", name), zip_options.clone())?;
        zip.write_all(serde_json::to_string_pretty(&array_metadata)?.as_bytes())?;

        // Write chunk data
        let (chunk_data, dictionary) = series_to_bytes(series, use_dictionary)?;
        zip.start_file(format!("{}/c/0", name), zip_options.clone())?;
        zip.write_all(&chunk_data)?;

        // Write dictionary if present
        if let Some(dict) = dictionary {
            zip.start_file(format!("{}/dictionary.json", name), zip_options.clone())?;
            zip.write_all(serde_json::to_string_pretty(&dict)?.as_bytes())?;
        }
    }

    zip.finish()?;
    Ok(())
}
