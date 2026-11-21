// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

use std::fs::File;
use std::io::{BufReader, Cursor, Read};
use std::path::Path;

use flate2::bufread::GzDecoder;
use nifti::{NiftiHeader, NiftiType};

use crate::error::NiftiFileError;

const READ_BUFFER: usize = 1 << 20;
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];
const MAGIC_SINGLE_FILE: &[u8; 4] = b"n+1\0";

enum Source {
    Plain(BufReader<File>),
    Gz(Box<GzDecoder<BufReader<File>>>),
    MemPlain(Cursor<Vec<u8>>),
    MemGz(Box<GzDecoder<Cursor<Vec<u8>>>>),
}

impl Read for Source {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Source::Plain(r) => r.read(buf),
            Source::Gz(r) => r.read(buf),
            Source::MemPlain(r) => r.read(buf),
            Source::MemGz(r) => r.read(buf),
        }
    }
}

pub struct NiftiFile {
    header: NiftiHeader,
    source: Source,
}

fn is_gz_name(path: &Path) -> bool {
    path.file_name()
        .map(|n| n.to_string_lossy().ends_with(".gz"))
        .unwrap_or(false)
}

fn read_header(source: &mut impl Read) -> Result<NiftiHeader, NiftiFileError> {
    let header =
        NiftiHeader::from_reader(source).map_err(|e| NiftiFileError::Header(e.to_string()))?;
    if &header.magic != MAGIC_SINGLE_FILE {
        return Err(NiftiFileError::NoVolumeData);
    }
    Ok(header)
}

const GZIP_MAX_RATIO: usize = 1032;

impl NiftiFile {
    pub fn open(path: &Path) -> Result<Self, NiftiFileError> {
        let file = BufReader::with_capacity(READ_BUFFER, File::open(path)?);
        let source = if is_gz_name(path) {
            Source::Gz(Box::new(GzDecoder::new(file)))
        } else {
            Source::Plain(file)
        };
        Self::from_source(source)
    }

    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, NiftiFileError> {
        let gz = bytes.starts_with(&GZIP_MAGIC);
        let cursor = Cursor::new(bytes);
        let source = if gz {
            Source::MemGz(Box::new(GzDecoder::new(cursor)))
        } else {
            Source::MemPlain(cursor)
        };
        Self::from_source(source)
    }

    pub fn header_from_bytes(bytes: &[u8]) -> Result<NiftiHeader, NiftiFileError> {
        if bytes.starts_with(&GZIP_MAGIC) {
            read_header(&mut GzDecoder::new(bytes))
        } else {
            read_header(&mut &*bytes)
        }
    }

    fn from_source(mut source: Source) -> Result<Self, NiftiFileError> {
        let header = read_header(&mut source)?;
        let data_start = (header.vox_offset as usize).max(352);
        let skip = (data_start - 348) as u64;
        let skipped = std::io::copy(&mut (&mut source).take(skip), &mut std::io::sink())?;
        if skipped != skip {
            return Err(NiftiFileError::IncompatibleLength {
                got: skipped as usize,
                want: skip as usize,
            });
        }
        Ok(Self { header, source })
    }

    pub fn header(&self) -> &NiftiHeader {
        &self.header
    }

    pub fn datatype(&self) -> Result<NiftiType, NiftiFileError> {
        self.header
            .data_type()
            .map_err(|e| NiftiFileError::Header(e.to_string()))
    }

    pub fn voxel_count(&self) -> Result<usize, NiftiFileError> {
        let dims = self
            .header
            .dim()
            .map_err(|e| NiftiFileError::Header(e.to_string()))?;
        dims.iter()
            .try_fold(1usize, |a, &d| a.checked_mul(usize::from(d)))
            .ok_or(NiftiFileError::BadVolumeSize)
    }

    pub fn read_data(mut self, max_bytes: Option<usize>) -> Result<Vec<u8>, NiftiFileError> {
        let datatype = self.datatype()?;
        let total = self
            .voxel_count()?
            .checked_mul(datatype.size_of())
            .ok_or(NiftiFileError::BadVolumeSize)?;
        let want = max_bytes.map_or(total, |m| m.min(total));
        let reserve = match &self.source {
            Source::MemPlain(c) => {
                let left = c.get_ref().len().saturating_sub(c.position() as usize);
                if want > left {
                    return Err(NiftiFileError::IncompatibleLength { got: left, want });
                }
                want
            }
            Source::MemGz(d) => {
                want.min(d.get_ref().get_ref().len().saturating_mul(GZIP_MAX_RATIO))
            }
            _ => want,
        };
        let mut data = Vec::new();
        data.try_reserve_exact(reserve)
            .map_err(|_| NiftiFileError::Reserve(reserve))?;
        let got = (&mut self.source)
            .take(want as u64)
            .read_to_end(&mut data)?;
        if got != want {
            return Err(NiftiFileError::IncompatibleLength { got, want });
        }
        Ok(data)
    }
}
