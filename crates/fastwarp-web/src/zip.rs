// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

//! A zip archive in memory with stored (uncompressed) entries, for files that are compressed
//! already (`.nii.gz`) or small. Each entry is written straight into the archive and its header
//! completed afterwards, so no entry is held twice.

use std::io::{self, Write};

const LOCAL_HEADER: u32 = 0x0403_4b50;
const CENTRAL_HEADER: u32 = 0x0201_4b50;
const END_OF_CENTRAL_DIRECTORY: u32 = 0x0605_4b50;
const VERSION: u16 = 20;
const UTF8_NAMES: u16 = 1 << 11;
const LOCAL_HEADER_BYTES: usize = 30;

struct Entry {
    name: String,
    offset: u32,
    crc: u32,
    size: u32,
    time: u16,
    date: u16,
}

pub struct ZipWriter {
    buf: Vec<u8>,
    entries: Vec<Entry>,
    dos_time: u16,
    dos_date: u16,
}

impl Default for ZipWriter {
    fn default() -> Self {
        Self::new()
    }
}

fn too_large(what: &str) -> io::Error {
    io::Error::other(format!(
        "{what} is too large for a zip archive without the zip64 extension (4 GiB, 65535 entries)"
    ))
}

impl ZipWriter {
    /// An empty archive whose entries carry the time 1980-01-01 00:00.
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            entries: Vec::new(),
            dos_time: 0,
            dos_date: (1 << 5) | 1,
        }
    }

    /// An empty archive like [`ZipWriter::new`] with room for `bytes` bytes, so that an archive
    /// that stays within them is never moved while it grows; without room when that much memory
    /// cannot be had, like [`ZipWriter::new`].
    pub fn with_capacity(bytes: usize) -> Self {
        let mut buf = Vec::new();
        if buf.try_reserve_exact(bytes).is_err() {
            return Self::new();
        }
        Self { buf, ..Self::new() }
    }

    /// Sets the modification time of the entries added after it (local time, 1980 to 2107).
    pub fn set_time(
        &mut self,
        year: u16,
        month: u16,
        day: u16,
        hour: u16,
        minute: u16,
        second: u16,
    ) {
        if (1980..=2107).contains(&year)
            && (1..=12).contains(&month)
            && (1..=31).contains(&day)
            && hour < 24
            && minute < 60
            && second < 60
        {
            self.dos_date = ((year - 1980) << 9) | (month << 5) | day;
            self.dos_time = (hour << 11) | (minute << 5) | (second / 2);
        }
    }

    /// Starts an entry; write its content into the returned writer, then call
    /// [`EntryWriter::finish`].
    pub fn begin(&mut self, name: &str) -> io::Result<EntryWriter<'_>> {
        if name.is_empty() || name.len() > usize::from(u16::MAX) {
            return Err(io::Error::other(format!("invalid zip entry name {name:?}")));
        }
        if self.entries.len() >= usize::from(u16::MAX) {
            return Err(too_large("the number of entries"));
        }
        let offset = u32::try_from(self.buf.len()).map_err(|_| too_large("the archive"))?;
        let b = &mut self.buf;
        put32(b, LOCAL_HEADER);
        put16(b, VERSION);
        put16(b, UTF8_NAMES);
        put16(b, 0);
        put16(b, self.dos_time);
        put16(b, self.dos_date);
        put32(b, 0);
        put32(b, 0);
        put32(b, 0);
        put16(b, name.len() as u16);
        put16(b, 0);
        b.extend_from_slice(name.as_bytes());
        Ok(EntryWriter {
            zip: self,
            name: name.to_string(),
            offset,
            hasher: crc32fast::Hasher::new(),
            size: 0,
        })
    }

    /// Adds an entry with `data` as its content.
    pub fn add(&mut self, name: &str, data: &[u8]) -> io::Result<()> {
        let mut entry = self.begin(name)?;
        entry.write_all(data)?;
        entry.finish()
    }

    /// Writes the central directory and returns the archive.
    pub fn finish(mut self) -> io::Result<Vec<u8>> {
        let start = u32::try_from(self.buf.len()).map_err(|_| too_large("the archive"))?;
        let b = &mut self.buf;
        for e in &self.entries {
            put32(b, CENTRAL_HEADER);
            put16(b, VERSION);
            put16(b, VERSION);
            put16(b, UTF8_NAMES);
            put16(b, 0);
            put16(b, e.time);
            put16(b, e.date);
            put32(b, e.crc);
            put32(b, e.size);
            put32(b, e.size);
            put16(b, e.name.len() as u16);
            put16(b, 0);
            put16(b, 0);
            put16(b, 0);
            put16(b, 0);
            put32(b, 0);
            put32(b, e.offset);
            b.extend_from_slice(e.name.as_bytes());
        }
        let size = u32::try_from(b.len() - start as usize).map_err(|_| too_large("the archive"))?;
        u32::try_from(b.len() + 22).map_err(|_| too_large("the archive"))?;
        let count = self.entries.len() as u16;
        put32(b, END_OF_CENTRAL_DIRECTORY);
        put16(b, 0);
        put16(b, 0);
        put16(b, count);
        put16(b, count);
        put32(b, size);
        put32(b, start);
        put16(b, 0);
        Ok(self.buf)
    }
}

/// The content of one entry, written into the archive as it comes.
pub struct EntryWriter<'a> {
    zip: &'a mut ZipWriter,
    name: String,
    offset: u32,
    hasher: crc32fast::Hasher,
    size: u64,
}

impl EntryWriter<'_> {
    /// Completes the entry's header with its checksum and size.
    pub fn finish(self) -> io::Result<()> {
        let size = u32::try_from(self.size).map_err(|_| too_large(&self.name))?;
        u32::try_from(self.zip.buf.len()).map_err(|_| too_large("the archive"))?;
        let crc = self.hasher.finalize();
        let at = self.offset as usize;
        let header = &mut self.zip.buf[at..at + LOCAL_HEADER_BYTES];
        header[14..18].copy_from_slice(&crc.to_le_bytes());
        header[18..22].copy_from_slice(&size.to_le_bytes());
        header[22..26].copy_from_slice(&size.to_le_bytes());
        self.zip.entries.push(Entry {
            name: self.name,
            offset: self.offset,
            crc,
            size,
            time: self.zip.dos_time,
            date: self.zip.dos_date,
        });
        Ok(())
    }
}

impl Write for EntryWriter<'_> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.hasher.update(data);
        self.size += data.len() as u64;
        self.zip.buf.extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn put16(b: &mut Vec<u8>, v: u16) {
    b.extend_from_slice(&v.to_le_bytes());
}

fn put32(b: &mut Vec<u8>, v: u32) {
    b.extend_from_slice(&v.to_le_bytes());
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
pub(crate) mod tests {
    use super::*;

    fn u16_at(b: &[u8], at: usize) -> u16 {
        u16::from_le_bytes([b[at], b[at + 1]])
    }

    fn u32_at(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
    }

    #[derive(Debug, PartialEq)]
    pub(crate) struct Read {
        pub name: String,
        pub data: Vec<u8>,
        pub time: u16,
        pub date: u16,
    }

    pub(crate) fn read_zip(b: &[u8]) -> Vec<Read> {
        let end = b.len() - 22;
        assert_eq!(u32_at(b, end), END_OF_CENTRAL_DIRECTORY);
        assert_eq!(u16_at(b, end + 20), 0);
        let count = u16_at(b, end + 10) as usize;
        assert_eq!(u16_at(b, end + 8), count as u16);
        let cd_size = u32_at(b, end + 12) as usize;
        let mut at = u32_at(b, end + 16) as usize;
        assert_eq!(at + cd_size, end);
        let mut out = Vec::new();
        for _ in 0..count {
            assert_eq!(u32_at(b, at), CENTRAL_HEADER);
            assert_eq!(u16_at(b, at + 10), 0, "stored");
            assert_eq!(u16_at(b, at + 8), UTF8_NAMES);
            let crc = u32_at(b, at + 16);
            let csize = u32_at(b, at + 20) as usize;
            let usize_ = u32_at(b, at + 24) as usize;
            assert_eq!(csize, usize_);
            let name_len = u16_at(b, at + 28) as usize;
            let extra = u16_at(b, at + 30) as usize;
            let comment = u16_at(b, at + 32) as usize;
            let local = u32_at(b, at + 42) as usize;
            let name = String::from_utf8(b[at + 46..at + 46 + name_len].to_vec()).unwrap();
            let (time, date) = (u16_at(b, at + 12), u16_at(b, at + 14));
            at += 46 + name_len + extra + comment;

            assert_eq!(u32_at(b, local), LOCAL_HEADER);
            assert_eq!(u16_at(b, local + 8), 0);
            assert_eq!(u32_at(b, local + 14), crc);
            assert_eq!(u32_at(b, local + 18) as usize, csize);
            assert_eq!(u32_at(b, local + 22) as usize, csize);
            assert_eq!((u16_at(b, local + 10), u16_at(b, local + 12)), (time, date));
            let lname = u16_at(b, local + 26) as usize;
            let lextra = u16_at(b, local + 28) as usize;
            assert_eq!(&b[local + 30..local + 30 + lname], name.as_bytes());
            let start = local + 30 + lname + lextra;
            let data = b[start..start + csize].to_vec();
            assert_eq!(crc32fast::hash(&data), crc, "{name}");
            out.push(Read {
                name,
                data,
                time,
                date,
            });
        }
        assert_eq!(at, end);
        out
    }

    #[test]
    fn crc_matches_the_standard_check_value() {
        let mut z = ZipWriter::new();
        z.add("check.txt", b"123456789").unwrap();
        let b = z.finish().unwrap();
        assert_eq!(u32_at(&b, 14), 0xCBF4_3926);
    }

    #[test]
    fn entries_read_back() {
        let big: Vec<u8> = (0..300_000u32).map(|i| (i * 7 % 251) as u8).collect();
        let mut z = ZipWriter::new();
        z.add("README.txt", b"hello\n").unwrap();
        z.add("empty", b"").unwrap();
        let mut e = z.begin("1Warp.nii.gz").unwrap();
        for chunk in big.chunks(4096) {
            e.write_all(chunk).unwrap();
        }
        e.finish().unwrap();
        z.set_time(2026, 11, 22, 13, 45, 31);
        z.add("dir/Grüße.json", b"{}").unwrap();
        let b = z.finish().unwrap();
        let r = read_zip(&b);
        let names: Vec<_> = r.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(
            names,
            ["README.txt", "empty", "1Warp.nii.gz", "dir/Grüße.json"]
        );
        assert_eq!(r[0].data, b"hello\n");
        assert!(r[1].data.is_empty());
        assert_eq!(r[2].data, big);
        assert_eq!(r[3].data, b"{}");
        assert_eq!(r[3].date, (46 << 9) | (11 << 5) | 22);
        assert_eq!(r[3].time, (13 << 11) | (45 << 5) | 15);
    }

    #[test]
    fn an_empty_archive_is_the_end_record_alone() {
        let b = ZipWriter::new().finish().unwrap();
        assert_eq!(b.len(), 22);
        assert!(read_zip(&b).is_empty());
    }

    #[test]
    fn invalid_names_and_times_are_refused() {
        let mut z = ZipWriter::new();
        assert!(z.begin("").is_err());
        z.set_time(1979, 1, 1, 0, 0, 0);
        z.set_time(2026, 13, 1, 0, 0, 0);
        z.add("a", b"x").unwrap();
        let r = read_zip(&z.finish().unwrap());
        assert_eq!((r[0].date, r[0].time), ((1 << 5) | 1, 0));
    }
}
