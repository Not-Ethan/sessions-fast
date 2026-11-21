//! A minimal Parquet writer: one row group, one PLAIN uncompressed data page
//! (v1) per column, Thrift compact metadata. Strings are BYTE_ARRAY annotated
//! UTF8/STRING, `day` is INT32 annotated DATE.

use std::io;
use std::path::Path;

pub enum Data<'a> {
    /// Concatenated bytes plus end offsets (one per row). `valid` = None for a
    /// REQUIRED column; Some(v) for OPTIONAL with v[i] = row i is non-null
    /// (null rows have an empty range).
    Str { bytes: &'a [u8], ends: &'a [u32], valid: Option<&'a [bool]> },
    I32 { values: &'a [i32], date: bool },
    I64 { values: &'a [i64] },
}

pub struct Column<'a> {
    pub name: &'a str,
    pub data: Data<'a>,
}

// Parquet enums
const T_INT32: i32 = 1;
const T_INT64: i32 = 2;
const T_BYTE_ARRAY: i32 = 6;
const REQUIRED: i32 = 0;
const OPTIONAL: i32 = 1;
const CONV_UTF8: i32 = 0;
const CONV_DATE: i32 = 6;
const ENC_PLAIN: i32 = 0;
const ENC_RLE: i32 = 3;

// Thrift compact types
const CT_I32: u8 = 5;
const CT_I64: u8 = 6;
const CT_BINARY: u8 = 8;
const CT_LIST: u8 = 9;
const CT_STRUCT: u8 = 12;

struct Thrift {
    buf: Vec<u8>,
    last: Vec<i16>,
    cur: i16,
}

impl Thrift {
    fn new() -> Self {
        Thrift { buf: Vec::new(), last: Vec::new(), cur: 0 }
    }
    fn varint(&mut self, mut v: u64) {
        while v >= 0x80 {
            self.buf.push((v as u8) | 0x80);
            v >>= 7;
        }
        self.buf.push(v as u8);
    }
    fn zigzag(v: i64) -> u64 {
        ((v << 1) ^ (v >> 63)) as u64
    }
    fn field(&mut self, id: i16, ty: u8) {
        let delta = id - self.cur;
        if delta > 0 && delta <= 15 {
            self.buf.push(((delta as u8) << 4) | ty);
        } else {
            self.buf.push(ty);
            self.varint(Self::zigzag(id as i64));
        }
        self.cur = id;
    }
    fn i32(&mut self, id: i16, v: i32) {
        self.field(id, CT_I32);
        self.varint(Self::zigzag(v as i64));
    }
    fn i64(&mut self, id: i16, v: i64) {
        self.field(id, CT_I64);
        self.varint(Self::zigzag(v));
    }
    fn binary(&mut self, id: i16, b: &[u8]) {
        self.field(id, CT_BINARY);
        self.varint(b.len() as u64);
        self.buf.extend_from_slice(b);
    }
    fn struct_begin(&mut self, id: i16) {
        self.field(id, CT_STRUCT);
        self.push_scope();
    }
    fn push_scope(&mut self) {
        self.last.push(self.cur);
        self.cur = 0;
    }
    fn struct_end(&mut self) {
        self.buf.push(0);
        self.cur = self.last.pop().unwrap();
    }
    fn list_begin(&mut self, id: i16, elem: u8, n: usize) {
        self.field(id, CT_LIST);
        if n < 15 {
            self.buf.push(((n as u8) << 4) | elem);
        } else {
            self.buf.push(0xF0 | elem);
            self.varint(n as u64);
        }
    }
    /// A struct element inside a list.
    fn elem_begin(&mut self) {
        self.push_scope();
    }
    fn list_i32(&mut self, v: i32) {
        self.varint(Self::zigzag(v as i64));
    }
    fn list_binary(&mut self, b: &[u8]) {
        self.varint(b.len() as u64);
        self.buf.extend_from_slice(b);
    }
}

/// Appends fixed-width integers in little-endian byte order.
fn extend_le<T: Copy + ToLe>(out: &mut Vec<u8>, values: &[T]) {
    if cfg!(target_endian = "little") {
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values)) };
        out.extend_from_slice(bytes);
    } else {
        for &v in values {
            v.put_le(out);
        }
    }
}

trait ToLe {
    fn put_le(self, out: &mut Vec<u8>);
}
impl ToLe for i32 {
    fn put_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}
impl ToLe for i64 {
    fn put_le(self, out: &mut Vec<u8>) {
        out.extend_from_slice(&self.to_le_bytes());
    }
}

fn varint_len(mut v: u64) -> usize {
    let mut n = 1;
    while v >= 0x80 {
        v >>= 7;
        n += 1;
    }
    n
}

struct ChunkMeta {
    ty: i32,
    name: String,
    num_values: i64,
    size: i64,
    offset: i64,
}

/// Writes one Parquet file with `nrows` rows.
pub fn write_file(path: &Path, nrows: usize, cols: &[Column]) -> io::Result<()> {
    std::fs::write(path, encode(nrows, cols))
}

/// The bytes of one Parquet file with `nrows` rows.
pub fn encode(nrows: usize, cols: &[Column]) -> Vec<u8> {
    let mut est = 1024;
    for c in cols {
        est += match &c.data {
            Data::Str { bytes, .. } => bytes.len() + 5 * nrows + 64,
            Data::I32 { .. } => 4 * nrows + 64,
            Data::I64 { .. } => 8 * nrows + 64,
        };
    }
    let mut out: Vec<u8> = Vec::with_capacity(est);
    out.extend_from_slice(b"PAR1");
    let mut metas = Vec::new();
    if nrows > 0 {
        for c in cols {
            let offset = out.len();
            let (ty, data_len, optional) = match &c.data {
                Data::Str { bytes, ends, valid } => {
                    assert_eq!(ends.len(), nrows);
                    let nonnull = match valid {
                        Some(v) => v.iter().filter(|&&x| x).count(),
                        None => nrows,
                    };
                    let mut len = bytes.len() + 4 * nonnull;
                    if valid.is_some() {
                        let groups = nrows.div_ceil(8);
                        len += 4 + varint_len(((groups as u64) << 1) | 1) + groups;
                    }
                    (T_BYTE_ARRAY, len, valid.is_some())
                }
                Data::I32 { values, .. } => {
                    assert_eq!(values.len(), nrows);
                    (T_INT32, 4 * nrows, false)
                }
                Data::I64 { values } => {
                    assert_eq!(values.len(), nrows);
                    (T_INT64, 8 * nrows, false)
                }
            };
            // Page header
            let mut h = Thrift::new();
            h.i32(1, 0); // DATA_PAGE
            h.i32(2, data_len as i32);
            h.i32(3, data_len as i32);
            h.struct_begin(5);
            h.i32(1, nrows as i32);
            h.i32(2, ENC_PLAIN);
            h.i32(3, ENC_RLE);
            h.i32(4, ENC_RLE);
            h.struct_end();
            h.buf.push(0);
            out.extend_from_slice(&h.buf);
            let body_start = out.len();
            match &c.data {
                Data::Str { bytes, ends, valid } => {
                    if let Some(v) = valid {
                        let groups = nrows.div_ceil(8);
                        let mut lv = Thrift::new();
                        lv.varint(((groups as u64) << 1) | 1);
                        let total = lv.buf.len() + groups;
                        out.extend_from_slice(&(total as u32).to_le_bytes());
                        out.extend_from_slice(&lv.buf);
                        let mut byte = 0u8;
                        for (i, &x) in v.iter().enumerate() {
                            byte |= (x as u8) << (i & 7);
                            if i & 7 == 7 {
                                out.push(byte);
                                byte = 0;
                            }
                        }
                        if nrows & 7 != 0 {
                            out.push(byte);
                        }
                    }
                    let mut prev = 0usize;
                    for (i, &e) in ends.iter().enumerate() {
                        let e = e as usize;
                        let is_valid = match valid {
                            Some(v) => v[i],
                            None => true,
                        };
                        if is_valid {
                            out.extend_from_slice(&((e - prev) as u32).to_le_bytes());
                            out.extend_from_slice(&bytes[prev..e]);
                        }
                        prev = e;
                    }
                }
                Data::I32 { values, .. } => extend_le(&mut out, values),
                Data::I64 { values } => extend_le(&mut out, values),
            }
            assert_eq!(out.len() - body_start, data_len, "page size mismatch for {}", c.name);
            let _ = optional;
            metas.push(ChunkMeta {
                ty,
                name: c.name.to_string(),
                num_values: nrows as i64,
                size: (out.len() - offset) as i64,
                offset: offset as i64,
            });
        }
    }

    // FileMetaData
    let mut m = Thrift::new();
    m.i32(1, 1);
    m.list_begin(2, CT_STRUCT, cols.len() + 1);
    m.elem_begin();
    m.binary(4, b"schema");
    m.i32(5, cols.len() as i32);
    m.struct_end();
    for c in cols {
        m.elem_begin();
        let (ty, rep, conv) = match &c.data {
            Data::Str { valid, .. } => (T_BYTE_ARRAY, if valid.is_some() { OPTIONAL } else { REQUIRED }, Some(CONV_UTF8)),
            Data::I32 { date, .. } => (T_INT32, REQUIRED, if *date { Some(CONV_DATE) } else { None }),
            Data::I64 { .. } => (T_INT64, REQUIRED, None),
        };
        m.i32(1, ty);
        m.i32(3, rep);
        m.binary(4, c.name.as_bytes());
        if let Some(cv) = conv {
            m.i32(6, cv);
            // LogicalType union: STRING = field 1, DATE = field 6 (empty structs)
            m.struct_begin(10);
            m.struct_begin(if cv == CONV_UTF8 { 1 } else { 6 });
            m.struct_end();
            m.struct_end();
        }
        m.struct_end();
    }
    m.i64(3, nrows as i64);
    if nrows > 0 {
        m.list_begin(4, CT_STRUCT, 1);
        m.elem_begin();
        m.list_begin(1, CT_STRUCT, metas.len());
        let mut total = 0i64;
        for cm in &metas {
            total += cm.size;
            m.elem_begin();
            m.i64(2, cm.offset);
            m.struct_begin(3);
            m.i32(1, cm.ty);
            m.list_begin(2, CT_I32, 2);
            m.list_i32(ENC_PLAIN);
            m.list_i32(ENC_RLE);
            m.list_begin(3, CT_BINARY, 1);
            m.list_binary(cm.name.as_bytes());
            m.i32(4, 0); // UNCOMPRESSED
            m.i64(5, cm.num_values);
            m.i64(6, cm.size);
            m.i64(7, cm.size);
            m.i64(9, cm.offset);
            m.struct_end();
            m.struct_end();
        }
        m.i64(2, total);
        m.i64(3, nrows as i64);
        m.struct_end();
    } else {
        m.list_begin(4, CT_STRUCT, 0);
    }
    m.binary(6, b"shopping-sessions");
    m.buf.push(0);
    out.extend_from_slice(&m.buf);
    out.extend_from_slice(&(m.buf.len() as u32).to_le_bytes());
    out.extend_from_slice(b"PAR1");
    out
}
