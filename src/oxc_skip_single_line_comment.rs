use std::arch::x86_64::{
    _mm256_add_epi8, _mm256_and_si256, _mm256_cmpeq_epi8, _mm256_load_si256, _mm256_loadu_si256,
    _mm256_movemask_epi8, _mm256_or_si256, _mm256_set1_epi8, _mm256_shuffle_epi8,
    _mm256_srli_epi16, _mm_add_epi8, _mm_and_si128, _mm_cmpeq_epi8, _mm_load_si128,
    _mm_loadu_si128, _mm_movemask_epi8, _mm_or_si128, _mm_set1_epi8, _mm_shuffle_epi8,
    _mm_srli_epi16,
};

pub fn skip_single_line_comment_scalar(string: &[u8]) -> usize {
    for (idx, &byte) in string.iter().enumerate() {
        match byte {
            LF | CR => return idx,
            // SAFETY: string must be valid UTF-8.
            XX => match unsafe { string.get_unchecked(idx + 1..idx + 3) } {
                &[0x80, 0xA8 | 0xA9] => return idx,
                _ => {}
            },
            _ => {}
        }
    }
    string.len()
}

const LF: u8 = 0x0A;
const CR: u8 = 0x0D;
const XX: u8 = 0xE2;

#[derive(Clone, Copy)]
enum ScalarLookup {
    Empty,
    LineEnding,
    XXByte,
}

const SCALAR_LOOKUP: [ScalarLookup; 256] = {
    let mut lookup = [ScalarLookup::Empty; 256];
    lookup[LF as usize] = ScalarLookup::LineEnding;
    lookup[CR as usize] = ScalarLookup::LineEnding;
    lookup[XX as usize] = ScalarLookup::XXByte;
    lookup
};

pub fn skip_single_line_comment_scalar_lookup(string: &[u8]) -> usize {
    for (idx, &byte) in string.iter().enumerate() {
        match SCALAR_LOOKUP[byte as usize] {
            ScalarLookup::Empty => {}
            ScalarLookup::LineEnding => return idx,
            // SAFETY: string must be valid UTF-8.
            ScalarLookup::XXByte => match unsafe { string.get_unchecked(idx + 1..idx + 3) } {
                &[0x80, 0xA8 | 0xA9] => return idx,
                _ => {}
            },
        }
    }
    string.len()
}

pub fn skip_single_line_comment_memchr(string: &[u8]) -> usize {
    while let Some(idx) = memchr::memchr3(LF, CR, XX, string) {
        match string[idx] {
            LF | CR => return idx,
            XX => match unsafe { string.get_unchecked(idx + 1..idx + 3) } {
                &[0x80, 0xA8 | 0xA9] => return idx,
                _ => {}
            },
            _ => {}
        }
    }
    string.len()
}

/// # Safety
/// Uses OS instructions. Intrinsically unsafe.
pub unsafe fn skip_single_line_comment_simd_naive(string: &[u8]) -> usize {
    // Originally played with pointer ranges directly using
    // `string.as_ptr_range()`; however, the current index-based solution
    // was found to be equal in performance without more `unsafe` code.

    let lf = _mm_set1_epi8(LF as i8);
    let cr = _mm_set1_epi8(CR as i8);
    let xx = _mm_set1_epi8(XX as i8);

    let mut index = 0;
    let mut rest = string;
    while rest.len() >= 16 {
        let chunk = _mm_loadu_si128(rest.as_ptr() as *const _);

        let line_feeds = _mm_cmpeq_epi8(chunk, lf);
        let carriage_returns = _mm_cmpeq_epi8(chunk, cr);
        let xx_bytes = _mm_cmpeq_epi8(chunk, xx);

        let mask = _mm_movemask_epi8(_mm_or_si128(
            xx_bytes,
            _mm_or_si128(line_feeds, carriage_returns),
        ));

        if mask != 0 {
            let offset_in_chunk = mask.trailing_zeros() as usize;
            assert!(offset_in_chunk < 16); // eliminate bound checks
            index += offset_in_chunk;
            if rest[offset_in_chunk] == XX {
                index += skip_single_line_comment_scalar(&rest[offset_in_chunk..]);
            }
            return index;
        }

        index += 16;
        rest = &rest[16..];
    }

    index + skip_single_line_comment_scalar(rest)
}

/// # Safety
/// Uses OS instructions. Intrinsically unsafe.
pub unsafe fn skip_single_line_comment_wide_simd_naive(string: &[u8]) -> usize {
    let lf = _mm256_set1_epi8(LF as i8);
    let cr = _mm256_set1_epi8(CR as i8);
    let xx = _mm256_set1_epi8(XX as i8);

    let mut index = 0;
    let mut rest = string;
    while rest.len() >= 32 {
        let chunk = _mm256_loadu_si256(rest.as_ptr() as *const _);

        let line_feeds = _mm256_cmpeq_epi8(chunk, lf);
        let carriage_returns = _mm256_cmpeq_epi8(chunk, cr);
        let xx_bytes = _mm256_cmpeq_epi8(chunk, xx);

        let mask = _mm256_movemask_epi8(_mm256_or_si256(
            xx_bytes,
            _mm256_or_si256(line_feeds, carriage_returns),
        ));

        if mask != 0 {
            let offset_in_chunk = mask.trailing_zeros() as usize;
            assert!(offset_in_chunk < 32); // eliminate bound check
            index += offset_in_chunk;
            if rest[offset_in_chunk] == XX {
                index += skip_single_line_comment_scalar(&rest[offset_in_chunk..]);
            }
            return index;
        }

        index += 32;
        rest = &rest[32..];
    }

    index + skip_single_line_comment_scalar(rest)
}

// Adapted from https://0x80.pl/articles/simd-byte-lookup.html (Special case 1)

const EMPTY_TAG: u8 = 0;
const LF_TAG: u8 = 1 << 0;
const CR_TAG: u8 = 1 << 1;
const XX_TAG: u8 = 1 << 2;

#[repr(align(16))]
struct Array16([u8; 16]);

const LO_NIBBLES_LOOKUP: Array16 = {
    let mut lookup = [EMPTY_TAG; 16];
    lookup[(LF & 0x0F) as usize] |= LF_TAG;
    lookup[(CR & 0x0F) as usize] |= CR_TAG;
    lookup[(XX & 0x0F) as usize] |= XX_TAG;
    Array16(lookup)
};

const HI_NIBBLES_LOOKUP: Array16 = {
    let mut lookup = [EMPTY_TAG; 16];
    lookup[(LF >> 4) as usize] |= LF_TAG;
    lookup[(CR >> 4) as usize] |= CR_TAG;
    lookup[(XX >> 4) as usize] |= XX_TAG;
    Array16(lookup)
};

/// # Safety
/// Uses OS instructions. Intrinsically unsafe.
pub unsafe fn skip_single_line_comment_simd_shuffle(string: &[u8]) -> usize {
    let lower_nibbles = _mm_set1_epi8(0x0f);
    let lo_nibbles_lookup = _mm_load_si128(LO_NIBBLES_LOOKUP.0.as_ptr() as *const _);
    let hi_nibbles_lookup = _mm_load_si128(HI_NIBBLES_LOOKUP.0.as_ptr() as *const _);
    let magic_add = _mm_set1_epi8(127);

    let mut index = 0;
    let mut rest = string;
    while rest.len() >= 16 {
        let chunk = _mm_loadu_si128(rest.as_ptr() as *const _);

        let lo_nibbles = _mm_and_si128(chunk, lower_nibbles);
        let hi_nibbles = _mm_and_si128(_mm_srli_epi16(chunk, 4), lower_nibbles);

        let lo_translated = _mm_shuffle_epi8(lo_nibbles_lookup, lo_nibbles);
        let hi_translated = _mm_shuffle_epi8(hi_nibbles_lookup, hi_nibbles);

        // Previous:
        // let result = _mm_andnot_si128(
        //     _mm_cmpeq_epi8(_mm_and_si128(lo_translated, hi_translated), zero),
        //     full,
        // );

        // Because all of the tag bits are < 128, adding 127 will set the 7th
        // bit for creating the mask for all non-zero elements.
        let result = _mm_add_epi8(_mm_and_si128(lo_translated, hi_translated), magic_add);

        let mask = _mm_movemask_epi8(result);

        if mask != 0 {
            let offset_in_chunk = mask.trailing_zeros() as usize;
            assert!(offset_in_chunk < 16); // eliminate bound checks
            index += offset_in_chunk;
            if rest[offset_in_chunk] == XX {
                index += skip_single_line_comment_scalar(&rest[offset_in_chunk..]);
            }
            return index;
        }

        index += 16;
        rest = &rest[16..];
    }

    index + skip_single_line_comment_scalar(rest)
}

#[repr(align(32))]
struct Array32([u8; 32]);

const LO_NIBBLES_LOOKUP_WIDE: Array32 = {
    let mut lookup = [EMPTY_TAG; 32];
    lookup[(LF & 0x0F) as usize] |= LF_TAG;
    lookup[(CR & 0x0F) as usize] |= CR_TAG;
    lookup[(XX & 0x0F) as usize] |= XX_TAG;
    Array32(lookup)
};

const HI_NIBBLES_LOOKUP_WIDE: Array32 = {
    let mut lookup = [EMPTY_TAG; 32];
    lookup[(LF >> 4) as usize] |= LF_TAG;
    lookup[(CR >> 4) as usize] |= CR_TAG;
    lookup[(XX >> 4) as usize] |= XX_TAG;
    Array32(lookup)
};

/// # Safety
/// Uses OS instructions. Intrinsically unsafe.
pub unsafe fn skip_single_line_comment_wide_simd_shuffle(string: &[u8]) -> usize {
    let lower_nibbles = _mm256_set1_epi8(0x0f);
    let lo_nibbles_lookup = _mm256_load_si256(LO_NIBBLES_LOOKUP_WIDE.0.as_ptr() as *const _);
    let hi_nibbles_lookup = _mm256_load_si256(HI_NIBBLES_LOOKUP_WIDE.0.as_ptr() as *const _);
    let magic_add = _mm256_set1_epi8(127);

    let mut index = 0;
    let mut rest = string;
    while rest.len() >= 32 {
        let chunk = _mm256_loadu_si256(rest.as_ptr() as *const _);

        let lo_nibbles = _mm256_and_si256(chunk, lower_nibbles);
        let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16(chunk, 4), lower_nibbles);

        let lo_translated = _mm256_shuffle_epi8(lo_nibbles_lookup, lo_nibbles);
        let hi_translated = _mm256_shuffle_epi8(hi_nibbles_lookup, hi_nibbles);

        let intersection = _mm256_and_si256(lo_translated, hi_translated);

        let result = _mm256_add_epi8(intersection, magic_add);

        let mask = _mm256_movemask_epi8(result);

        if mask != 0 {
            let offset_in_chunk = mask.trailing_zeros() as usize;
            assert!(offset_in_chunk < 32); // eliminate bound checks
            index += offset_in_chunk;
            if rest[offset_in_chunk] == XX {
                index += skip_single_line_comment_scalar(&rest[offset_in_chunk..]);
            }
            return index;
        }

        index += 32;
        rest = &rest[32..];
    }

    index + skip_single_line_comment_scalar(rest)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn oxc_skip_single_line_comment() {
        let mut other_bytes = [0 as u8; 256];

        for i in 0..256 {
            other_bytes[i] = i as u8;
        }

        other_bytes[LF as usize] = 0;
        other_bytes[CR as usize] = 0;
        other_bytes[XX as usize] = 0;

        let cases: &[(&[u8], usize)] = &[
            (b"\n", 0),
            (b"Hello, world!\n", 13),
            (b"Hello, world!\n                       ", 13),
            (b"Hello, world!\r                       ", 13),
            ("Hello, world!\u{2028}                      ".as_bytes(), 13),
            ("Hello, world!\u{2029}                      ".as_bytes(), 13),
            ("Hello, world!\u{2029}                      ".as_bytes(), 13),
            (&other_bytes, other_bytes.len()),
        ];

        for &(input, expected_len) in cases {
            eprintln!("Testing {input:?}");
            assert_eq!(skip_single_line_comment_scalar(input), expected_len);
            assert_eq!(skip_single_line_comment_scalar_lookup(input), expected_len);
            assert_eq!(skip_single_line_comment_memchr(input), expected_len);
            assert_eq!(
                unsafe { skip_single_line_comment_simd_naive(input) },
                expected_len
            );
            assert_eq!(
                unsafe { skip_single_line_comment_wide_simd_naive(input) },
                expected_len
            );
            assert_eq!(
                unsafe { skip_single_line_comment_simd_shuffle(input) },
                expected_len
            );
            assert_eq!(
                unsafe { skip_single_line_comment_wide_simd_shuffle(input) },
                expected_len
            );
        }
    }
}
