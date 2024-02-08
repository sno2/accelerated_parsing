//! Expectations for this benchmark:
//! - No irregular line endings
//! - No comments at the end of the source

use accelerated_parsing::oxc_skip_single_line_comment::{
    skip_single_line_comment_memchr, skip_single_line_comment_scalar,
    skip_single_line_comment_scalar_lookup, skip_single_line_comment_simd_naive,
    skip_single_line_comment_simd_shuffle, skip_single_line_comment_wide_simd_naive,
    skip_single_line_comment_wide_simd_shuffle,
};
use criterion::{criterion_group, criterion_main, Criterion};

const REACT_LINE_COMMENTS: &[u8] =
    include_bytes!("../fixtures/react-development-line-comments.txt");

pub fn criterion_benchmark(c: &mut Criterion) {
    let add_benches =
        |group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
         input: &[u8]| {
            group.bench_function("scalar", |b| {
                b.iter(|| {
                    let mut i = 0;
                    while i < input.len() {
                        i += skip_single_line_comment_scalar(&input[i..]) + 1;
                    }
                })
            });
            group.bench_function("scalar_lookup", |b| {
                b.iter(|| {
                    let mut i = 0;
                    while i < input.len() {
                        i += skip_single_line_comment_scalar_lookup(&input[i..]) + 1;
                    }
                })
            });
            group.bench_function("memchr", |b| {
                b.iter(|| {
                    let mut i = 0;
                    while i < input.len() {
                        i += skip_single_line_comment_memchr(&input[i..]) + 1;
                    }
                })
            });
            group.bench_function("simd_naive", |b| {
                b.iter(|| {
                    let mut i = 0;
                    while i < input.len() {
                        i += unsafe { skip_single_line_comment_simd_naive(&input[i..]) } + 1;
                    }
                })
            });
            group.bench_function("wide_simd_naive", |b| {
                b.iter(|| {
                    let mut i = 0;
                    while i < input.len() {
                        i += unsafe { skip_single_line_comment_wide_simd_naive(&input[i..]) } + 1;
                    }
                })
            });
            group.bench_function("simd_shuffle", |b| {
                b.iter(|| {
                    let mut i = 0;
                    while i < input.len() {
                        i += unsafe { skip_single_line_comment_simd_shuffle(&input[i..]) } + 1;
                    }
                })
            });
            group.bench_function("wide_simd_shuffle", |b| {
                b.iter(|| {
                    let mut i = 0;
                    while i < input.len() {
                        i += unsafe { skip_single_line_comment_wide_simd_shuffle(&input[i..]) } + 1;
                    }
                })
            });
        };

    let mut short_padded = c.benchmark_group("react_line_comments");
    add_benches(&mut short_padded, REACT_LINE_COMMENTS);
    short_padded.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
