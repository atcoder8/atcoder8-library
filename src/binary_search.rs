//! Implement bisection search for ranges represented by 32-bit or larger
//! integers or real primitive types.

use std::ops::{
    Range, RangeBounds, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

// Binary search with `i32`

fn binary_search_with_i32_for_inc<R, F>(rng: R, is_ok: F) -> Option<i32>
where
    R: RangeBounds<i32>,
    F: Fn(i32) -> bool,
{
    let mut left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start + 1,
        std::ops::Bound::Unbounded => std::i32::MIN,
    };

    let right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end + 1,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::i32::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    if is_ok(left) {
        return Some(left);
    }

    if !is_ok(right - 1) {
        return None;
    }

    let mut size = right - left;

    while size > 1 {
        let half = size / 2;
        let mid = left + half;

        if !is_ok(mid) {
            left = mid;
        }
        size -= half;
    }

    let boundary = if is_ok(left) { left } else { left + 1 };
    Some(boundary)
}

fn binary_search_with_i32_for_dec<R, F>(rng: R, is_ok: F) -> Option<i32>
where
    R: RangeBounds<i32>,
    F: Fn(i32) -> bool,
{
    let left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start + 1,
        std::ops::Bound::Unbounded => std::i32::MIN,
    };

    let right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end + 1,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::i32::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    if is_ok(right - 1) {
        return Some(right - 1);
    }

    if !is_ok(left) {
        return None;
    }

    let boundary = binary_search_with_i32_for_inc(rng, |mid| !is_ok(mid)).unwrap() - 1;
    Some(boundary)
}

pub fn binary_search_with_i32<R, F>(rng: R, is_ok: F, dec: bool) -> Option<i32>
where
    R: RangeBounds<i32>,
    F: Fn(i32) -> bool,
{
    if dec {
        binary_search_with_i32_for_dec(rng, is_ok)
    } else {
        binary_search_with_i32_for_inc(rng, is_ok)
    }
}

pub trait BinarySearchWithI32: Sized + RangeBounds<i32> {
    /// Returns the smallest integer `x` in the range for which `is_ok(x) = true`.
    /// If no such integer exists, returns None.
    ///
    /// # Arguments
    ///
    /// * `is_ok` - Monotonic function. Weakly monotonicity is also included.
    /// * `dec` - Indicates that `is_ok` is a monotonically decreasing function if true,
    /// or a monotonically increasing function if false.
    fn binary_search<F>(self, is_ok: F, dec: bool) -> Option<i32>
    where
        F: Fn(i32) -> bool,
    {
        binary_search_with_i32(self, is_ok, dec)
    }
}

impl BinarySearchWithI32 for RangeFull {}

impl BinarySearchWithI32 for RangeTo<i32> {}

impl BinarySearchWithI32 for RangeToInclusive<i32> {}

impl BinarySearchWithI32 for RangeFrom<i32> {}

impl BinarySearchWithI32 for Range<i32> {}

impl BinarySearchWithI32 for RangeInclusive<i32> {}

// Binary search with `i64`

fn binary_search_with_i64_for_inc<R, F>(rng: R, is_ok: F) -> Option<i64>
where
    R: RangeBounds<i64>,
    F: Fn(i64) -> bool,
{
    let mut left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start + 1,
        std::ops::Bound::Unbounded => std::i64::MIN,
    };

    let right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end + 1,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::i64::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    if is_ok(left) {
        return Some(left);
    }

    if !is_ok(right - 1) {
        return None;
    }

    let mut size = right - left;

    while size > 1 {
        let half = size / 2;
        let mid = left + half;

        if !is_ok(mid) {
            left = mid;
        }
        size -= half;
    }

    let boundary = if is_ok(left) { left } else { left + 1 };
    Some(boundary)
}

fn binary_search_with_i64_for_dec<R, F>(rng: R, is_ok: F) -> Option<i64>
where
    R: RangeBounds<i64>,
    F: Fn(i64) -> bool,
{
    let left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start + 1,
        std::ops::Bound::Unbounded => std::i64::MIN,
    };

    let right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end + 1,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::i64::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    if is_ok(right - 1) {
        return Some(right - 1);
    }

    if !is_ok(left) {
        return None;
    }

    let boundary = binary_search_with_i64_for_inc(rng, |mid| !is_ok(mid)).unwrap() - 1;
    Some(boundary)
}

pub fn binary_search_with_i64<R, F>(rng: R, is_ok: F, dec: bool) -> Option<i64>
where
    R: RangeBounds<i64>,
    F: Fn(i64) -> bool,
{
    if dec {
        binary_search_with_i64_for_dec(rng, is_ok)
    } else {
        binary_search_with_i64_for_inc(rng, is_ok)
    }
}

pub trait BinarySearchWithI64: Sized + RangeBounds<i64> {
    /// Returns the smallest integer `x` in the range for which `is_ok(x) = true`.
    /// If no such integer exists, returns None.
    ///
    /// # Arguments
    ///
    /// * `is_ok` - Monotonic function. Weakly monotonicity is also included.
    /// * `dec` - Indicates that `is_ok` is a monotonically decreasing function if true,
    /// or a monotonically increasing function if false.
    fn binary_search<F>(self, is_ok: F, dec: bool) -> Option<i64>
    where
        F: Fn(i64) -> bool,
    {
        binary_search_with_i64(self, is_ok, dec)
    }
}

impl BinarySearchWithI64 for RangeFull {}

impl BinarySearchWithI64 for RangeTo<i64> {}

impl BinarySearchWithI64 for RangeToInclusive<i64> {}

impl BinarySearchWithI64 for RangeFrom<i64> {}

impl BinarySearchWithI64 for Range<i64> {}

impl BinarySearchWithI64 for RangeInclusive<i64> {}

// Binary search with `i128`

fn binary_search_with_i128_for_inc<R, F>(rng: R, is_ok: F) -> Option<i128>
where
    R: RangeBounds<i128>,
    F: Fn(i128) -> bool,
{
    let mut left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start + 1,
        std::ops::Bound::Unbounded => std::i128::MIN,
    };

    let right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end + 1,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::i128::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    if is_ok(left) {
        return Some(left);
    }

    if !is_ok(right - 1) {
        return None;
    }

    let mut size = right - left;

    while size > 1 {
        let half = size / 2;
        let mid = left + half;

        if !is_ok(mid) {
            left = mid;
        }
        size -= half;
    }

    let boundary = if is_ok(left) { left } else { left + 1 };
    Some(boundary)
}

fn binary_search_with_i128_for_dec<R, F>(rng: R, is_ok: F) -> Option<i128>
where
    R: RangeBounds<i128>,
    F: Fn(i128) -> bool,
{
    let left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start + 1,
        std::ops::Bound::Unbounded => std::i128::MIN,
    };

    let right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end + 1,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::i128::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    if is_ok(right - 1) {
        return Some(right - 1);
    }

    if !is_ok(left) {
        return None;
    }

    let boundary = binary_search_with_i128_for_inc(rng, |mid| !is_ok(mid)).unwrap() - 1;
    Some(boundary)
}

pub fn binary_search_with_i128<R, F>(rng: R, is_ok: F, dec: bool) -> Option<i128>
where
    R: RangeBounds<i128>,
    F: Fn(i128) -> bool,
{
    if dec {
        binary_search_with_i128_for_dec(rng, is_ok)
    } else {
        binary_search_with_i128_for_inc(rng, is_ok)
    }
}

pub trait BinarySearchWithI128: Sized + RangeBounds<i128> {
    /// Returns the smallest integer `x` in the range for which `is_ok(x) = true`.
    /// If no such integer exists, returns None.
    ///
    /// # Arguments
    ///
    /// * `is_ok` - Monotonic function. Weakly monotonicity is also included.
    /// * `dec` - Indicates that `is_ok` is a monotonically decreasing function if true,
    /// or a monotonically increasing function if false.
    fn binary_search<F>(self, is_ok: F, dec: bool) -> Option<i128>
    where
        F: Fn(i128) -> bool,
    {
        binary_search_with_i128(self, is_ok, dec)
    }
}

impl BinarySearchWithI128 for RangeFull {}

impl BinarySearchWithI128 for RangeTo<i128> {}

impl BinarySearchWithI128 for RangeToInclusive<i128> {}

impl BinarySearchWithI128 for RangeFrom<i128> {}

impl BinarySearchWithI128 for Range<i128> {}

impl BinarySearchWithI128 for RangeInclusive<i128> {}

// Binary search with `isize`

fn binary_search_with_isize_for_inc<R, F>(rng: R, is_ok: F) -> Option<isize>
where
    R: RangeBounds<isize>,
    F: Fn(isize) -> bool,
{
    let mut left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start + 1,
        std::ops::Bound::Unbounded => std::isize::MIN,
    };

    let right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end + 1,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::isize::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    if is_ok(left) {
        return Some(left);
    }

    if !is_ok(right - 1) {
        return None;
    }

    let mut size = right - left;

    while size > 1 {
        let half = size / 2;
        let mid = left + half;

        if !is_ok(mid) {
            left = mid;
        }
        size -= half;
    }

    let boundary = if is_ok(left) { left } else { left + 1 };
    Some(boundary)
}

fn binary_search_with_isize_for_dec<R, F>(rng: R, is_ok: F) -> Option<isize>
where
    R: RangeBounds<isize>,
    F: Fn(isize) -> bool,
{
    let left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start + 1,
        std::ops::Bound::Unbounded => std::isize::MIN,
    };

    let right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end + 1,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::isize::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    if is_ok(right - 1) {
        return Some(right - 1);
    }

    if !is_ok(left) {
        return None;
    }

    let boundary = binary_search_with_isize_for_inc(rng, |mid| !is_ok(mid)).unwrap() - 1;
    Some(boundary)
}

pub fn binary_search_with_isize<R, F>(rng: R, is_ok: F, dec: bool) -> Option<isize>
where
    R: RangeBounds<isize>,
    F: Fn(isize) -> bool,
{
    if dec {
        binary_search_with_isize_for_dec(rng, is_ok)
    } else {
        binary_search_with_isize_for_inc(rng, is_ok)
    }
}

pub trait BinarySearchWithIsize: Sized + RangeBounds<isize> {
    /// Returns the smallest integer `x` in the range for which `is_ok(x) = true`.
    /// If no such integer exists, returns None.
    ///
    /// # Arguments
    ///
    /// * `is_ok` - Monotonic function. Weakly monotonicity is also included.
    /// * `dec` - Indicates that `is_ok` is a monotonically decreasing function if true,
    /// or a monotonically increasing function if false.
    fn binary_search<F>(self, is_ok: F, dec: bool) -> Option<isize>
    where
        F: Fn(isize) -> bool,
    {
        binary_search_with_isize(self, is_ok, dec)
    }
}

impl BinarySearchWithIsize for RangeFull {}

impl BinarySearchWithIsize for RangeTo<isize> {}

impl BinarySearchWithIsize for RangeToInclusive<isize> {}

impl BinarySearchWithIsize for RangeFrom<isize> {}

impl BinarySearchWithIsize for Range<isize> {}

impl BinarySearchWithIsize for RangeInclusive<isize> {}

// Binary search with `u32`

fn binary_search_with_u32_for_inc<R, F>(rng: R, is_ok: F) -> Option<u32>
where
    R: RangeBounds<u32>,
    F: Fn(u32) -> bool,
{
    let mut left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start + 1,
        std::ops::Bound::Unbounded => std::u32::MIN,
    };

    let right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end + 1,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::u32::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    if is_ok(left) {
        return Some(left);
    }

    if !is_ok(right - 1) {
        return None;
    }

    let mut size = right - left;

    while size > 1 {
        let half = size / 2;
        let mid = left + half;

        if !is_ok(mid) {
            left = mid;
        }
        size -= half;
    }

    let boundary = if is_ok(left) { left } else { left + 1 };
    Some(boundary)
}

fn binary_search_with_u32_for_dec<R, F>(rng: R, is_ok: F) -> Option<u32>
where
    R: RangeBounds<u32>,
    F: Fn(u32) -> bool,
{
    let left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start + 1,
        std::ops::Bound::Unbounded => std::u32::MIN,
    };

    let right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end + 1,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::u32::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    if is_ok(right - 1) {
        return Some(right - 1);
    }

    if !is_ok(left) {
        return None;
    }

    let boundary = binary_search_with_u32_for_inc(rng, |mid| !is_ok(mid)).unwrap() - 1;
    Some(boundary)
}

pub fn binary_search_with_u32<R, F>(rng: R, is_ok: F, dec: bool) -> Option<u32>
where
    R: RangeBounds<u32>,
    F: Fn(u32) -> bool,
{
    if dec {
        binary_search_with_u32_for_dec(rng, is_ok)
    } else {
        binary_search_with_u32_for_inc(rng, is_ok)
    }
}

pub trait BinarySearchWithU32: Sized + RangeBounds<u32> {
    /// Returns the smallest integer `x` in the range for which `is_ok(x) = true`.
    /// If no such integer exists, returns None.
    ///
    /// # Arguments
    ///
    /// * `is_ok` - Monotonic function. Weakly monotonicity is also included.
    /// * `dec` - Indicates that `is_ok` is a monotonically decreasing function if true,
    /// or a monotonically increasing function if false.
    fn binary_search<F>(self, is_ok: F, dec: bool) -> Option<u32>
    where
        F: Fn(u32) -> bool,
    {
        binary_search_with_u32(self, is_ok, dec)
    }
}

impl BinarySearchWithU32 for RangeFull {}

impl BinarySearchWithU32 for RangeTo<u32> {}

impl BinarySearchWithU32 for RangeToInclusive<u32> {}

impl BinarySearchWithU32 for RangeFrom<u32> {}

impl BinarySearchWithU32 for Range<u32> {}

impl BinarySearchWithU32 for RangeInclusive<u32> {}

// Binary search with `u64`

fn binary_search_with_u64_for_inc<R, F>(rng: R, is_ok: F) -> Option<u64>
where
    R: RangeBounds<u64>,
    F: Fn(u64) -> bool,
{
    let mut left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start + 1,
        std::ops::Bound::Unbounded => std::u64::MIN,
    };

    let right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end + 1,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::u64::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    if is_ok(left) {
        return Some(left);
    }

    if !is_ok(right - 1) {
        return None;
    }

    let mut size = right - left;

    while size > 1 {
        let half = size / 2;
        let mid = left + half;

        if !is_ok(mid) {
            left = mid;
        }
        size -= half;
    }

    let boundary = if is_ok(left) { left } else { left + 1 };
    Some(boundary)
}

fn binary_search_with_u64_for_dec<R, F>(rng: R, is_ok: F) -> Option<u64>
where
    R: RangeBounds<u64>,
    F: Fn(u64) -> bool,
{
    let left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start + 1,
        std::ops::Bound::Unbounded => std::u64::MIN,
    };

    let right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end + 1,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::u64::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    if is_ok(right - 1) {
        return Some(right - 1);
    }

    if !is_ok(left) {
        return None;
    }

    let boundary = binary_search_with_u64_for_inc(rng, |mid| !is_ok(mid)).unwrap() - 1;
    Some(boundary)
}

pub fn binary_search_with_u64<R, F>(rng: R, is_ok: F, dec: bool) -> Option<u64>
where
    R: RangeBounds<u64>,
    F: Fn(u64) -> bool,
{
    if dec {
        binary_search_with_u64_for_dec(rng, is_ok)
    } else {
        binary_search_with_u64_for_inc(rng, is_ok)
    }
}

pub trait BinarySearchWithU64: Sized + RangeBounds<u64> {
    /// Returns the smallest integer `x` in the range for which `is_ok(x) = true`.
    /// If no such integer exists, returns None.
    ///
    /// # Arguments
    ///
    /// * `is_ok` - Monotonic function. Weakly monotonicity is also included.
    /// * `dec` - Indicates that `is_ok` is a monotonically decreasing function if true,
    /// or a monotonically increasing function if false.
    fn binary_search<F>(self, is_ok: F, dec: bool) -> Option<u64>
    where
        F: Fn(u64) -> bool,
    {
        binary_search_with_u64(self, is_ok, dec)
    }
}

impl BinarySearchWithU64 for RangeFull {}

impl BinarySearchWithU64 for RangeTo<u64> {}

impl BinarySearchWithU64 for RangeToInclusive<u64> {}

impl BinarySearchWithU64 for RangeFrom<u64> {}

impl BinarySearchWithU64 for Range<u64> {}

impl BinarySearchWithU64 for RangeInclusive<u64> {}

// Binary search with `u128`

fn binary_search_with_u128_for_inc<R, F>(rng: R, is_ok: F) -> Option<u128>
where
    R: RangeBounds<u128>,
    F: Fn(u128) -> bool,
{
    let mut left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start + 1,
        std::ops::Bound::Unbounded => std::u128::MIN,
    };

    let right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end + 1,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::u128::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    if is_ok(left) {
        return Some(left);
    }

    if !is_ok(right - 1) {
        return None;
    }

    let mut size = right - left;

    while size > 1 {
        let half = size / 2;
        let mid = left + half;

        if !is_ok(mid) {
            left = mid;
        }
        size -= half;
    }

    let boundary = if is_ok(left) { left } else { left + 1 };
    Some(boundary)
}

fn binary_search_with_u128_for_dec<R, F>(rng: R, is_ok: F) -> Option<u128>
where
    R: RangeBounds<u128>,
    F: Fn(u128) -> bool,
{
    let left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start + 1,
        std::ops::Bound::Unbounded => std::u128::MIN,
    };

    let right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end + 1,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::u128::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    if is_ok(right - 1) {
        return Some(right - 1);
    }

    if !is_ok(left) {
        return None;
    }

    let boundary = binary_search_with_u128_for_inc(rng, |mid| !is_ok(mid)).unwrap() - 1;
    Some(boundary)
}

pub fn binary_search_with_u128<R, F>(rng: R, is_ok: F, dec: bool) -> Option<u128>
where
    R: RangeBounds<u128>,
    F: Fn(u128) -> bool,
{
    if dec {
        binary_search_with_u128_for_dec(rng, is_ok)
    } else {
        binary_search_with_u128_for_inc(rng, is_ok)
    }
}

pub trait BinarySearchWithU128: Sized + RangeBounds<u128> {
    /// Returns the smallest integer `x` in the range for which `is_ok(x) = true`.
    /// If no such integer exists, returns None.
    ///
    /// # Arguments
    ///
    /// * `is_ok` - Monotonic function. Weakly monotonicity is also included.
    /// * `dec` - Indicates that `is_ok` is a monotonically decreasing function if true,
    /// or a monotonically increasing function if false.
    fn binary_search<F>(self, is_ok: F, dec: bool) -> Option<u128>
    where
        F: Fn(u128) -> bool,
    {
        binary_search_with_u128(self, is_ok, dec)
    }
}

impl BinarySearchWithU128 for RangeFull {}

impl BinarySearchWithU128 for RangeTo<u128> {}

impl BinarySearchWithU128 for RangeToInclusive<u128> {}

impl BinarySearchWithU128 for RangeFrom<u128> {}

impl BinarySearchWithU128 for Range<u128> {}

impl BinarySearchWithU128 for RangeInclusive<u128> {}

// Binary search with `usize`

fn binary_search_with_usize_for_inc<R, F>(rng: R, is_ok: F) -> Option<usize>
where
    R: RangeBounds<usize>,
    F: Fn(usize) -> bool,
{
    let mut left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start + 1,
        std::ops::Bound::Unbounded => std::usize::MIN,
    };

    let right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end + 1,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::usize::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    if is_ok(left) {
        return Some(left);
    }

    if !is_ok(right - 1) {
        return None;
    }

    let mut size = right - left;

    while size > 1 {
        let half = size / 2;
        let mid = left + half;

        if !is_ok(mid) {
            left = mid;
        }
        size -= half;
    }

    let boundary = if is_ok(left) { left } else { left + 1 };
    Some(boundary)
}

fn binary_search_with_usize_for_dec<R, F>(rng: R, is_ok: F) -> Option<usize>
where
    R: RangeBounds<usize>,
    F: Fn(usize) -> bool,
{
    let left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start + 1,
        std::ops::Bound::Unbounded => std::usize::MIN,
    };

    let right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end + 1,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::usize::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    if is_ok(right - 1) {
        return Some(right - 1);
    }

    if !is_ok(left) {
        return None;
    }

    let boundary = binary_search_with_usize_for_inc(rng, |mid| !is_ok(mid)).unwrap() - 1;
    Some(boundary)
}

pub fn binary_search_with_usize<R, F>(rng: R, is_ok: F, dec: bool) -> Option<usize>
where
    R: RangeBounds<usize>,
    F: Fn(usize) -> bool,
{
    if dec {
        binary_search_with_usize_for_dec(rng, is_ok)
    } else {
        binary_search_with_usize_for_inc(rng, is_ok)
    }
}

pub trait BinarySearchWithUsize: Sized + RangeBounds<usize> {
    /// Returns the smallest integer `x` in the range for which `is_ok(x) = true`.
    /// If no such integer exists, returns None.
    ///
    /// # Arguments
    ///
    /// * `is_ok` - Monotonic function. Weakly monotonicity is also included.
    /// * `dec` - Indicates that `is_ok` is a monotonically decreasing function if true,
    /// or a monotonically increasing function if false.
    fn binary_search<F>(self, is_ok: F, dec: bool) -> Option<usize>
    where
        F: Fn(usize) -> bool,
    {
        binary_search_with_usize(self, is_ok, dec)
    }
}

impl BinarySearchWithUsize for RangeFull {}

impl BinarySearchWithUsize for RangeTo<usize> {}

impl BinarySearchWithUsize for RangeToInclusive<usize> {}

impl BinarySearchWithUsize for RangeFrom<usize> {}

impl BinarySearchWithUsize for Range<usize> {}

impl BinarySearchWithUsize for RangeInclusive<usize> {}

// Binary search with `f32`

fn binary_search_with_f32_for_inc<R, F>(rng: R, is_ok: F, eps: f32) -> Option<f32>
where
    R: RangeBounds<f32>,
    F: Fn(f32) -> bool,
{
    let mut left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start,
        std::ops::Bound::Unbounded => std::f32::MIN,
    };

    let right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::f32::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    assert!(
        eps >= 0.0,
        "Allowable margin of error must be a positive number."
    );

    if is_ok(left) {
        return Some(left);
    }

    if !is_ok(right) {
        return None;
    }

    let mut size = right - left;

    while size > eps {
        let half = size / 2.0;
        let mid = left + half;

        if !is_ok(mid) {
            left = mid;
        }
        size -= half;
    }

    Some(right)
}

fn binary_search_with_f32_for_dec<R, F>(rng: R, is_ok: F, eps: f32) -> Option<f32>
where
    R: RangeBounds<f32>,
    F: Fn(f32) -> bool,
{
    let left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start,
        std::ops::Bound::Unbounded => std::f32::MIN,
    };

    let mut right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::f32::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    assert!(
        eps >= 0.0,
        "Allowable margin of error must be a positive number."
    );

    if is_ok(right) {
        return Some(right);
    }

    if !is_ok(left) {
        return None;
    }

    let mut size = right - left;

    while size > eps {
        let half = size / 2.0;
        let mid = right - half;

        if !is_ok(mid) {
            right = mid;
        }
        size -= half;
    }

    Some(left)
}

pub fn binary_search_with_f32<R, F>(rng: R, is_ok: F, eps: f32, dec: bool) -> Option<f32>
where
    R: RangeBounds<f32>,
    F: Fn(f32) -> bool,
{
    if dec {
        binary_search_with_f32_for_dec(rng, is_ok, eps)
    } else {
        binary_search_with_f32_for_inc(rng, is_ok, eps)
    }
}

pub trait BinarySearchWithF32: Sized + RangeBounds<f32> {
    /// Returns the smallest real number `x` in the range for which `is_ok(x) = true`.
    /// However, an error with the true value within `eps` is allowed.
    /// If no such real number exists, returns None.
    ///
    /// # Arguments
    ///
    /// * `is_ok` - Monotonic function. Weakly monotonicity is also included.
    /// * `eps` - Allowable margin of error
    /// * `dec` - Indicates that `is_ok` is a monotonically decreasing function if true,
    /// or a monotonically increasing function if false.
    fn binary_search<F>(self, is_ok: F, eps: f32, dec: bool) -> Option<f32>
    where
        F: Fn(f32) -> bool,
    {
        binary_search_with_f32(self, is_ok, eps, dec)
    }
}

impl BinarySearchWithF32 for RangeFull {}

impl BinarySearchWithF32 for RangeTo<f32> {}

impl BinarySearchWithF32 for RangeToInclusive<f32> {}

impl BinarySearchWithF32 for RangeFrom<f32> {}

impl BinarySearchWithF32 for Range<f32> {}

impl BinarySearchWithF32 for RangeInclusive<f32> {}

// Binary search with `f64`

fn binary_search_with_f64_for_inc<R, F>(rng: R, is_ok: F, eps: f64) -> Option<f64>
where
    R: RangeBounds<f64>,
    F: Fn(f64) -> bool,
{
    let mut left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start,
        std::ops::Bound::Unbounded => std::f64::MIN,
    };

    let right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::f64::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    assert!(
        eps >= 0.0,
        "Allowable margin of error must be a positive number."
    );

    if is_ok(left) {
        return Some(left);
    }

    if !is_ok(right) {
        return None;
    }

    let mut size = right - left;

    while size > eps {
        let half = size / 2.0;
        let mid = left + half;

        if !is_ok(mid) {
            left = mid;
        }
        size -= half;
    }

    Some(right)
}

fn binary_search_with_f64_for_dec<R, F>(rng: R, is_ok: F, eps: f64) -> Option<f64>
where
    R: RangeBounds<f64>,
    F: Fn(f64) -> bool,
{
    let left = match rng.start_bound() {
        std::ops::Bound::Included(&start) => start,
        std::ops::Bound::Excluded(&start) => start,
        std::ops::Bound::Unbounded => std::f64::MIN,
    };

    let mut right = match rng.end_bound() {
        std::ops::Bound::Included(&end) => end,
        std::ops::Bound::Excluded(&end) => end,
        std::ops::Bound::Unbounded => std::f64::MAX,
    };

    assert!(left < right, "The interval represented by `rng` is empty.");

    assert!(
        eps >= 0.0,
        "Allowable margin of error must be a positive number."
    );

    if is_ok(right) {
        return Some(right);
    }

    if !is_ok(left) {
        return None;
    }

    let mut size = right - left;

    while size > eps {
        let half = size / 2.0;
        let mid = right - half;

        if !is_ok(mid) {
            right = mid;
        }
        size -= half;
    }

    Some(left)
}

pub fn binary_search_with_f64<R, F>(rng: R, is_ok: F, eps: f64, dec: bool) -> Option<f64>
where
    R: RangeBounds<f64>,
    F: Fn(f64) -> bool,
{
    if dec {
        binary_search_with_f64_for_dec(rng, is_ok, eps)
    } else {
        binary_search_with_f64_for_inc(rng, is_ok, eps)
    }
}

pub trait BinarySearchWithF64: Sized + RangeBounds<f64> {
    /// Returns the smallest real number `x` in the range for which `is_ok(x) = true`.
    /// However, an error with the true value within `eps` is allowed.
    /// If no such real number exists, returns None.
    ///
    /// # Arguments
    ///
    /// * `is_ok` - Monotonic function. Weakly monotonicity is also included.
    /// * `eps` - Allowable margin of error
    /// * `dec` - Indicates that `is_ok` is a monotonically decreasing function if true,
    /// or a monotonically increasing function if false.
    fn binary_search<F>(self, is_ok: F, eps: f64, dec: bool) -> Option<f64>
    where
        F: Fn(f64) -> bool,
    {
        binary_search_with_f64(self, is_ok, eps, dec)
    }
}

impl BinarySearchWithF64 for RangeFull {}

impl BinarySearchWithF64 for RangeTo<f64> {}

impl BinarySearchWithF64 for RangeToInclusive<f64> {}

impl BinarySearchWithF64 for RangeFrom<f64> {}

impl BinarySearchWithF64 for Range<f64> {}

impl BinarySearchWithF64 for RangeInclusive<f64> {}
