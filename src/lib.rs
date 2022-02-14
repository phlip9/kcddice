//! # kcddice
//!
//! A simple utility for optimally playing the Kingdom Come: Deliverance dice
//! game : )
//!
//! ## Rules
//!
//! https://kingdom-come-deliverance.fandom.com/wiki/Dice
//!
//! ## Explanation
//!
//! More specifically, this tool takes as input the currently rolled dice on the
//! board and outputs all possible actions along with their expected values and
//! bust probabilities.
//!
//! The expected value in this case is the expectation of the turn total assuming
//! the player plays to maximize their expected value.
//!
//! The bust probability is the probability that no scoring dice are rolled in
//! the next turn, assuming the player choose the given action.

#[macro_use]
mod macros;

pub mod cli;
mod dice;
mod multiset;
mod parse;
mod search;

use std::cmp;

pub(crate) const DEFAULT_TARGET_SCORE: u16 = 4000;

///////////////////
// Combinatorics //
///////////////////

/// The number of factorials to precompute in our static lookup table. Note this
/// number is chosen so as not to overflow a u32.
pub(crate) const NUM_FACTORIALS: usize = 13;

/// A precomputed lookup table of factorials from `0 <= n < NUM_FACTORIALS`.
/// `FACTORIAL_LT[n] = n!`.
const FACTORIAL_LT: [u32; NUM_FACTORIALS] = precompute_factorials();

const fn precompute_factorials() -> [u32; NUM_FACTORIALS] {
    let mut factorials: [u32; NUM_FACTORIALS] = [1; NUM_FACTORIALS];

    // need ghetto for-loop in const fn...
    let mut idx = 1;
    loop {
        if idx >= NUM_FACTORIALS {
            break;
        }
        factorials[idx] = (idx as u32) * factorials[idx - 1];
        idx += 1;
    }

    factorials
}

pub(crate) const fn factorial(n: u32) -> u32 {
    FACTORIAL_LT[n as usize]
}

/// count `n choose k` without replacement.
pub(crate) const fn num_combinations(n: u32, k: u32) -> u32 {
    factorial(n) / (factorial(k) * factorial(n - k))
}

/// count `n choose k` with replacement. also known as `n multichoose k`.
#[inline]
pub(crate) const fn num_multisets(n: u32, k: u32) -> u32 {
    num_combinations(n + k - 1, k)
}

////////////////////////////
// Unstable std functions //
////////////////////////////

/// A total ordering on f64's. Needed until `f64::total_cmp` is stabilized.
/// See: (https://doc.rust-lang.org/stable/std/primitive.f64.html#method.total_cmp)
#[inline]
pub(crate) fn total_cmp_f64(a: &f64, b: &f64) -> cmp::Ordering {
    let mut left = a.to_bits() as i64;
    let mut right = b.to_bits() as i64;

    left ^= (((left >> 63) as u64) >> 1) as i64;
    right ^= (((right >> 63) as u64) >> 1) as i64;

    left.cmp(&right)
}

/// Returns `true` if the iterator `iter` is sorted, according to the comparator
/// function `compare`, i.e., `x_1 <= x2 <= ... <= x_n`.
// TODO(philiphayes): use `std::slice::is_sorted_by` when it stabilizes.
pub(crate) fn is_sorted_by<T, F>(mut iter: impl Iterator<Item = T>, mut compare: F) -> bool
where
    F: FnMut(&T, &T) -> Option<cmp::Ordering>,
{
    let mut prev = match iter.next() {
        Some(first) => first,
        None => return true,
    };

    for next in iter {
        if let Some(cmp::Ordering::Greater) | None = compare(&prev, &next) {
            return false;
        }
        prev = next;
    }

    true
}

/// Returns `true` if the iterator `iter` is totally ordered, according to the
/// comparator function `compare`, i.e., `x_1 < x2 < ... < x_n`.
pub(crate) fn is_total_order_by<T, F>(mut iter: impl Iterator<Item = T>, mut compare: F) -> bool
where
    F: FnMut(&T, &T) -> Option<cmp::Ordering>,
{
    let mut prev = match iter.next() {
        Some(first) => first,
        None => return true,
    };

    for next in iter {
        if let Some(cmp::Ordering::Greater) | Some(cmp::Ordering::Equal) | None =
            compare(&prev, &next)
        {
            return false;
        }
        prev = next;
    }

    true
}

pub(crate) fn is_partitioned<T>(
    mut iter: impl Iterator<Item = T>,
    mut pred: impl FnMut(T) -> bool,
) -> bool {
    iter.all(&mut pred) || !iter.any(pred)
}

///////////////
// Bit Hacks //
///////////////

/// Returns `true` if `x` has _any_ nibbles `nb` in the range `m < nb < n`. The
/// nibbles considered are selected according to `mask`, where `mask` has a `1`
/// in each selectable nibble.
///
/// For example, calling this with `mask = 0x0010_1101` will only look at the
/// 1st, 3rd, 4th, and 6th nibbles. Calling with `mask = 0x1111_1111` will look
/// at all nibbles. Technically you can also look at "unaligned" nibbles too :)
///
/// See [`u32_any_nibs_between_ref`] for a human-readable implementation.
#[inline]
pub(crate) fn u32_any_nibs_between(x: u32, mask: u32, m: u32, n: u32) -> bool {
    debug_assert!((0..=7).contains(&m));
    debug_assert!((0..=8).contains(&n));

    let a = mask * 7;
    let b = mask * 8;
    let w = mask * (7 + n);
    let t = mask * (7 - m);

    let u = x & a;
    let z = (w - u) & (!x) & (u + t) & b;

    z != 0
}

/// Sum all nibbles in `x`.
#[inline]
pub(crate) fn u32_sum_all_nibs(x: u32) -> u32 {
    // a mask that selects the lo nibble in each byte.
    const NIBS_0246: u32 = 0x0f0f_0f0f;

    // horizontal sum hi and lo nibbles in each byte, placing in the lo nibble.
    let y = (x & NIBS_0246) + ((x >> 4) & NIBS_0246);

    // if y = [y0, y1, y2, y3] bytes and each byte b is in the range 0 <= b < 64,
    // then multiplying by 0x0101_0101 will yield
    // z = [y0, y0 + y1, y0 + y1 + y2, y0 + y1 + y2 + y3] without any overflows.
    //
    // since each byte in y is the sum of two nibbles nb where 0 <= nb < 16, it
    // follows that 0 <= b = nb_lo + nb_hi < 32 < 64, so we won't have any
    // overflows.
    let z = y.wrapping_mul(0x0101_0101);

    // select the last byte in z, which contains our desired sum:
    // z3 = y0 + y1 + y2 + y3
    //    = (nb0 + nb1) + (nb2 + nb3) + (nb4 + nb5) + (nb6 + nb7)
    z >> 24
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test {
    use super::*;
    use proptest::prelude::*;

    fn factorial_ref(n: u32) -> u32 {
        (1..=n).product()
    }

    #[test]
    fn test_factorial_lt() {
        for n in 0..NUM_FACTORIALS as u32 {
            assert_eq!(factorial_ref(n), factorial(n));
        }
    }

    /// spread `x` into an array of nibbles.
    fn u32_into_nib_le(x: u32) -> [u8; 8] {
        let [b0, b1, b2, b3] = x.to_le_bytes();
        [
            b0 & 0x0f,
            (b0 >> 4) & 0x0f,
            b1 & 0x0f,
            (b1 >> 4) & 0x0f,
            b2 & 0x0f,
            (b2 >> 4) & 0x0f,
            b3 & 0x0f,
            (b3 >> 4) & 0x0f,
        ]
    }

    /// reference implementation of [`u32_any_nibs_between`]
    fn u32_any_nibs_between_ref(x: u32, mask: u32, m: u32, n: u32) -> bool {
        assert!((0..=7).contains(&m));
        assert!((0..=8).contains(&n));

        let m = m as u8;
        let n = n as u8;

        u32_into_nib_le(x)
            .into_iter()
            .zip(u32_into_nib_le(mask).into_iter())
            .any(|(nb_x, nb_mask)| (nb_mask == 0x1) && (m < nb_x) && (nb_x < n))
    }

    // reference implementation of [`std::arch::x86::bmi::_pdep32`] so I don't
    // need to gate tests.
    fn pdep32_fallback(src: u32, mut mask: u32) -> u32 {
        // iterate bit from lsb -> msb
        let mut bit = 1;
        let mut r = 0;
        while mask != 0 {
            if src & bit != 0 {
                // add the lsb from mask to r
                r |= mask & mask.wrapping_neg();
            }
            // clear lsb
            mask &= mask - 1;
            bit <<= 1;
        }
        r
    }

    // Efficiently generate combinations of mask bits that match a "`meta_mask`"
    // (by match I mean `meta_mask & mask == mask`).
    fn arb_mask(meta_mask: u32) -> impl Strategy<Value = u32> {
        let nmasks: u32 = 1 << meta_mask.count_ones();

        (0..=nmasks).prop_map(move |mask_idx| pdep32_fallback(mask_idx, meta_mask))
    }

    #[test]
    fn test_u32_any_nibs_between() {
        assert!(u32_any_nibs_between(0x1234_5678, 0x1010_1010, 0, 3));
        assert!(!u32_any_nibs_between(0x1234_5678, 0x0011_1011, 0, 3));

        let cfg = ProptestConfig::with_cases(5000);
        proptest!(cfg, |(x in any::<u32>(), mask in arb_mask(0x1111_1111), m in (0u32..=7), n in (0u32..=8))| {
            assert_eq!(u32_any_nibs_between_ref(x, mask, m, n), u32_any_nibs_between(x, mask, m, n));
        });
    }

    fn u32_sum_all_nibs_ref(x: u32) -> u32 {
        u32_into_nib_le(x).into_iter().map(|x| x as u32).sum()
    }

    #[test]
    fn test_u32_sum_all_nibs() {
        let cfg = ProptestConfig::with_cases(5000);
        proptest!(cfg, |(x in any::<u32>())| {
            prop_assert_eq!(u32_sum_all_nibs_ref(x), u32_sum_all_nibs(x));
 
        });
    }
}
