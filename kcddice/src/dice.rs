use crate::{
    factorial, is_partitioned, is_sorted_by, is_total_order_by, multiset::MultisetU4x8,
    num_multisets, u32_any_nibs_between, u64_leading_byte_idx_lt, u64_leading_zero_bytes,
    u64_trailing_byte_idx_lt,
};
use approx::relative_eq;
use claim::{debug_assert_le, debug_assert_lt};
#[cfg(test)]
use proptest::{
    arbitrary::Arbitrary,
    strategy::{BoxedStrategy, Strategy},
};
use rand::{
    distributions::{Distribution, Open01},
    Rng,
};
use serde::{Deserialize, Serialize};
use std::{
    cmp, fmt,
    hash::{Hash, Hasher},
    iter::FusedIterator,
    ops::Range,
    str::FromStr,
};

// TODO(philiphayes): implement jokers/devils
// const FACE_JOKER: u8 = 0;

/// A DieKind's face distribution, as a cumulative distribution function (CDF)
/// for more efficient sampling.
#[derive(Clone)]
pub struct DieDistr([f64; 8]);

impl DieDistr {
    const fn new(cdf: [f64; 6]) -> Self {
        Self([0.0, cdf[0], cdf[1], cdf[2], cdf[3], cdf[4], cdf[5], 0.0])
    }

    /// Would use this instead of hacky macro below, but can't do float arithmetic
    /// in const functions...
    #[cfg(test)]
    fn from_pmf(p: [f64; 6]) -> Self {
        Self::new([
            p[0],
            p[0] + p[1],
            p[0] + p[1] + p[2],
            p[0] + p[1] + p[2] + p[3],
            p[0] + p[1] + p[2] + p[3] + p[4],
            1.0,
        ])
    }

    #[cfg(test)]
    pub(crate) fn into_pmf(self) -> [f64; 6] {
        [
            self.p_face(1),
            self.p_face(2),
            self.p_face(3),
            self.p_face(4),
            self.p_face(5),
            self.p_face(6),
        ]
    }

    // const fn new_joker(cdf: [f64; 6]) -> Self {
    //     Self([
    //         cdf[0], 0.0, cdf[1], cdf[2], cdf[3], cdf[4], cdf[5], 0.0,
    //     ])
    // }

    #[inline]
    fn p_face(&self, face: u8) -> f64 {
        self.0[face as usize] - self.0[(face - 1) as usize]
    }

    /// convert a standard sample r ∈ (0, 1) to a die face, according to this CDF.
    #[inline]
    fn sample_to_face(&self, r: f64) -> u8 {
        // just inline a binary search lol
        if r > self.0[3] {
            if r > self.0[5] {
                6
            } else if r > self.0[4] {
                5
            } else {
                4
            }
        } else if r > self.0[2] {
            3
        } else if r > self.0[1] {
            2
        } else {
            1
        }
    }
}

impl Distribution<u8> for DieDistr {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> u8 {
        // sample r ∈ (0, 1)
        let r = Open01.sample(rng);
        self.sample_to_face(r)
    }
}

// very annoying hack b/c we can't do float arithmetic in const fn's...
macro_rules! die_distr {
    ($p_unif:expr) => {
        die_distr![$p_unif, $p_unif, $p_unif, $p_unif, $p_unif, $p_unif]
    };
    // pi := Pr[Roll_{die_kind} = i]
    ($p1:expr, $p2:expr, $p3:expr, $p4:expr, $p5:expr, $p6:expr $(,)?) => {
        // cumsum the pmf into a cdf
        DieDistr::new([
            $p1,
            $p1 + $p2,
            $p1 + $p2 + $p3,
            $p1 + $p2 + $p3 + $p4,
            $p1 + $p2 + $p3 + $p4 + $p5,
            1.0,
        ])
    };
}

impl fmt::Debug for DieDistr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pmf = [
            self.p_face(1),
            self.p_face(2),
            self.p_face(3),
            self.p_face(4),
            self.p_face(5),
            self.p_face(6),
        ];
        write!(f, "{:?}", pmf)
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[allow(clippy::upper_case_acronyms)]
pub enum DieKind {
    SENTINEL = 0,
    Standard,
    Alfonse,
    Ambrose,
    Biased,
    Even,
    HeavenlyKingdomDie,
    HenrysBeta,
    HolyTrinity,
    LuCiFer,
    Lucky,
    LuckyPlaying,
    Misfortune,
    OddDie,
    Shrinking,
    Strip,
    TheCommonest,
    Unpopular,
}

impl Default for DieKind {
    #[inline]
    fn default() -> Self {
        Self::SENTINEL
    }
}

impl DieKind {
    pub const fn all() -> &'static [DieKind; 17] {
        &[
            Self::Standard,
            Self::Alfonse,
            Self::Ambrose,
            Self::Biased,
            Self::Even,
            Self::HeavenlyKingdomDie,
            Self::HenrysBeta,
            Self::HolyTrinity,
            Self::LuCiFer,
            Self::Lucky,
            Self::LuckyPlaying,
            Self::Misfortune,
            Self::OddDie,
            Self::Shrinking,
            Self::Strip,
            Self::TheCommonest,
            Self::Unpopular,
        ]
    }

    fn is_sentinel(self) -> bool {
        self == Self::SENTINEL
    }

    pub const fn from_u8(idx: u8) -> Self {
        match idx {
            0 => panic!("SENTINEL"),
            1 => Self::Standard,
            2 => Self::Alfonse,
            3 => Self::Ambrose,
            4 => Self::Biased,
            5 => Self::Even,
            6 => Self::HeavenlyKingdomDie,
            7 => Self::HenrysBeta,
            8 => Self::HolyTrinity,
            9 => Self::LuCiFer,
            10 => Self::Lucky,
            11 => Self::LuckyPlaying,
            12 => Self::Misfortune,
            13 => Self::OddDie,
            14 => Self::Shrinking,
            15 => Self::Strip,
            16 => Self::TheCommonest,
            17 => Self::Unpopular,
            _ => panic!("UNKNOWN"),
        }
    }

    pub fn from_memnonic(s: &str) -> Option<Self> {
        let kind = match s {
            "" | "s" => Self::Standard,
            "a" => Self::Alfonse,
            "am" => Self::Ambrose,
            "b" => Self::Biased,
            "e" => Self::Even,
            "hb" => Self::HenrysBeta,
            "hk" => Self::HeavenlyKingdomDie,
            "ht" => Self::HolyTrinity,
            "l" => Self::LuCiFer,
            "lk" => Self::Lucky,
            "lp" => Self::LuckyPlaying,
            "m" => Self::Misfortune,
            "o" => Self::OddDie,
            "sh" => Self::Shrinking,
            "sp" => Self::Strip,
            "tc" => Self::TheCommonest,
            "u" => Self::Unpopular,
            _ => return None,
        };
        Some(kind)
    }

    pub fn as_memnonic(self) -> &'static str {
        match self {
            Self::SENTINEL => panic!("SENTINEL"),
            Self::Standard => "",
            Self::Alfonse => "a",
            Self::Ambrose => "am",
            Self::Biased => "b",
            Self::Even => "e",
            Self::HenrysBeta => "hb",
            Self::HeavenlyKingdomDie => "hk",
            Self::HolyTrinity => "ht",
            Self::LuCiFer => "l",
            Self::Lucky => "lk",
            Self::LuckyPlaying => "lp",
            Self::Misfortune => "m",
            Self::OddDie => "o",
            Self::Shrinking => "sh",
            Self::Strip => "sp",
            Self::TheCommonest => "tc",
            Self::Unpopular => "u",
        }
    }

    pub fn as_human_readable(self) -> &'static str {
        match self {
            Self::SENTINEL => panic!("SENTINEL"),
            Self::Standard => "Standard",
            Self::Alfonse => "Alfonse's",
            Self::Ambrose => "Ambrose's",
            Self::Biased => "Biased",
            Self::Even => "Even Number",
            Self::HenrysBeta => "Henry's Beta",
            Self::HeavenlyKingdomDie => "Heavenly Kingdom",
            Self::HolyTrinity => "Holy Trinity",
            Self::LuCiFer => "Lu/Ci/Fer",
            Self::Lucky => "Lucky",
            Self::LuckyPlaying => "Lucky Playing",
            Self::Misfortune => "Die of Misfortune",
            Self::OddDie => "Odd",
            Self::Shrinking => "Shrinking",
            Self::Strip => "Strip",
            Self::Unpopular => "Unpopular",
            Self::TheCommonest => "The Commonest",
        }
    }

    pub(crate) fn die_distr(self) -> DieDistr {
        // not all dice probabilities on the wiki add up to 1.0 exactly, so
        // without better info, I'm going to slightly bias the dice against 1
        // and 5 (the best rolls) to make everything add up : )

        match self {
            Self::SENTINEL => panic!("SENTINEL"),
            Self::Standard | Self::TheCommonest => die_distr![1.0 / 6.0],
            Self::Alfonse => die_distr![0.385, 0.077, 0.077, 0.077, 0.154, 0.230],
            Self::Ambrose => die_distr![0.286, 0.214, 0.0715, 0.0715, 0.143, 0.214],
            Self::Biased => die_distr![
                3.0 / 12.0,
                4.0 / 12.0,
                1.0 / 12.0,
                1.0 / 12.0,
                2.0 / 12.0,
                1.0 / 12.0,
            ],
            Self::LuCiFer => die_distr![0.13, 0.13, 0.13, 0.13, 0.13, 0.35],
            Self::Even => die_distr![
                1.0 / 15.0,
                4.0 / 15.0,
                1.0 / 15.0,
                4.0 / 15.0,
                1.0 / 15.0,
                4.0 / 15.0,
            ],
            Self::HenrysBeta => die_distr![
                1.0 / 9.0,
                4.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
            ],
            Self::HeavenlyKingdomDie => die_distr![0.368, 0.105, 0.105, 0.105, 0.105, 0.212],
            Self::HolyTrinity => die_distr![0.182, 0.228, 0.455, 0.045, 0.045, 0.045],
            Self::Lucky => die_distr![0.273, 0.045, 0.091, 0.136, 0.182, 0.273],
            Self::LuckyPlaying => die_distr![
                6.0 / 18.0,
                0.0,
                1.0 / 18.0,
                1.0 / 18.0,
                6.0 / 18.0,
                4.0 / 18.0,
            ],
            Self::Misfortune => die_distr![0.046, 0.227, 0.227, 0.227, 0.227, 0.046],
            Self::OddDie => die_distr![
                4.0 / 15.0,
                1.0 / 15.0,
                4.0 / 15.0,
                1.0 / 15.0,
                4.0 / 15.0,
                1.0 / 15.0,
            ],
            Self::Shrinking => die_distr![
                2.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                1.0 / 9.0,
                3.0 / 9.0,
            ],
            Self::Strip => die_distr![
                4.0 / 16.0,
                2.0 / 16.0,
                2.0 / 16.0,
                2.0 / 16.0,
                3.0 / 16.0,
                3.0 / 16.0,
            ],
            Self::Unpopular => die_distr![
                1.0 / 11.0,
                3.0 / 11.0,
                2.0 / 11.0,
                2.0 / 11.0,
                2.0 / 11.0,
                1.0 / 11.0,
            ],
        }
    }
}

impl fmt::Display for DieKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_memnonic())
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DieKindTable([DieKind; 6]);

impl DieKindTable {
    pub fn from_die_kinds(kinds: &[DieKind]) -> Self {
        assert!(kinds.len() <= 6);

        let mut arr = [DieKind::SENTINEL; 6];

        arr[..kinds.len()].copy_from_slice(kinds);
        arr[..kinds.len()].sort_unstable();

        assert!(is_total_order_by(
            arr.into_iter().take_while(|kind| !kind.is_sentinel()),
            |k1, k2| Some(k1.cmp(k2))
        ));

        Self(arr)
    }

    pub fn all_standard() -> Self {
        Self::from_die_kinds(&[DieKind::Standard; 1])
    }

    #[inline]
    pub fn get_kind(&self, idx: u8) -> DieKind {
        self.0[idx as usize]
    }

    pub fn get_kind_idx(&self, kind: DieKind) -> Option<u8> {
        (0..6).find_map(|idx| {
            if self.get_kind(idx) == kind {
                Some(idx)
            } else {
                None
            }
        })
    }
}

impl Default for DieKindTable {
    fn default() -> Self {
        Self::all_standard()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DieKindCounts(MultisetU4x8);

impl DieKindCounts {
    #[inline]
    pub fn new() -> Self {
        Self(MultisetU4x8::new())
    }

    // pub fn all_standard(ndice: u8) -> Self {
    //     let mut arr = ArrayVec::new();
    //     arr.push((DieKind::Standard, ndice));
    //     Self(arr)
    // }

    pub fn from_dice_vec(dice: DiceVec) -> Self {
        let counts = Self(MultisetU4x8::from_iter_flat(
            dice.into_iter_no_sentinel().map(Die::kind_idx),
        ));
        debug_assert!(counts.invariant());
        counts
    }

    fn invariant(&self) -> bool {
        self.ndice() <= 6
    }

    #[inline]
    pub fn ndice(&self) -> u8 {
        self.0.len()
    }

    fn spread_face(&self, face: u8) -> DiceVec {
        let mut dice = DiceVec::new();
        let mut len = 0;
        for (kind_idx, nkind) in self.into_iter() {
            for i in 0..nkind {
                dice.set_die(len + i, Die::new(face, kind_idx));
            }
            len += nkind;
        }
        dice
    }

    #[inline]
    pub fn sub_counts(&mut self, other: Self) {
        debug_assert!(self.is_superset_of(&other));
        self.0 -= other.0;
        debug_assert!(self.invariant());
    }

    pub fn is_superset_of(&self, other: &Self) -> bool {
        self.0.is_superset_of(other.0)
    }

    fn num_multisets(&self, nfaces: u32) -> u32 {
        // (6 multichoose k1) * (6 multichoose k2) * ..
        self.into_iter()
            .map(|(_kind_idx, nkind)| num_multisets(nfaces, nkind as u32))
            .product()
    }

    pub fn all_multisets(&self) -> Vec<DiceVec> {
        fn rec(
            cb: &mut impl FnMut(DiceVec),
            acc: DiceVec,
            acc_len: u8,
            current_face: u8,
            kind_counts: DieKindCounts,
        ) {
            let ndice = kind_counts.ndice();

            // time to return the accumulator
            if ndice == 0 {
                cb(acc);
                return;
            // no more die faces
            } else if current_face > 6 {
                return;
            }

            // "spread" the current face across the possible dice kinds
            // like [s:3,hk:2,o:1].spread_face(1) => [1, 1, 1, 1hk, 1hk, 1o]
            let spread_dice = kind_counts.spread_face(current_face);

            // choose how many times we'll repeat this face
            for nface in (0..=ndice).rev() {
                let new_acc_len = acc_len + nface;
                // generate `spread_dice multichoose nface` for this iteration
                for mset in spread_dice.multisets_iter(nface) {
                    let new_acc = acc.extended_from(acc_len, mset);

                    let mut new_kind_counts = kind_counts;
                    new_kind_counts.sub_counts(DieKindCounts::from_dice_vec(mset));

                    rec(cb, new_acc, new_acc_len, current_face + 1, new_kind_counts);
                }
            }
        }

        let mut out = Vec::with_capacity(self.num_multisets(6) as usize);
        rec(&mut |dice| out.push(dice), DiceVec::new(), 0, 1, *self);
        out
    }

    /// Returns an `Iterator` of `(kind_idx: u8, count: u8)` tuples.
    #[allow(clippy::should_implement_trait)]
    #[inline]
    pub fn into_iter(self) -> impl Iterator<Item = (u8, u8)> {
        self.0.into_iter()
    }

    pub(crate) fn roll_dice<R: Rng>(self, rng: &mut R, dice_table: DieKindTable) -> DiceVec {
        let mut dice_arr = [Die::sentinel(); 6];
        let mut die_idx = 0;

        for (kind_idx, count) in self.into_iter().filter(|(_, count)| *count > 0) {
            // println!("roll_dice -> kind_idx: {kind_idx}, count: {count}");

            let distr = dice_table.get_kind(kind_idx).die_distr();
            for face in distr.sample_iter(&mut *rng).take(count as usize) {
                dice_arr[die_idx] = Die::new(face, kind_idx);
                die_idx += 1;
            }
        }

        dice_arr[0..die_idx].sort_unstable();
        // println!("roll_dice -> dice_arr: {dice_arr:?}")

        let dice_vec = DiceVec::from_array6(dice_arr);
        debug_assert!(dice_vec.invariant());

        dice_vec
    }
}

impl Default for DieKindCounts {
    fn default() -> Self {
        Self::new()
    }
}

impl FromIterator<(u8, u8)> for DieKindCounts {
    #[inline]
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (u8, u8)>,
    {
        Self(MultisetU4x8::from_iter(iter))
    }
}

#[derive(Copy, Clone, Eq)]
#[repr(transparent)]
pub struct Die(u8);

impl Default for Die {
    #[inline]
    fn default() -> Self {
        Self::sentinel()
    }
}

impl Die {
    #[inline]
    pub const fn new(face: u8, kind_idx: u8) -> Self {
        Self((face << 4) | (kind_idx & 0x0f))
    }

    #[inline]
    pub const fn sentinel() -> Self {
        Self(0)
    }

    #[inline]
    pub fn is_sentinel(self) -> bool {
        self == Self::sentinel()
    }

    #[inline]
    fn as_u8(self) -> u8 {
        self.0
    }

    #[inline]
    fn from_u8(inner: u8) -> Self {
        Self(inner)
    }

    #[inline]
    pub fn face(self) -> u8 {
        self.0 >> 4
    }

    #[inline]
    pub fn kind_idx(self) -> u8 {
        self.0 & 0x0f
    }
}

impl fmt::Debug for Die {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.face(), self.kind_idx())
    }
}

impl cmp::PartialEq for Die {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_u8().eq(&other.as_u8())
    }
}

impl cmp::PartialOrd for Die {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.as_u8().partial_cmp(&other.as_u8())
    }
    #[inline]
    fn lt(&self, other: &Self) -> bool {
        self.as_u8() < other.as_u8()
    }
    #[inline]
    fn le(&self, other: &Self) -> bool {
        self.as_u8() <= other.as_u8()
    }
    #[inline]
    fn gt(&self, other: &Self) -> bool {
        self.as_u8() > other.as_u8()
    }
    #[inline]
    fn ge(&self, other: &Self) -> bool {
        self.as_u8() >= other.as_u8()
    }
}

impl cmp::Ord for Die {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.as_u8().cmp(&other.as_u8())
    }
}

impl Hash for Die {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u8(self.as_u8())
    }
}

// ensure size_of(Die) == 1
const _: [(); 1] = [(); std::mem::size_of::<Die>()];

/// A list of `Die`s packed into a `u64`. The layout is `0x0000_d5d4_d3d2_d1d0`,
/// where `di` is the i'th `Die` in the list in byte representation.
///
/// The packed `u64` works somewhat like a C-string, with upper bytes in
/// byte-indexes >= `len(dice)` set to the sentinel value, `0x00 == Die::sentinel()`.
///
/// This representation has the advantage of letting us operate on whole `DiceVec`s
/// without iteration or branches in many cases by using lots of ugly bit-twiddling
/// hacks : )
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct DiceVec(u64);

impl_total_size_static!(DiceVec);

impl DiceVec {
    /// A new empty list of dice.
    #[inline]
    pub const fn new() -> Self {
        Self::from_array8([Die::sentinel(); 8])
    }

    #[inline]
    const fn as_u64(self) -> u64 {
        self.0
    }

    #[inline]
    const fn from_array8(arr: [Die; 8]) -> Self {
        // SAFETY: Die is #[repr(transparent)] and size_of::<Die>() == 1
        //         ==> mem::transmute is safe here.
        let arr: [u8; 8] = unsafe { std::mem::transmute(arr) };
        Self(u64::from_le_bytes(arr))
        // TODO(philiphayes): const assertion?
    }

    #[inline]
    const fn into_array8(self) -> [Die; 8] {
        // SAFETY: Die is #[repr(transparent)] and size_of::<Die>() == 1
        //         ==> mem::transmute is safe here.
        unsafe { std::mem::transmute(self.0.to_le_bytes()) }
    }

    #[inline]
    const fn from_array6(arr: [Die; 6]) -> Self {
        Self::from_array8([
            arr[0],
            arr[1],
            arr[2],
            arr[3],
            arr[4],
            arr[5],
            Die::sentinel(),
            Die::sentinel(),
        ])
    }

    #[inline]
    const fn into_array6(self) -> [Die; 6] {
        let arr = self.into_array8();
        [arr[0], arr[1], arr[2], arr[3], arr[4], arr[5]]
    }

    #[inline]
    fn set_die(&mut self, idx: u8, die: Die) {
        debug_assert_lt!(idx, 8);

        let idx = idx as u32;
        let shift = 8 * idx;

        self.0 = (self.0 & !(0xff << shift)) | ((die.as_u8() as u64) << shift);

        debug_assert!(self.invariant());
    }

    #[inline]
    fn get_die(self, idx: u8) -> Die {
        debug_assert_lt!(idx, 8);

        let idx = idx as u32;
        let shift = 8 * idx;

        Die::from_u8(((self.0 >> shift) & 0xff) as u8)
    }

    fn invariant(&self) -> bool {
        is_sorted_by(self.into_iter_no_sentinel(), |d1, d2| Some(d1.cmp(d2)))
            && is_partitioned(self.into_iter_all(), |die| !die.is_sentinel())
            && self.get_die(6).is_sentinel()
            && self.get_die(7).is_sentinel()
            && self.len() <= 6
    }

    #[inline]
    fn from_sorted_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Die>,
    {
        let dice_vec = Self(
            (0..8)
                .zip(iter)
                .map(|(idx, die)| (die.as_u8() as u64) << (8 * idx))
                .fold(0, |x, y| (x | y)),
        );
        debug_assert!(dice_vec.invariant());
        dice_vec
    }

    #[inline]
    pub fn into_iter_all(
        self,
    ) -> impl Iterator<Item = Die> + ExactSizeIterator + DoubleEndedIterator {
        self.into_array6().into_iter()
    }

    #[inline]
    pub fn into_iter_no_sentinel(self) -> impl Iterator<Item = Die> {
        // self.into_iter_all().filter(|die| !die.is_sentinel())
        U64NonZeroBytesIter::new(self.0).map(Die::from_u8)
    }

    #[inline]
    pub fn len(self) -> u8 {
        8 - (u64_leading_zero_bytes(self.0) as u8)
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn from_slice(dice: &mut [Die]) -> Self {
        dice.sort_unstable();
        Self::from_sorted_slice(dice)
    }

    pub fn from_sorted_slice(dice: &[Die]) -> Self {
        let mut arr = Self::new().into_array6();
        arr[..dice.len()].copy_from_slice(dice);

        let dice_vec = Self::from_array6(arr);
        debug_assert!(dice_vec.invariant());
        dice_vec
    }

    #[inline]
    fn extended_from(self, idx: u8, other: Self) -> Self {
        debug_assert_lt!(idx, 8);
        debug_assert_eq!(self.len(), idx);

        let out = Self(self.0 | other.0 << (8 * (idx as u32)));

        debug_assert!(out.invariant());
        out
    }

    #[inline]
    fn truncated(self, new_len: u8) -> Self {
        debug_assert_le!(new_len, 6);

        let out = Self(self.0 & !(0xffff_ffff_ffff_ffff_u64 << (8 * new_len)));

        debug_assert_eq!(out.len(), cmp::min(self.len(), new_len));
        debug_assert!(out.invariant());
        out
    }

    #[cfg(test)]
    fn push_die(&mut self, die: Die) {
        self.set_die(self.len(), die);
        debug_assert!(self.invariant());
    }

    #[cfg(test)]
    fn push_n_die(&mut self, die: Die, n: u8) {
        for _ in 0..n {
            self.push_die(die);
        }
        debug_assert!(self.invariant());
    }

    #[cfg(test)]
    fn merge(&self, other: &Self) -> Self {
        Self::from_sorted_iter(itertools::merge(
            self.into_iter_no_sentinel(),
            other.into_iter_no_sentinel(),
        ))
    }

    #[cfg(test)]
    fn split_next_die(self) -> (Self, Self) {
        let die = self.get_die(0);
        let split_idx = self.into_array6().partition_point(|&other| other == die) as u8;
        (self.truncated(split_idx), self.select_range(split_idx..6))
    }

    /// Only return unique dice.
    #[cfg(test)]
    fn unique_dice(&self) -> Self {
        let mut out = Self::new();
        let mut idx = 0;
        for die in self.into_iter_no_sentinel() {
            if !out.into_array6().contains(&die) {
                out.set_die(idx, die);
                idx += 1;
            }
        }
        debug_assert!(out.invariant());
        out
    }

    #[inline]
    pub fn into_die_kind_counts(self) -> DieKindCounts {
        DieKindCounts::from_dice_vec(self)
    }

    fn die_kind_idxs(&self) -> impl Iterator<Item = u8> {
        self.into_die_kind_counts()
            .into_iter()
            .filter_map(
                |(kind_idx, count)| {
                    if count != 0 {
                        Some(kind_idx)
                    } else {
                        None
                    }
                },
            )
    }

    fn die_with_kind_idx(self, kind_idx: u8) -> DiceVec {
        Self::from_sorted_iter(
            self.into_iter_no_sentinel()
                .filter(|die| die.kind_idx() == kind_idx),
        )
    }

    fn group_by_die_kind_idx(self) -> impl Iterator<Item = (u8, DiceVec)> {
        self.die_kind_idxs()
            .into_iter()
            .map(move |kind_idx| (kind_idx, self.die_with_kind_idx(kind_idx)))
    }

    #[inline]
    pub fn score(&self) -> u16 {
        DiceCounts::from_dice_vec(*self).score()
    }

    #[inline]
    pub fn exact_score(&self) -> u16 {
        DiceCounts::from_dice_vec(*self).exact_score()
    }

    #[inline]
    pub fn is_valid_hold(&self) -> bool {
        DiceCounts::from_dice_vec(*self).is_valid_hold()
    }

    #[inline]
    pub fn is_bust(&self) -> bool {
        DiceCounts::from_dice_vec(*self).is_bust()
    }

    /// Return the probability of rolling this set of dice, potentially containing
    /// many different kinds of dice with different die distributions.
    pub fn p_roll(&self, kind_table: &DieKindTable) -> f64 {
        self.group_by_die_kind_idx()
            .map(|(kind_idx, dice_kind)| {
                let distr = kind_table.get_kind(kind_idx).die_distr();
                DiceCounts::from_dice_vec(dice_kind).p_roll_with_distr(distr)
            })
            .product()
    }

    #[inline]
    pub fn multisets_iter(self, ndice: u8) -> DiceVecMultisetsIter {
        DiceVecMultisetsIter::new(self, ndice)
    }

    #[inline]
    fn select_range(self, range: Range<u8>) -> Self {
        let start = range.start as u32;
        let end = range.end as u32;
        let len = (end - start) as u32;

        debug_assert_lt!(start, 8);
        debug_assert_lt!(end, 8);

        let mask = !(0xffff_ffff_ffff_ffff_u64 << (8 * len));
        let out = Self((self.0 >> (8 * start)) & mask);

        debug_assert!(out.invariant());
        out
    }
}

impl cmp::PartialEq for DiceVec {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_u64().eq(&other.as_u64())
    }
}

impl cmp::Eq for DiceVec {}

impl cmp::PartialOrd for DiceVec {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl cmp::Ord for DiceVec {
    #[inline]
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        // first convert each u64 to big-endian before comparing. this places
        // the 1st die in the most-significant byte (, and 2nd die in 2nd
        // most-significant byte, and ...), which makes the default integer
        // comparison equivalent to a lexicographic array comparison.
        self.as_u64().to_be().cmp(&other.as_u64().to_be())
    }
}

impl Hash for DiceVec {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.as_u64())
    }
}

impl fmt::Debug for DiceVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let dice = self.into_iter_no_sentinel().collect::<Vec<_>>();
        write!(f, "{:?}", dice)
    }
}

/// An `Iterator` over all non-zero bytes in `x`, from the
/// least-significant byte to the most-significant byte.
pub struct U64NonZeroBytesIter {
    x: u64,
}

impl U64NonZeroBytesIter {
    #[inline]
    fn new(x: u64) -> Self {
        Self { x }
    }
}

impl ExactSizeIterator for U64NonZeroBytesIter {
    #[inline]
    fn len(&self) -> usize {
        (8 - u64_leading_zero_bytes(self.x)) as usize
    }
}

impl Iterator for U64NonZeroBytesIter {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let x = self.x;
        if x != 0 {
            let next = (x & 0xff) as u8;
            self.x = x >> u8::BITS;
            Some(next)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl FusedIterator for U64NonZeroBytesIter {}

/// An `Iterator` over all k-combinations of the multiset defined by a given
/// `DiceVec`.
pub struct DiceVecMultisetsIter {
    /// The complete multiset we're selecting from.
    mset: DiceVec,
    /// The last k-combination we'll output.
    last: DiceVec,
    /// The current k-combination we're about to output.
    comb: DiceVec,
    /// Every combination will have `.len() == k`
    k: u8,
    /// `true` if we've output all combinations and have nothing more to do.
    done: bool,
}

impl DiceVecMultisetsIter {
    fn new(mset: DiceVec, k: u8) -> Self {
        let n = mset.len();
        assert!(k <= n);

        Self {
            mset,
            last: mset.select_range((n - k)..n),
            comb: mset.select_range(0..k),
            k,
            done: false,
        }
    }

    fn update_to_next_combination(&mut self) {
        // find the rightmost element in `comb` that is less than the "max" value
        // it can have (which is the same as the element in the final combination
        // at the same index)
        // let maybe_next_idx = self
        //     .comb
        //     .into_iter_all()
        //     .zip(self.last.into_iter_all())
        //     .rposition(|(x, l)| x < l);
        let maybe_next_idx = u64_leading_byte_idx_lt(self.comb.as_u64(), self.last.as_u64());

        // if comb == last, then there will be no rightmost element less than the
        // max, so we return None to show that we're done.
        let (i, x) = match maybe_next_idx {
            Some(i) => (i, self.comb.get_die(i).as_u8()),
            None => {
                // comb == last => done!
                self.done = true;
                return;
            }
        };

        // find the successor element (the next element in the multiset greater
        // than arr[i])
        // let maybe_succ_idx = self.mset.into_iter_all().position(|y| x < y);
        let maybe_succ_idx = u64_trailing_byte_idx_lt(x, self.mset.as_u64());

        let j = match maybe_succ_idx {
            Some(j) => j as u8,
            // since we've already checked we're not at the end, it cannot happen
            // that there is no successor.
            None => unreachable!(),
        };

        // replaces comb[i] with its successor and the remainder with all elements
        // that follow in mset.
        self.comb = self
            .comb
            .truncated(i)
            .extended_from(i, self.mset.select_range(j..(j + (self.k - i))));
        debug_assert!(self.comb.invariant());
    }
}

impl Iterator for DiceVecMultisetsIter {
    type Item = DiceVec;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            None
        } else {
            let comb = self.comb;
            self.update_to_next_combination();
            Some(comb)
        }
    }
}

impl FusedIterator for DiceVecMultisetsIter {}

/// A compressed representation of a set of dice, stored as counts of each die
/// packed into a u32. Visually,
///
/// `XXXX 1111 2222 3333 4444 5555 6666 XXXX`, where `XXXX`s are unused nibbles
/// and `NNNN` is the nibble for the n'th die.
///
/// This representation has the added benefit of being order invariant, since
/// the order of dice in a set is not important.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct DiceCounts(MultisetU4x8);

impl DiceCounts {
    /// A new empty set of dice rolls.
    #[inline]
    pub const fn new() -> Self {
        Self(MultisetU4x8::new())
    }

    #[inline]
    const fn from_counts(cs: [u8; 6]) -> Self {
        Self(MultisetU4x8::from_counts([
            0, cs[0], cs[1], cs[2], cs[3], cs[4], cs[5], 0,
        ]))
    }

    pub fn from_dice_vec(dice_vec: DiceVec) -> Self {
        dice_vec.into_iter_no_sentinel().collect()
    }

    /// A convenience function for constructing a `DiceCounts` set from an unordered
    /// list of dice rolls.
    #[cfg(test)]
    pub fn from_faces(faces: &[u8]) -> Self {
        Self(MultisetU4x8::from_iter_flat(faces.iter().copied()))
    }

    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0.as_u32()
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.0.is_empty()
    }

    // The number of dice in this set
    #[inline]
    pub fn len(self) -> u8 {
        self.0.len()
    }

    #[inline]
    pub fn get_count(self, face: u8) -> u8 {
        self.0.get_count(face)
    }

    #[inline]
    #[cfg(test)]
    pub fn set_count(&mut self, face: u8, count: u8) {
        self.0.set_count(face, count)
    }

    #[inline]
    pub fn add_count(&mut self, face: u8, count: u8) {
        self.0 += MultisetU4x8::from_count(face, count);
    }

    #[cfg(test)]
    #[inline]
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0)
    }

    /// Returns the score of a set of dice. will still work with non-scoring dice
    /// mixed in.
    pub fn score(self) -> u16 {
        if self.is_empty() {
            return 0;
        }

        let c1 = self.get_count(1);
        let c2 = self.get_count(2);
        let c3 = self.get_count(3);
        let c4 = self.get_count(4);
        let c5 = self.get_count(5);
        let c6 = self.get_count(6);

        // check for straights
        if c1 > 0 && c2 > 0 && c3 > 0 && c4 > 0 && c5 > 0 && c6 > 0 {
            return 1500;
        } else if c2 > 0 && c3 > 0 && c4 > 0 && c5 > 0 && c6 > 0 {
            let extra_5 = if c5 == 2 { 50 } else { 0 };
            // can't have an extra 1, since then we would just have a full straight
            return 750 + extra_5;
        } else if c1 > 0 && c2 > 0 && c3 > 0 && c4 > 0 && c5 > 0 {
            let extra_1 = if c1 == 2 { 100 } else { 0 };
            let extra_5 = if c5 == 2 { 50 } else { 0 };
            return 500 + extra_1 + extra_5;
        }

        let mut score: u16 = 0;

        for dice in 1u16..=6 {
            let count = self.get_count(dice as u8) as u16;

            if count == 0 {
                continue;
            }

            // score single or double 1s or 5s
            if count < 3 {
                if dice == 1 {
                    score += 100 * count;
                } else if dice == 5 {
                    score += 50 * count;
                }
            } else {
                let mult: u16 = 1 << (count - 3);
                let base = if dice == 1 { 1000 } else { dice * 100 };
                score += base * mult
            }
        }

        score
    }

    /// Return the score of a set of dice. If there are any non-scoring dice in
    /// the set, then this returns 0.
    pub fn exact_score(self) -> u16 {
        if self.is_empty() {
            return 0;
        }

        // check for pure straights and straights with 1 added scoring dice
        if self == Self::from_counts([1, 1, 1, 1, 1, 1]) {
            return 1500;
        } else if self == Self::from_counts([0, 1, 1, 1, 1, 1]) {
            return 750;
        } else if self == Self::from_counts([0, 1, 1, 1, 2, 1]) {
            return 750 + 50;
        } else if self == Self::from_counts([1, 1, 1, 1, 1, 0]) {
            return 500;
        } else if self == Self::from_counts([2, 1, 1, 1, 1, 0]) {
            return 500 + 100;
        } else if self == Self::from_counts([1, 1, 1, 1, 2, 0]) {
            return 500 + 50;
        }

        let mut score: u16 = 0;

        for dice in 1u16..=6 {
            let count = self.get_count(dice as u8) as u16;

            if count == 0 {
                continue;
            }

            // score single or double 1s or 5s
            if count < 3 {
                if dice == 1 {
                    score += 100 * count;
                } else if dice == 5 {
                    score += 50 * count;
                } else {
                    // less than 3 _non-scoring_ dice, this is not a valid scoring
                    // dice set.
                    return 0;
                }
            } else {
                let mult: u16 = 1 << (count - 3);
                let base = if dice == 1 { 1000 } else { dice * 100 };
                score += base * mult
            }
        }

        score
    }

    pub fn is_valid_hold(self) -> bool {
        // spurious clippy lint...
        #[allow(clippy::if_same_then_else)]
        if self.is_empty() {
            false
        } else if self == Self::from_counts([1, 1, 1, 1, 1, 1]) {
            true
        } else if self == Self::from_counts([0, 1, 1, 1, 1, 1]) {
            true
        } else if self == Self::from_counts([0, 1, 1, 1, 2, 1]) {
            true
        } else if self == Self::from_counts([1, 1, 1, 1, 1, 0]) {
            true
        } else if self == Self::from_counts([2, 1, 1, 1, 1, 0]) {
            true
        } else if self == Self::from_counts([1, 1, 1, 1, 2, 0]) {
            true
        } else {
            // do we have less than 3 (and greater than 0) of any non-scoring
            // dice (2, 3, 4, or 6)? then this is not a valid hold.
            !u32_any_nibs_between(self.as_u32(), 0x0101_1100, 0, 3)
        }
    }

    /// Return true if this set has no scoring dice, also called a "bust".
    #[inline]
    pub fn is_bust(self) -> bool {
        relative_eq!(self.score() as f64, 0.0)
    }

    /// Return the probability of rolling this set of dice (assuming all standard
    /// dice).
    ///
    /// let n = number of dice in the set
    ///     P = n! / (6^n * ∏_{i∈[1,6]} c_i!)
    ///         where c_i is the count of the i'th dice in the set
    #[cfg(test)]
    fn p_roll(self) -> f64 {
        let n = self.len();

        let prod: u32 = (1..=6)
            .map(|roll| factorial(self.get_count(roll) as u32))
            .product();

        let m = 6_u32.pow(n as u32);

        (factorial(n as u32) as f64) / ((prod * m) as f64)
    }

    /// Return the probability of rolling this set of dice, with faces weighted
    /// according to the die distribution `distr`.
    pub fn p_roll_with_distr(self, distr: DieDistr) -> f64 {
        let n = self.len() as u32;

        let (prod, p_joint) = (1..=6).fold((1, 1.0), |(prod, p_joint), face| {
            let nface = self.get_count(face);
            let prod = prod * factorial(nface as u32);
            let p_joint = p_joint * distr.p_face(face).powi(nface as i32);
            (prod, p_joint)
        });

        ((factorial(n) / prod) as f64) * p_joint
    }

    pub fn to_rolls(self) -> Vec<u8> {
        let mut rolls = Vec::new();
        for roll in 1..=6 {
            for _ in 0..self.get_count(roll) {
                rolls.push(roll);
            }
        }
        rolls
    }

    #[cfg(test)]
    fn all_multisets(total_dice: u8) -> AllDiceMultisetsIter {
        AllDiceMultisetsIter::new(total_dice)
    }

    /// Return all possible sub-multisets of size `ndice` of a given set of dice rolls.
    #[cfg(test)]
    fn multisets(&self, ndice: u8) -> Vec<DiceCounts> {
        fn rec(
            cb: &mut impl FnMut(DiceCounts),
            mut acc: DiceCounts,
            mut left: DiceCounts,
            ndice: u8,
        ) {
            // time to return the accumulator
            if ndice == 0 {
                cb(acc);
                return;
            }

            // if ndice == 1 we can just return each unique element
            if ndice == 1 {
                for roll in 1..=6 {
                    if left.get_count(roll) > 0 {
                        let mut new_acc = acc;
                        new_acc.add_count(roll, 1);
                        cb(new_acc);
                    }
                }
                return;
            }

            let elts_left = left.len();

            // special case: |S| = n means just one possible combination: S
            if elts_left == ndice {
                cb(acc.add(left));
                return;
            }

            // will never have enough; skip
            if elts_left < ndice {
                return;
            }

            // find next non-zero-count roll
            let roll = (1..=6).find(|&roll| left.get_count(roll) > 0).unwrap();
            let count = left.get_count(roll);

            // completely remove the element from the set
            left.set_count(roll, 0);

            // add zero or more of this element
            for to_add in (0..=cmp::min(count, ndice)).rev() {
                acc.set_count(roll, to_add);
                rec(cb, acc, left, ndice - to_add);
            }
        }

        let mut out = Vec::new();
        rec(
            &mut |counts| out.push(counts),
            DiceCounts::new(),
            *self,
            ndice,
        );
        out
    }
}

impl fmt::Debug for DiceCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (self.to_rolls()).fmt(f)
    }
}

#[cfg(test)]
impl cmp::Ord for DiceCounts {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.to_rolls().cmp(&other.to_rolls())
    }
}

#[cfg(test)]
impl cmp::PartialOrd for DiceCounts {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl FromIterator<Die> for DiceCounts {
    #[inline]
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Die>,
    {
        Self(MultisetU4x8::from_iter_flat(
            iter.into_iter().map(|die| die.face()),
        ))
    }
}

#[cfg(test)]
#[rustfmt::skip]
impl Arbitrary for DiceCounts {
    type Parameters = ();
    type Strategy = BoxedStrategy<DiceCounts>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
        static ROLLS_SET: [u8; 36] = [
            1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2,
            3, 3, 3, 3, 3, 3,
            4, 4, 4, 4, 4, 4,
            5, 5, 5, 5, 5, 5,
            6, 6, 6, 6, 6, 6,
        ];

        proptest::sample::subsequence(ROLLS_SET.as_ref(), 0..=6)
            .prop_map(|rolls| DiceCounts::from_faces(&rolls))
            .boxed()
    }
}

/// Parse a comma/space/tab separated list of dice into a `DiceCounts` set.
/// Enclosing brackets ('[' or ']') optional.
impl FromStr for DiceCounts {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut counts = DiceCounts::new();

        let s = s.trim_start_matches('[');
        let s = s.trim_end_matches(']');

        let splitters = &[',', ' ', '\n', '\t'];

        for roll_str in s.split(splitters).filter(|s| !s.is_empty()) {
            let roll = roll_str
                .parse::<u8>()
                .map_err(|err| format!("dice roll is not a valid integer: {}", err))?;
            if (1..=6).contains(&roll) {
                counts.add_count(roll, 1);
            } else {
                return Err(format!("roll is out of range [1, 6]: {}", roll));
            }
        }

        Ok(counts)
    }
}

cfg_test! {
    /// An `Iterator` over combinations (with replacement) of _all_ dice roll outcomes.
    struct AllDiceMultisetsIter {
        /// the _next_ combination we'll output (unless we're done).
        counts: DiceCounts,
        /// total number of dice rolls per combination.
        total_dice: u8,
        /// set to `true` when we're done generating.
        done: bool,
    }

    impl AllDiceMultisetsIter {
        fn new(total_dice: u8) -> Self {
            // initialize with our the very first combination: [1, 1, .., 1]
            let mut counts = DiceCounts::new();
            counts.set_count(1, total_dice);

            Self {
                counts,
                total_dice,
                done: false,
            }
        }
    }

    impl Iterator for AllDiceMultisetsIter {
        type Item = DiceCounts;

        fn next(&mut self) -> Option<Self::Item> {
            if self.done {
                return None;
            }

            // this is the combination we're about to output
            let current = self.counts;

            // we're done when we finally reach [6, 6, .., 6]
            if self.counts.get_count(6) == self.total_dice {
                self.done = true;
            } else {
                // patch self.counts to be the _next_ combination we'll output
                for i in 1..=5 {
                    let count_i = self.counts.get_count(i);
                    if count_i > 0 {
                        self.counts.add_count(i + 1, 1);
                        for j in 2..=i {
                            self.counts.set_count(j, 0);
                        }
                        self.counts.set_count(1, count_i - 1);
                        break;
                    }
                }
            }

            Some(current)
        }
    }
}

#[cfg(test)]
pub mod prop {
    use super::*;
    use proptest::{prelude::*, sample::select};
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro64Star;

    const N: usize = DieKind::all().len();
    pub static DIE_KIND_MATRIX: [DieKind; N * N] = die_kind_matrix();

    pub const fn die_kind_matrix() -> [DieKind; N * N] {
        const N: usize = DieKind::all().len();
        let mut out = [DieKind::SENTINEL; N * N];

        let mut idx = 0;

        loop {
            if idx >= N * N {
                break;
            }

            let kind = (idx / N) + 1;
            out[idx] = DieKind::from_u8(kind as u8);

            idx += 1;
        }

        out
    }

    pub fn arb_die_kind() -> impl Strategy<Value = DieKind> {
        select(DieKind::all().as_slice())
    }

    pub fn arb_dice_set_compact() -> impl Strategy<Value = (DieKindTable, DieKindCounts)> {
        crate::parse::prop::arb_dice_set().prop_map(|dice_set| dice_set.to_compact_form())
    }

    pub fn arb_rng() -> impl Strategy<Value = Xoroshiro64Star> {
        any::<u64>().prop_map(Xoroshiro64Star::seed_from_u64)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod test {
    use super::{prop::*, *};
    use crate::{num_combinations, parse};
    use approx::assert_relative_eq;
    use claim::assert_le;
    use proptest::{array::uniform8, prelude::*};
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro64Star;
    use std::{cmp::min, collections::HashSet};

    macro_rules! table {
        () => {
            DieKindTable::all_standard()
        };
        ($($x:tt),+ $(,)?) => {
            DieKindTable::from_die_kinds(&[ $( DieKind::from_memnonic(stringify!($x)).unwrap() ),+ ])
        };
    }

    macro_rules! dice {
        () => {
            DiceVec::new()
        };
        ($tbl:expr, [ $($x:tt),+ $(,)? ] $(,)?) => {
            parse::DiceVec::from_iter([$( parse::Die::from_str(stringify!($x)).unwrap() ),+ ].into_iter())
                .to_compact_form(&$tbl)
        };
        ($($x:tt),+ $(,)?) => {
            dice![table!(), [ $( $x ),+ ]]
        };
    }

    // simple recursive implementation
    fn all_dice_multisets_ref(ndice: u8) -> Vec<DiceCounts> {
        fn rec(cb: &mut impl FnMut(DiceCounts), counts: DiceCounts, current_dice: u8, ndice: u8) {
            // time to return the accumulator
            if ndice == 0 {
                cb(counts);
                return;
            // passed the end; nothing to return
            } else if current_dice > 6 {
                return;
            }

            // can choose 0..=ndice of the current element
            for choose_counts in 0..=ndice {
                let mut new_counts = counts;
                new_counts.set_count(current_dice, choose_counts);
                rec(cb, new_counts, current_dice + 1, ndice - choose_counts);
            }
        }

        let mut out = Vec::new();
        rec(&mut |counts| out.push(counts), DiceCounts::new(), 1, ndice);
        out.sort_unstable();
        out
    }

    #[allow(unused)]
    fn dice_vec_multisets_ref(mset: DiceVec, ndice: u8) -> Vec<DiceVec> {
        fn rec(cb: &mut impl FnMut(DiceVec), acc: DiceVec, left: DiceVec, ndice: u8) {
            // time to return the accumulator
            if ndice == 0 {
                cb(acc);
                return;
            }

            // if ndice == 1 we can just return each unique element
            if ndice == 1 {
                for die in left.unique_dice().into_iter_no_sentinel() {
                    let mut new_acc = acc;
                    new_acc.push_die(die);
                    cb(new_acc);
                }
                return;
            }

            let elts_left = left.len();

            // special case: |S| = n means just one possible combination: S
            if elts_left == ndice {
                cb(acc.merge(&left));
                return;
            }

            // will never have enough; skip
            if elts_left < ndice {
                return;
            }

            // take next set of identical dice
            let (die_set, left) = left.split_next_die();
            let die = die_set.get_die(0);
            let count = die_set.len();

            for to_add in (0..=min(count, ndice)).rev() {
                let mut new_acc = acc;
                new_acc.push_n_die(die, to_add);
                rec(cb, new_acc, left, ndice - to_add);
            }
        }

        let mut out = Vec::new();
        rec(&mut |dice| out.push(dice), DiceVec::new(), mset, ndice);
        out
    }

    #[test]
    fn test_dice_counts_score() {
        assert_eq!(0, DiceCounts::from_faces(&[]).score());
        assert_eq!(100, DiceCounts::from_faces(&[1]).score());
        assert_eq!(150, DiceCounts::from_faces(&[5, 1]).score());
        assert_eq!(0, DiceCounts::from_faces(&[2, 2, 3]).score());
        assert_eq!(1500, DiceCounts::from_faces(&[6, 5, 4, 3, 2, 1]).score());
        assert_eq!(250, DiceCounts::from_faces(&[1, 1, 5, 3, 2, 2]).score());
        assert_eq!(1050, DiceCounts::from_faces(&[1, 1, 5, 3, 1, 2]).score());
        assert_eq!(400, DiceCounts::from_faces(&[4, 4, 4]).score());
        assert_eq!(800, DiceCounts::from_faces(&[4, 4, 4, 4]).score());
        assert_eq!(1600, DiceCounts::from_faces(&[4, 4, 4, 4, 4]).score());
        assert_eq!(3200, DiceCounts::from_faces(&[4, 4, 4, 4, 4, 4]).score());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(1000))]

        #[test]
        fn test_prop_dice_counts_is_valid_hold(counts in any::<DiceCounts>()) {
            assert_eq!(counts.exact_score() > 0, counts.is_valid_hold());
        }
    }

    #[test]
    fn test_dice_counts_from_rolls() {
        assert!(DiceCounts::from_faces(&[]).is_empty());

        assert_eq!(0, DiceCounts::from_faces(&[]).len());
        assert_eq!(1, DiceCounts::from_faces(&[1]).len());
        assert_eq!(2, DiceCounts::from_faces(&[4, 2]).len());
        assert_eq!(6, DiceCounts::from_faces(&[6, 5, 4, 3, 2, 1]).len());

        assert_eq!(1, DiceCounts::from_faces(&[6, 5, 4, 3, 2, 1]).get_count(6));
        assert_eq!(3, DiceCounts::from_faces(&[6, 5, 3, 3, 3]).get_count(3));
        assert_eq!(0, DiceCounts::from_faces(&[6, 5, 3, 3, 3]).get_count(1));
        assert_eq!(6, DiceCounts::from_faces(&[3, 3, 3, 3, 3, 3]).get_count(3));
        assert_eq!(6, DiceCounts::from_faces(&[6, 6, 6, 6, 6, 6]).get_count(6));
        assert_eq!(0, DiceCounts::from_faces(&[6, 6, 6, 6, 6, 6]).get_count(3));
    }

    #[test]
    fn test_dice_counts_from_counts() {
        assert!(DiceCounts::from_counts([0; 6]).is_empty());

        assert_eq!(
            DiceCounts::from_counts([1; 6]),
            DiceCounts::from_faces(&[1, 2, 3, 4, 5, 6]),
        );

        assert_eq!(
            DiceCounts::from_counts([1, 0, 1, 0, 1, 0]),
            DiceCounts::from_faces(&[1, 3, 5]),
        );
        assert_eq!(
            DiceCounts::from_counts([1, 2, 0, 3, 0, 0]),
            DiceCounts::from_faces(&[1, 2, 2, 4, 4, 4]),
        );
    }

    #[test]
    fn test_all_dice_multisets_iter() {
        for k in 0..=6 {
            let mut combs = AllDiceMultisetsIter::new(k as u8).collect::<Vec<_>>();
            combs.sort_unstable();

            // outputs expected number of elements
            assert_eq!(num_multisets(6, k) as usize, combs.len());

            // no duplicates
            let combs_set = HashSet::<DiceCounts>::from_iter(combs.clone().into_iter());
            assert_eq!(num_multisets(6, k) as usize, combs_set.len());

            // matches recursive implementation
            let combs_ref = all_dice_multisets_ref(k as u8);
            assert_eq!(combs, combs_ref);
        }
    }

    #[test]
    fn test_all_dice_vec_multisets() {
        let mut dice = dice![1, 1, 1, 1, 1, 1];

        // if we only have one die kind, the multisets should match
        for ndice in (1..=6).rev() {
            dice = dice.truncated(ndice);

            let mut ref_combs = DiceCounts::all_multisets(ndice).collect::<Vec<_>>();
            ref_combs.sort_unstable();

            assert_eq!(
                DieKindCounts::from_dice_vec(dice)
                    .all_multisets()
                    .into_iter()
                    .map(DiceCounts::from_dice_vec)
                    .collect::<Vec<_>>(),
                ref_combs,
            );
        }

        let kind_counts = DieKindCounts::from_dice_vec(dice!(
            table![s, hk, o],
            [1, 1, 1, 1hk, 1hk, 1o],
        ));
        let combs = kind_counts.all_multisets();

        // outputs expected number of elements
        assert_eq!(kind_counts.num_multisets(6) as usize, combs.len());

        // no duplicates
        let combs_set = HashSet::<DiceVec>::from_iter(combs.into_iter());
        assert_eq!(kind_counts.num_multisets(6) as usize, combs_set.len());
    }

    #[test]
    fn test_dice_multisets() {
        // [1 1], [1 3], [1 5], [3 5]
        let expected = vec![
            DiceCounts::from_faces(&[1, 1]),
            DiceCounts::from_faces(&[1, 3]),
            DiceCounts::from_faces(&[1, 5]),
            DiceCounts::from_faces(&[3, 5]),
        ];
        assert_eq!(
            &expected,
            &DiceCounts::from_faces(&[1, 1, 3, 5]).multisets(2),
        );

        // when all the elements are unique, this is just the normal set
        // combinations
        for k in 0..=6 {
            assert_eq!(
                num_combinations(6, k) as usize,
                DiceCounts::from_faces(&[1, 2, 3, 4, 5, 6])
                    .multisets(k as u8)
                    .len(),
            );
        }
    }

    #[test]
    fn test_dice_vec_unique_dice() {
        assert_eq!(dice![], dice![].unique_dice());

        let t = table![s, hk, o];
        assert_eq!(dice!(t, [1, 2hk, 3o]), dice!(t, [1, 2hk, 3o]).unique_dice(),);
        assert_eq!(
            dice!(t, [1, 2hk, 3o]),
            dice!(t, [1, 1, 1, 2hk, 2hk, 3o]).unique_dice(),
        );
        assert_eq!(
            dice!(t, [1, 1hk, 1o]),
            dice!(t, [1, 1, 1, 1hk, 1hk, 1o]).unique_dice(),
        );
    }

    #[test]
    fn test_dice_vec_merge() {
        assert_eq!(dice![], dice![].merge(&dice![]));
        assert_eq!(dice![1], dice![1].merge(&dice![]));
        assert_eq!(dice![1], dice![].merge(&dice![1]));

        let t = table![s, hk, o];
        assert_eq!(
            dice!(t, [1, 2hk, 3o]),
            dice!(t, [2hk]).merge(&dice!(t, [1, 3o])),
        );
        assert_eq!(
            dice!(t, [1, 1, 2hk, 2hk, 3o, 3o]),
            dice!(t, [2hk, 3o]).merge(&dice!(t, [1, 1, 2hk, 3o]))
        );
        assert_eq!(
            dice!(t, [1, 1, 1hk, 1hk, 1o, 1o]),
            dice!(t, [1hk, 1o]).merge(&dice!(t, [1, 1, 1hk, 1o]))
        );
    }

    #[test]
    fn test_dice_vec_multisets() {
        use itertools::Itertools;

        let t = table!(s, hk, o);

        macro_rules! dicet {
            () => {
                DiceVec::new()
            };
            ($($x:tt),+ $(,)?) => {
                dice![t, [ $( $x ),+ ]]
            };
        }

        assert_eq!(
            vec![dicet![]],
            dicet![1, 1, 1, 2hk, 2hk, 3o]
                .multisets_iter(0)
                .collect_vec(),
        );
        assert_eq!(
            vec![dicet![1], dicet![2hk], dicet![3o]],
            dicet![1, 1, 1, 2hk, 2hk, 3o]
                .multisets_iter(1)
                .collect_vec(),
        );
        assert_eq!(
            vec![dicet![1], dicet![1hk], dicet![1o]],
            dicet![1, 1, 1, 1hk, 1hk, 1o]
                .multisets_iter(1)
                .collect_vec(),
        );
        assert_eq!(
            vec![
                dicet![1, 1],
                dicet![1, 2hk],
                dicet![1, 3o],
                dicet![2hk, 2hk],
                dicet![2hk, 3o]
            ],
            dicet![1, 1, 1, 2hk, 2hk, 3o]
                .multisets_iter(2)
                .collect_vec(),
        );
        assert_eq!(
            vec![
                dicet![1, 1, 1],
                dicet![1, 1, 2hk],
                dicet![1, 1, 3o],
                dicet![1, 2hk, 2hk],
                dicet![1, 2hk, 3o],
                dicet![2hk, 2hk, 3o],
            ],
            dicet![1, 1, 1, 2hk, 2hk, 3o]
                .multisets_iter(3)
                .collect_vec(),
        );
        assert_eq!(
            vec![dicet![1, 1, 1, 2hk, 2hk, 3o]],
            dicet![1, 1, 1, 2hk, 2hk, 3o]
                .multisets_iter(6)
                .collect_vec(),
        );

        assert_eq!(
            vec![
                dicet![1, 1hk],
                dicet![1, 2],
                dicet![1, 2hk],
                dicet![1, 3],
                dicet![1, 3hk],
                dicet![1hk, 2],
                dicet![1hk, 2hk],
                dicet![1hk, 3],
                dicet![1hk, 3hk],
                dicet![2, 2hk],
                dicet![2, 3],
                dicet![2, 3hk],
                dicet![2hk, 3],
                dicet![2hk, 3hk],
                dicet![3, 3hk],
            ],
            dicet![1, 1hk, 2, 2hk, 3, 3hk]
                .multisets_iter(2)
                .collect_vec(),
        );

        assert_eq!(
            vec![dicet![1, 1], dicet![1, 1o], dicet![1o, 1o]],
            dicet![1, 1, 1o, 1o, 1o, 1o].multisets_iter(2).collect_vec(),
        );

        assert_eq!(
            vec![
                dicet![1, 1, 2, 2],
                dicet![1, 1, 2, 3],
                dicet![1, 2, 2, 2],
                dicet![1, 2, 2, 3],
                dicet![2, 2, 2, 3],
            ],
            dicet![1, 1, 2, 2, 2, 3].multisets_iter(4).collect_vec(),
        );

        // when all the elements are unique, this is just the normal set
        // combinations
        for k in 0..=6 {
            assert_eq!(
                num_combinations(6, k as u32) as usize,
                dicet![1, 1hk, 1o, 2, 2hk, 2o].multisets_iter(k).count(),
            );
        }
    }

    #[test]
    fn test_dice_counts_probability() {
        assert_relative_eq!(
            3.0 / ((6.0_f64).powf(3.0)),
            DiceCounts::from_faces(&[1, 1, 3]).p_roll()
        );

        for n in 1..=6 {
            assert_relative_eq!(
                1.0,
                DiceCounts::all_multisets(n)
                    .map(|counts| counts.p_roll())
                    .sum::<f64>(),
                epsilon = 1e-10,
            );
        }
    }

    #[test]
    fn test_dice_vec_p_roll() {
        // rolling just standard dice should give approx. the same result as
        // DiceCounts::p_roll
        let t = table![s];
        assert_relative_eq!(
            dice!(t, [1, 1, 1, 2, 2, 3]).p_roll(&t),
            DiceCounts::from_dice_vec(dice!(t, [1, 1, 1, 2, 2, 3])).p_roll(),
        );
        assert_relative_eq!(
            dice!(t, [1, 1, 2, 3]).p_roll(&t),
            DiceCounts::from_dice_vec(dice!(t, [1, 1, 2, 3])).p_roll(),
        );

        // exhaustively check that all combinations of {s, hk, o} die kinds will
        // emit proper probability mass functions (where the pmf is effectively
        // { pmf(dice) := dice.p_roll()) | dice ∈ die_kinds.all_multisets() }

        let mut die_kind_combs = Vec::new();

        use super::DieKind::{HeavenlyKingdomDie as hk, OddDie as o, Standard as s};
        let t = table![s, hk, o];
        let s_idx = t.get_kind_idx(s).unwrap();
        let hk_idx = t.get_kind_idx(hk).unwrap();
        let o_idx = t.get_kind_idx(o).unwrap();

        for ndice in 0..=6 {
            for ns in 0..=ndice {
                let nleft = ndice - ns;
                for nhk in 0..=nleft {
                    let no = nleft - nhk;

                    die_kind_combs.push(DieKindCounts::from_iter([
                        (s_idx, ns),
                        (hk_idx, nhk),
                        (o_idx, no),
                    ]))
                }
            }
        }

        // comment this if we're testing exhaustively, o.w. takes a few secs
        let die_kind_combs = die_kind_combs.into_iter().step_by(12);

        for die_kinds in die_kind_combs {
            assert_relative_eq!(
                1.0,
                die_kinds
                    .all_multisets()
                    .into_iter()
                    .map(|dice| dice.p_roll(&t))
                    .sum(),
                epsilon = 1.0e-10
            );
        }
    }

    fn arb_valid_die() -> impl Strategy<Value = Die> {
        ((1_u8..=6), (0_u8..=15)).prop_map(|(face, kind_idx)| Die::new(face, kind_idx))
    }

    fn arb_die() -> impl Strategy<Value = Die> {
        prop_oneof![
            1 => Just(Die::sentinel()),
            1 => arb_valid_die(),
        ]
    }

    fn niters(n: u32) -> ProptestConfig {
        ProptestConfig::with_cases(n)
    }

    // ensure `Die`'s ordering is the same as the lexicographic ordering, i.e.,
    // compare faces, then compare kind_idxs.
    #[test]
    fn test_die_ord() {
        #[derive(PartialEq, Eq, PartialOrd, Ord)]
        struct DieModel {
            face: u8,
            kind_idx: u8,
        }

        proptest!(niters(2000), |(die1 in arb_die(), die2 in arb_die())| {
            let die_model1 = DieModel {
                face: die1.face(),
                kind_idx: die1.kind_idx(),
            };
            let die_model2 = DieModel {
                face: die2.face(),
                kind_idx: die2.kind_idx(),
            };

            assert_eq!(die1.eq(&die2), die_model1.eq(&die_model2));
            assert_eq!(die1.cmp(&die2), die_model1.cmp(&die_model2));
        });
    }

    // ensure `DiceVec`'s ordering is the same as the default lexicographic array
    // ordering.
    #[test]
    fn test_dice_vec_ord() {
        proptest!(niters(2000), |(
            mut dice1 in uniform8(arb_die()),
            mut dice2 in uniform8(arb_die()),
        )| {
            // a comparator which will ensure all the sentinels in the array
            // collect at the end.
            fn cmp_die(d1: &Die, d2: &Die) -> cmp::Ordering {
                if d1.is_sentinel() && d2.is_sentinel() {
                    cmp::Ordering::Equal
                } else if d1.is_sentinel() {
                    cmp::Ordering::Greater
                } else if d2.is_sentinel() {
                    cmp::Ordering::Less
                } else {
                    d1.cmp(d2)
                }
            }

            dice1.sort_unstable_by(cmp_die);
            dice2.sort_unstable_by(cmp_die);

            let dice_vec1 = DiceVec::from_array8(dice1);
            let dice_vec2 = DiceVec::from_array8(dice2);

            assert_eq!(
                dice1.eq(&dice2),
                dice_vec1.eq(&dice_vec2),
            );
            assert_eq!(
                dice1.cmp(&dice2),
                dice_vec1.cmp(&dice_vec2),
            );
        });
    }

    /// let X be some unknown Bernoulli distribution
    /// let p_actual be the _actual_ probability Pr[X = 1]
    /// let p_emp be our _empirical_ probability Pr[X = 1]
    /// let E = max_l1_error >= |p - r|_1 be our maximum tolerated error
    /// let Z = confidence level, given as Z-level of N(0, 1)
    ///
    /// => required number of trials is n = Z^2 / (4 E^2)
    fn num_trials(max_l1_error: f64, confidence: f64) -> usize {
        ((confidence * confidence) / (4.0 * (max_l1_error * max_l1_error))).ceil() as usize
    }

    // ensure DieDistr::sample returns approx. the true distribution.
    #[test]
    fn test_die_distr_sample() {
        // number of trials for at-most 1% error at 5σ confidence (=> n = 62,500)
        let max_err = 0.01;
        let confidence = 5.0;
        let n = num_trials(max_err, confidence);
        let mut rng = Xoroshiro64Star::seed_from_u64(0xd15c0);

        for die_kind in DieKind::all() {
            let distr = die_kind.die_distr();

            let counts = distr.clone().sample_iter(&mut rng).take(n).fold(
                [0_usize; 6],
                |mut counts, face| {
                    counts[(face - 1) as usize] += 1;
                    counts
                },
            );
            let emp_distr = counts.map(|count| (count as f64) / (n as f64));

            for face in 1..=6 {
                let p_actual = distr.p_face(face);
                let p_emp = emp_distr[(face - 1) as usize];
                let err = (p_actual - p_emp).abs();
                assert_le!(err, max_err);
            }
        }
    }

    #[test]
    fn test_dice_vec_roll_dice() {
        proptest!(niters(10), |((dice_table, die_kind_counts) in arb_dice_set_compact(), mut rng in arb_rng())| {
            let possible_rolls = die_kind_counts
                .all_multisets()
                .into_iter()
                .collect::<HashSet<_>>();

            // println!("possible_rolls: {possible_rolls:?}");

            for _ in 0..100 {
                let rolled_dice = die_kind_counts.roll_dice(&mut rng, dice_table);
                // println!("rolled_dice: {rolled_dice:?}");
                prop_assert!(possible_rolls.contains(&rolled_dice));
            }
        });
    }
}
