use crate::{factorial, is_sorted_by, is_total_order_by, num_multisets};
use approx::relative_eq;
use std::{
    cmp, fmt,
    iter::FusedIterator,
    ops::{AddAssign, Range},
    str::FromStr,
};
use tinyvec::ArrayVec;

#[cfg(test)]
use proptest::{
    arbitrary::Arbitrary,
    strategy::{BoxedStrategy, Strategy},
};

// TODO(philiphayes): implement jokers/devils
// const FACE_JOKER: u8 = 0;

pub struct DieDistr([f64; 8]);

impl DieDistr {
    const fn new(distr: [f64; 6]) -> Self {
        Self([
            0.0, distr[0], distr[1], distr[2], distr[3], distr[4], distr[5], 0.0,
        ])
    }

    // const fn new_joker(distr: [f64; 6]) -> Self {
    //     Self([
    //         distr[0], 0.0, distr[1], distr[2], distr[3], distr[4], distr[5], 0.0,
    //     ])
    // }

    const fn p_face(&self, face: u8) -> f64 {
        self.0[face as usize]
    }
}

#[repr(u8)]
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DieKind {
    Standard,
    HeavenlyKingdomDie,
    OddDie,
    // ..
}

impl Default for DieKind {
    fn default() -> Self {
        Self::Standard
    }
}

impl DieKind {
    fn from_memnonic(s: &str) -> Option<Self> {
        let kind = match s {
            "" | "s" => Self::Standard,
            "hk" => Self::HeavenlyKingdomDie,
            "o" => Self::OddDie,
            _ => return None,
        };
        Some(kind)
    }

    fn as_memnonic(self) -> &'static str {
        match self {
            Self::Standard => "",
            Self::HeavenlyKingdomDie => "hk",
            Self::OddDie => "o",
        }
    }

    fn die_distr(self) -> DieDistr {
        // not all dice probabilities on the wiki add up to 1.0 exactly, so
        // without better info, I'm going to slightly bias the dice against 1
        // and 5 (the best rolls) to make everything add up : )

        match self {
            Self::Standard => DieDistr::new([1.0 / 6.0; 6]),
            Self::HeavenlyKingdomDie => DieDistr::new([0.368, 0.105, 0.105, 0.105, 0.105, 0.212]),
            Self::OddDie => DieDistr::new([
                4.0 / 15.0,
                1.0 / 15.0,
                4.0 / 15.0,
                1.0 / 15.0,
                4.0 / 15.0,
                1.0 / 15.0,
            ]),
        }
    }
}

impl fmt::Display for DieKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_memnonic())
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct DieKindCounts(ArrayVec<[(DieKind, u8); 6]>);

impl DieKindCounts {
    pub fn new() -> Self {
        Self(ArrayVec::new())
    }

    pub fn all_standard(ndice: u8) -> Self {
        let mut arr = ArrayVec::new();
        arr.push((DieKind::Standard, ndice));
        Self(arr)
    }

    pub fn from_dice_vec(dice: DiceVec) -> Self {
        let counts = Self(
            dice.group_by_die_kind()
                .map(|(kind, kind_dice)| (kind, kind_dice.len()))
                .collect(),
        );
        debug_assert!(counts.invariant());
        counts
    }

    pub fn validate_init_set(&self, rolled_kinds: &DieKindCounts) -> Result<(), String> {
        if self.ndice() != 6 {
            return Err("initial die kinds set must contain exactly 6 dice".to_string());
        }
        if !self.is_superset_of(rolled_kinds) {
            return Err(format!(
                "rolled dice can only contain dice from the initial die kinds set: \
                 rolled kinds: '{}', initial kinds: '{}'",
                rolled_kinds, self
            ));
        }

        Ok(())
    }

    fn invariant(&self) -> bool {
        self.ndice() <= 6 && is_total_order_by(self.0.into_iter(), |x, y| Some(x.cmp(y)))
    }

    pub fn ndice(&self) -> u8 {
        self.0
            .into_iter()
            .map(|(_kind, kind_count)| kind_count)
            .sum()
    }

    fn get_count(&self, kind: DieKind) -> u8 {
        self.0
            .into_iter()
            .find_map(|(kind2, nkind2)| if kind2 == kind { Some(nkind2) } else { None })
            .unwrap_or(0)
    }

    fn spread_face(&self, face: u8) -> DiceVec {
        let mut dice = DiceVec::new();
        for (kind, nkind) in self.0.into_iter() {
            dice.push_n_die(Die::new(face, kind), nkind);
        }
        dice
    }

    fn contains(&self, kind: DieKind) -> bool {
        self.0.into_iter().any(|(kind2, _)| kind2 == kind)
    }

    fn add_count(&mut self, kind: DieKind, nkind: u8) {
        let idx = self.0.partition_point(|(other_kind, _)| other_kind < &kind);

        if let Some((kind2, nkind2)) = self.0.get_mut(idx) {
            if kind2 == &kind {
                nkind2.add_assign(nkind);
                return;
            }
        }

        self.0.insert(idx, (kind, nkind));
        debug_assert!(self.invariant());
    }

    fn sub_count(&mut self, kind: DieKind, nkind: u8) {
        use std::ops::SubAssign;
        let (_kind2, nkind2) = self
            .0
            .iter_mut()
            .find(|(kind2, _nkind2)| kind == *kind2)
            .unwrap();
        nkind2.sub_assign(nkind);
    }

    pub fn sub_counts(&mut self, other: Self) {
        debug_assert!(self.is_superset_of(&other));
        for (kind, nkind) in other.0.into_iter() {
            self.sub_count(kind, nkind);
        }
    }

    pub fn is_superset_of(&self, other: &Self) -> bool {
        other
            .0
            .into_iter()
            .all(|(other_kind, other_nkind)| self.get_count(other_kind) >= other_nkind)
    }

    fn num_multisets(&self, nfaces: u32) -> u32 {
        // (6 multichoose k1) * (6 multichoose k2) * ..
        self.0
            .into_iter()
            .map(|(_kind, nkind)| num_multisets(nfaces, nkind as u32))
            .product()
    }

    pub fn all_multisets(&self) -> Vec<DiceVec> {
        fn rec(
            cb: &mut impl FnMut(DiceVec),
            acc: DiceVec,
            kind_counts: DieKindCounts,
            current_face: u8,
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
                // generate `spread_dice multichoose nface` for this iteration
                for mset in spread_dice.multisets_iter(nface) {
                    let mut new_acc = acc;
                    new_acc.extend(mset);

                    let mut new_kind_counts = kind_counts;
                    new_kind_counts.sub_counts(DieKindCounts::from_dice_vec(mset));

                    rec(cb, new_acc, new_kind_counts, current_face + 1);
                }
            }
        }

        let mut out = Vec::with_capacity(self.num_multisets(6) as usize);
        rec(&mut |dice| out.push(dice), DiceVec::new(), *self, 1);
        out
    }
}

impl fmt::Display for DieKindCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use itertools::Itertools;
        let inner = self
            .0
            .into_iter()
            .map(|(kind, kind_count)| format!("{}:{}", kind.as_memnonic(), kind_count))
            .join(", ");
        write!(f, "[{}]", inner)
    }
}

impl FromIterator<(DieKind, u8)> for DieKindCounts {
    fn from_iter<T: IntoIterator<Item = (DieKind, u8)>>(iter: T) -> Self {
        let mut arr = ArrayVec::from_iter(iter.into_iter());
        arr.sort_unstable();

        let kinds = Self(arr);
        assert!(kinds.invariant());
        kinds
    }
}

// [s:3,hk:2,o:1]

impl FromStr for DieKindCounts {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut kinds = DieKindCounts::new();

        let s = s.trim_start_matches('[');
        let s = s.trim_end_matches(']');

        let splitters = &[',', ' ', '\n', '\t'];

        for kind_count_str in s.split(splitters).filter(|s| !s.is_empty()) {
            match kind_count_str.split_once(':') {
                Some((kind_str, count_str)) => {
                    let kind = DieKind::from_memnonic(kind_str)
                        .ok_or_else(|| format!("didn't recognize die kind: '{}'", kind_str))?;

                    if kinds.contains(kind) {
                        return Err(format!(
                            "the dice kinds set can't contain any duplicates: already contains kind: '{}'",
                            kind.as_memnonic(),
                        ));
                    }

                    let count = count_str.parse::<u8>().map_err(|err| {
                        format!("failed to parse die count: '{}', error: {}", count_str, err)
                    })?;

                    if !(1..=6).contains(&count) {
                        return Err(format!(
                            "die count needs to be in the range [1,6]: '{}'",
                            count
                        ));
                    }

                    kinds.add_count(kind, count);
                }
                None => return Err("".to_string()),
            }
        }

        Ok(kinds)
    }
}

#[derive(Copy, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Die {
    face: u8,
    kind: DieKind,
}

impl Die {
    fn new(face: u8, kind: DieKind) -> Self {
        Self { face, kind }
    }

    // fn p_die(self) -> f64 {
    //     self.kind.die_distr().p_face(self.face)
    // }
}

impl fmt::Debug for Die {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.face, self.kind)
    }
}

fn str_split_at_safe(s: &str, mid: usize) -> Option<(&str, &str)> {
    if s.is_char_boundary(mid) {
        Some(unsafe { (s.get_unchecked(0..mid), s.get_unchecked(mid..s.len())) })
    } else {
        None
    }
}

impl FromStr for Die {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        str_split_at_safe(s, 1)
            .and_then(|(first, rest)| {
                let face = first.parse::<u8>().ok()?;

                if (1..=6).contains(&face) {
                    let kind = DieKind::from_memnonic(rest)?;
                    Some(Die { face, kind })
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                format!(
                    "invalid die string '{}': expected format '<die-face><die-kind>'",
                    s
                )
            })
    }
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DiceVec(ArrayVec<[Die; 6]>);

impl DiceVec {
    /// A new empty list of dice.
    #[inline]
    pub fn new() -> Self {
        Self(ArrayVec::new())
    }

    // #[cfg(test)]
    // fn from_rolls(rolls: &[u8]) -> Self {
    //
    // }

    // #[inline]
    // pub fn from_die(die: Die) -> Self {
    //     let mut dice = Self::new();
    //     dice.0.push(die);
    //     dice
    // }

    #[inline]
    fn from_sorted_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Die>,
    {
        let mut dice = Self::new();
        for die in iter {
            dice.0.push(die);
        }
        debug_assert!(dice.invariant());
        dice
    }

    // #[inline]
    // pub fn is_empty(&self) -> bool {
    //     self.0.is_empty()
    // }

    #[inline]
    pub fn len(&self) -> u8 {
        self.0.len() as u8
    }

    fn invariant(&self) -> bool {
        is_sorted_by(self.0.iter(), |d1, d2| Some(d1.cmp(d2)))
    }

    // fn get_face_count(&self, face: u8) -> u8 {
    //     self.0
    //         .iter()
    //         .map(|die| if die.face == face { 1 } else { 0 })
    //         .sum()
    // }

    // fn get_die_count(&self, die: Die) -> u8 {
    //     self.0
    //         .iter()
    //         .map(|&other| if other == die { 1 } else { 0 })
    //         .sum()
    // }

    // #[inline]
    // fn get_die(self, idx: usize) -> Die {
    //     self.0[idx as usize]
    // }

    fn add_die(&mut self, die: Die) {
        let idx = self.0.partition_point(|&other| other < die);
        self.0.insert(idx, die);
        debug_assert!(self.invariant());
    }

    #[inline]
    fn push_die(&mut self, die: Die) {
        self.0.push(die);
        debug_assert!(self.invariant());
    }

    #[inline]
    fn push_n_die(&mut self, die: Die, n: u8) {
        for _ in 0..n {
            self.push_die(die);
        }
        debug_assert!(self.invariant());
    }

    #[inline]
    fn extend(&mut self, other: Self) {
        self.0.extend(other.into_iter());
        debug_assert!(self.invariant());
    }

    #[cfg(test)]
    fn merge(&self, other: &Self) -> Self {
        Self::from_sorted_iter(itertools::merge(self.into_iter(), other.into_iter()))
    }

    #[inline]
    #[cfg(test)]
    fn truncate(&mut self, new_len: u8) {
        self.0.truncate(new_len as usize);
    }

    // fn remove_face(self, face: u8) -> Self {
    //     let without_face =
    //         Self::from_sorted_iter(self.0.into_iter().filter(|die| die.face != face));
    //     debug_assert!(without_face.invariant());
    //     without_face
    // }

    #[cfg(test)]
    fn split_next_die(mut self) -> (Self, Self) {
        let die = self.0[0];
        let split_idx = self.0.partition_point(|&other| other == die);
        let other_set = self.0.split_off(split_idx);
        (self, Self(other_set))
    }

    /// Only return unique dice.
    #[cfg(test)]
    fn unique_dice(&self) -> Self {
        let (_prev, unique) = self.into_iter().fold(
            (Die::default(), DiceVec::new()),
            |(prev, mut unique), curr| {
                if prev != curr {
                    unique.0.push(curr);
                }
                (curr, unique)
            },
        );
        debug_assert!(unique.invariant());
        unique
    }

    #[inline]
    pub fn into_die_kind_counts(self) -> DieKindCounts {
        DieKindCounts::from_dice_vec(self)
    }

    fn die_kinds(&self) -> ArrayVec<[DieKind; 6]> {
        let mut kinds = ArrayVec::new();
        for die in self.into_iter() {
            if !kinds.contains(&die.kind) {
                kinds.push(die.kind);
            }
        }
        kinds.sort();
        kinds
    }

    fn die_with_kind(&self, kind: DieKind) -> DiceVec {
        Self::from_sorted_iter(self.into_iter().filter(|die| die.kind == kind))
    }

    fn group_by_die_kind(self) -> impl Iterator<Item = (DieKind, DiceVec)> {
        self.die_kinds()
            .into_iter()
            .map(move |kind| (kind, self.die_with_kind(kind)))
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
    pub fn p_roll(&self) -> f64 {
        self.group_by_die_kind()
            .map(|(kind, dice_kind)| {
                DiceCounts::from_dice_vec(dice_kind).p_roll_with_distr(kind.die_distr())
            })
            .product()
    }

    #[inline]
    pub fn multisets_iter(self, ndice: u8) -> DiceVecMultisetsIter {
        DiceVecMultisetsIter::new(self, ndice)
    }

    #[inline]
    fn copy_slice(&self, range: Range<u8>) -> Self {
        let start = range.start as usize;
        let end = range.end as usize;
        let len = end - start;

        let mut arr = ArrayVec::new();
        arr.grab_spare_slice_mut()[0..len].copy_from_slice(&self.0[start..end]);
        arr.set_len(len);

        let new = Self(arr);
        debug_assert!(new.invariant());

        new
    }
}

impl fmt::Debug for DiceVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use itertools::Itertools;
        let pieces = self.into_iter().map(|die| format!("{:?}", die)).join(", ");
        write!(f, "[{}]", pieces)
    }
}

impl IntoIterator for DiceVec {
    type Item = Die;
    type IntoIter = tinyvec::ArrayVecIterator<[Die; 6]>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl FromIterator<Die> for DiceVec {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Die>,
    {
        let mut dice = DiceVec::new();
        for die in iter.into_iter() {
            dice.add_die(die)
        }
        dice
    }
}

impl FromStr for DiceVec {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut dice = DiceVec::new();

        let s = s.trim_start_matches('[');
        let s = s.trim_end_matches(']');

        let splitters = &[',', ' ', '\n', '\t'];

        for die_str in s.split(splitters).filter(|s| !s.is_empty()) {
            if dice.len() >= 6 {
                return Err("too many dice".to_string());
            }

            let die = die_str.parse::<Die>()?;
            dice.add_die(die);
        }

        Ok(dice)
    }
}

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
            last: mset.copy_slice((n - k)..n),
            comb: mset.copy_slice(0..k),
            k,
            done: false,
        }
    }

    fn update_to_next_combination(&mut self) {
        // find the rightmost element in `comb` that is less than the "max" value
        // it can have (which is the same as the element in the final combination
        // at the same index)
        let maybe_next_idx = self
            .comb
            .0
            .iter()
            .zip(self.last.0.iter())
            .rposition(|(x, l)| x < l);

        // if comb == last, then there will be no rightmost element less than the
        // max, so we return None to show that we're done.
        let (i, x) = match maybe_next_idx {
            Some(i) => (i, self.comb.0[i]),
            None => {
                // comb == last => done!
                self.done = true;
                return;
            }
        };

        // find the successor element (the next element in the multiset greater
        // than arr[i])
        let maybe_succ_idx = self.mset.0.iter().position(|y| &x < y);

        let j = match maybe_succ_idx {
            Some(j) => j,
            // since we've already checked we're not at the end, it cannot happen
            // that there is no successor.
            None => unreachable!(),
        };

        // replaces comb[i] with its successor and the remainder with all elements
        // that follow in mset.
        let k = self.k as usize;
        self.comb.0[i..k].copy_from_slice(&self.mset.0[j..(j + (k - i))]);
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
pub struct DiceCounts(pub u32);

impl DiceCounts {
    /// A new empty set of dice rolls.
    #[inline]
    pub const fn new() -> Self {
        Self(0)
    }

    // #[inline]
    // const fn from_roll_count(roll: u8, count: u8) -> Self {
    //     Self(((count as u32) & 0x7) << (4 * (roll as u32)))
    // }

    #[inline]
    const fn from_counts(cs: [u8; 6]) -> Self {
        Self(u32::from_le_bytes([
            cs[0] << 4,
            cs[1] | (cs[2] << 4),
            cs[3] | (cs[4] << 4),
            cs[5],
        ]))
    }

    pub fn from_dice_vec(dice_vec: DiceVec) -> Self {
        dice_vec.into_iter().collect()
    }

    /// A convenience function for constructing a `DiceCounts` set from an unordered
    /// list of dice rolls.
    #[cfg(test)]
    pub fn from_rolls(rolls: &[u8]) -> Self {
        let mut counts_arr = [0u8; 7];
        for roll in rolls {
            counts_arr[*roll as usize] += 1;
        }

        let mut counts = Self::new();
        for roll in 1..=6 {
            counts.set_count(roll, counts_arr[roll as usize]);
        }
        counts
    }

    #[inline]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    // The number of dice in this set
    #[inline]
    pub fn len(self) -> u8 {
        let mut len = 0;
        for roll in 1..=6 {
            len += self.get_count(roll);
        }
        len
    }

    #[inline]
    pub fn get_count(self, roll: u8) -> u8 {
        ((self.0 >> (4 * (roll as u32))) & 0x7) as u8
    }

    #[inline]
    #[cfg(test)]
    pub fn set_count(&mut self, roll: u8, count: u8) {
        self.0 = (self.0 & !(0x7 << (4 * (roll as u32))))
            + (((count as u32) & 0x7) << (4 * (roll as u32)));
    }

    #[inline]
    pub fn add_count(&mut self, roll: u8, count: u8) {
        self.0 += ((count as u32) & 0x7) << (4 * (roll as u32));
    }

    #[cfg(test)]
    #[inline]
    const fn add(self, other: Self) -> Self {
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
            let c2 = self.get_count(2);
            let c3 = self.get_count(3);
            let c4 = self.get_count(4);
            let c6 = self.get_count(6);

            (c2 != 1 && c2 != 2)
                && (c3 != 1 && c3 != 2)
                && (c4 != 1 && c4 != 2)
                && (c6 != 1 && c6 != 2)
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

impl cmp::Ord for DiceCounts {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.to_rolls().cmp(&other.to_rolls())
    }
}

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
        let mut counts = DiceCounts::new();
        for die in iter.into_iter() {
            counts.add_count(die.face, 1);
        }
        counts
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
            .prop_map(|rolls| DiceCounts::from_rolls(&rolls))
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

///////////
// Tests //
///////////

#[cfg(test)]
mod test {
    use super::*;
    use crate::num_combinations;
    use approx::assert_relative_eq;
    use claim::assert_err;
    use proptest::prelude::*;
    use std::{cmp::min, collections::HashSet};

    macro_rules! dice {
        () => {
            DiceVec::new()
        };
        ($($x:tt),+ $(,)?) => {
            DiceVec::from_iter([ $( Die::from_str(stringify!($x)).unwrap() ),+ ].into_iter())
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
                for die in left.unique_dice() {
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
            let die = die_set.0[0];
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
        assert_eq!(0, DiceCounts::from_rolls(&[]).score());
        assert_eq!(100, DiceCounts::from_rolls(&[1]).score());
        assert_eq!(150, DiceCounts::from_rolls(&[5, 1]).score());
        assert_eq!(0, DiceCounts::from_rolls(&[2, 2, 3]).score());
        assert_eq!(1500, DiceCounts::from_rolls(&[6, 5, 4, 3, 2, 1]).score());
        assert_eq!(250, DiceCounts::from_rolls(&[1, 1, 5, 3, 2, 2]).score());
        assert_eq!(1050, DiceCounts::from_rolls(&[1, 1, 5, 3, 1, 2]).score());
        assert_eq!(400, DiceCounts::from_rolls(&[4, 4, 4]).score());
        assert_eq!(800, DiceCounts::from_rolls(&[4, 4, 4, 4]).score());
        assert_eq!(1600, DiceCounts::from_rolls(&[4, 4, 4, 4, 4]).score());
        assert_eq!(3200, DiceCounts::from_rolls(&[4, 4, 4, 4, 4, 4]).score());
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
        assert!(DiceCounts::from_rolls(&[]).is_empty());

        assert_eq!(0, DiceCounts::from_rolls(&[]).len());
        assert_eq!(1, DiceCounts::from_rolls(&[1]).len());
        assert_eq!(2, DiceCounts::from_rolls(&[4, 2]).len());
        assert_eq!(6, DiceCounts::from_rolls(&[6, 5, 4, 3, 2, 1]).len());

        assert_eq!(1, DiceCounts::from_rolls(&[6, 5, 4, 3, 2, 1]).get_count(6));
        assert_eq!(3, DiceCounts::from_rolls(&[6, 5, 3, 3, 3]).get_count(3));
        assert_eq!(0, DiceCounts::from_rolls(&[6, 5, 3, 3, 3]).get_count(1));
        assert_eq!(6, DiceCounts::from_rolls(&[3, 3, 3, 3, 3, 3]).get_count(3));
        assert_eq!(6, DiceCounts::from_rolls(&[6, 6, 6, 6, 6, 6]).get_count(6));
        assert_eq!(0, DiceCounts::from_rolls(&[6, 6, 6, 6, 6, 6]).get_count(3));
    }

    #[test]
    fn test_dice_counts_from_counts() {
        assert!(DiceCounts::from_counts([0; 6]).is_empty());

        assert_eq!(
            DiceCounts::from_counts([1; 6]),
            DiceCounts::from_rolls(&[1, 2, 3, 4, 5, 6]),
        );

        assert_eq!(
            DiceCounts::from_counts([1, 0, 1, 0, 1, 0]),
            DiceCounts::from_rolls(&[1, 3, 5]),
        );
        assert_eq!(
            DiceCounts::from_counts([1, 2, 0, 3, 0, 0]),
            DiceCounts::from_rolls(&[1, 2, 2, 4, 4, 4]),
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
            dice.truncate(ndice);

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

        let kind_counts = DieKindCounts::from_dice_vec(dice![1, 1, 1, 1hk, 1hk, 1o]);
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
            DiceCounts::from_rolls(&[1, 1]),
            DiceCounts::from_rolls(&[1, 3]),
            DiceCounts::from_rolls(&[1, 5]),
            DiceCounts::from_rolls(&[3, 5]),
        ];
        assert_eq!(
            &expected,
            &DiceCounts::from_rolls(&[1, 1, 3, 5]).multisets(2),
        );

        // when all the elements are unique, this is just the normal set
        // combinations
        for k in 0..=6 {
            assert_eq!(
                num_combinations(6, k) as usize,
                DiceCounts::from_rolls(&[1, 2, 3, 4, 5, 6])
                    .multisets(k as u8)
                    .len(),
            );
        }
    }

    #[test]
    fn test_dice_vec_from_str() {
        assert_eq!(dice![], DiceVec::from_str("").unwrap());
        assert_eq!(dice![], DiceVec::from_str("[]").unwrap());
        assert_eq!(
            dice![1, 2hk, 3o],
            DiceVec::from_str("[1, 2hk, 3o]").unwrap()
        );
    }

    #[test]
    fn test_dice_vec_unique_dice() {
        assert_eq!(dice![], dice![].unique_dice());
        assert_eq!(dice![1, 2hk, 3o], dice![1, 2hk, 3o].unique_dice());
        assert_eq!(
            dice![1, 2hk, 3o],
            dice![1, 1, 1, 2hk, 2hk, 3o].unique_dice()
        );
        assert_eq!(
            dice![1, 1hk, 1o],
            dice![1, 1, 1, 1hk, 1hk, 1o].unique_dice()
        );
    }

    #[test]
    fn test_dice_vec_merge() {
        assert_eq!(dice![], dice![].merge(&dice![]));
        assert_eq!(dice![1], dice![1].merge(&dice![]));
        assert_eq!(dice![1], dice![].merge(&dice![1]));
        assert_eq!(dice![1, 2hk, 3o], dice![2hk].merge(&dice![1, 3o]));
        assert_eq!(
            dice![1, 1, 2hk, 2hk, 3o, 3o],
            dice![2hk, 3o].merge(&dice![1, 1, 2hk, 3o])
        );
        assert_eq!(
            dice![1, 1, 1hk, 1hk, 1o, 1o],
            dice![1hk, 1o].merge(&dice![1, 1, 1hk, 1o])
        );
    }

    #[test]
    fn test_dice_vec_multisets() {
        use itertools::Itertools;
        assert_eq!(
            vec![dice![]],
            dice![1, 1, 1, 2hk, 2hk, 3o].multisets_iter(0).collect_vec(),
        );
        assert_eq!(
            vec![dice![1], dice![2hk], dice![3o]],
            dice![1, 1, 1, 2hk, 2hk, 3o].multisets_iter(1).collect_vec(),
        );
        assert_eq!(
            vec![dice![1], dice![1hk], dice![1o]],
            dice![1, 1, 1, 1hk, 1hk, 1o].multisets_iter(1).collect_vec(),
        );
        assert_eq!(
            vec![
                dice![1, 1],
                dice![1, 2hk],
                dice![1, 3o],
                dice![2hk, 2hk],
                dice![2hk, 3o]
            ],
            dice![1, 1, 1, 2hk, 2hk, 3o].multisets_iter(2).collect_vec(),
        );
        assert_eq!(
            vec![
                dice![1, 1, 1],
                dice![1, 1, 2hk],
                dice![1, 1, 3o],
                dice![1, 2hk, 2hk],
                dice![1, 2hk, 3o],
                dice![2hk, 2hk, 3o],
            ],
            dice![1, 1, 1, 2hk, 2hk, 3o].multisets_iter(3).collect_vec(),
        );
        assert_eq!(
            vec![dice![1, 1, 1, 2hk, 2hk, 3o]],
            dice![1, 1, 1, 2hk, 2hk, 3o].multisets_iter(6).collect_vec(),
        );

        assert_eq!(
            vec![
                dice![1, 1hk],
                dice![1, 2],
                dice![1, 2hk],
                dice![1, 3],
                dice![1, 3hk],
                dice![1hk, 2],
                dice![1hk, 2hk],
                dice![1hk, 3],
                dice![1hk, 3hk],
                dice![2, 2hk],
                dice![2, 3],
                dice![2, 3hk],
                dice![2hk, 3],
                dice![2hk, 3hk],
                dice![3, 3hk],
            ],
            dice![1, 1hk, 2, 2hk, 3, 3hk]
                .multisets_iter(2)
                .collect_vec(),
        );

        assert_eq!(
            vec![dice![1, 1], dice![1, 1o], dice![1o, 1o]],
            dice![1, 1, 1o, 1o, 1o, 1o].multisets_iter(2).collect_vec(),
        );

        assert_eq!(
            vec![
                dice![1, 1, 2, 2],
                dice![1, 1, 2, 3],
                dice![1, 2, 2, 2],
                dice![1, 2, 2, 3],
                dice![2, 2, 2, 3],
            ],
            dice![1, 1, 2, 2, 2, 3].multisets_iter(4).collect_vec(),
        );

        // when all the elements are unique, this is just the normal set
        // combinations
        for k in 0..=6 {
            assert_eq!(
                num_combinations(6, k as u32) as usize,
                dice![1, 1hk, 1o, 2, 2hk, 2o].multisets_iter(k).count(),
            );
        }
    }

    #[test]
    fn test_dice_counts_probability() {
        assert_relative_eq!(
            3.0 / ((6.0_f64).powf(3.0)),
            DiceCounts::from_rolls(&[1, 1, 3]).p_roll()
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
        assert_relative_eq!(
            dice![1, 1, 1, 2, 2, 3].p_roll(),
            DiceCounts::from_dice_vec(dice![1, 1, 1, 2, 2, 3]).p_roll(),
        );
        assert_relative_eq!(
            dice![1, 1, 2, 3].p_roll(),
            DiceCounts::from_dice_vec(dice![1, 1, 2, 3]).p_roll(),
        );

        use super::DieKind::{HeavenlyKingdomDie as hk, OddDie as o, Standard as s};

        // exhaustively check that all combinations of {s, hk, o} die kinds will
        // emit proper probability mass functions (where the pmf is effectively
        // { pmf(dice) := dice.p_roll()) | dice ∈ die_kinds.all_multisets() }

        let mut die_kind_combs = Vec::new();

        for ndice in 0..=6 {
            for ns in 0..=ndice {
                let nleft = ndice - ns;
                for nhk in 0..=nleft {
                    let no = nleft - nhk;

                    die_kind_combs.push(DieKindCounts::from_iter([(s, ns), (hk, nhk), (o, no)]))
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
                    .map(|dice| dice.p_roll())
                    .sum(),
                epsilon = 1.0e-10
            );
        }
    }

    #[test]
    fn test_die_kind_counts_from_str() {
        use super::DieKind::{HeavenlyKingdomDie as hk, OddDie as o, Standard as s};

        assert_err!(DieKindCounts::from_str("[s:0]"));
        assert_err!(DieKindCounts::from_str("[s:7]"));
        assert_err!(DieKindCounts::from_str("[hk:2,o:1,s:7]"));
        assert_err!(DieKindCounts::from_str("[s:1,o:1,s:1]"));
        assert_eq!(
            DieKindCounts::from_iter([(s, 6)]),
            DieKindCounts::from_str("[s:6]").unwrap(),
        );
        assert_eq!(
            DieKindCounts::from_iter([(s, 3), (hk, 2), (o, 1)]),
            DieKindCounts::from_str("[s:3,hk:2,o:1]").unwrap(),
        );
        assert_eq!(
            DieKindCounts::from_iter([(s, 3), (hk, 2), (o, 1)]),
            DieKindCounts::from_str("[o:1,hk:2,s:3]").unwrap(),
        );
        assert_eq!(
            DieKindCounts::from_iter([(s, 3), (hk, 2), (o, 1)]),
            DieKindCounts::from_str("[ s:3, hk:2, o:1 ]").unwrap(),
        );
        assert_eq!(
            DieKindCounts::from_iter([(s, 3), (hk, 2), (o, 1)]),
            DieKindCounts::from_str(" s:3, hk:2, o:1 ").unwrap(),
        );
    }
}
