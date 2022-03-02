use serde::{Deserialize, Serialize};

use crate::{
    dice::{self, DieKind, DieKindCounts, DieKindTable},
    search,
};
use std::{collections::BTreeMap, fmt, str::FromStr};

fn str_split_at_safe(s: &str, mid: usize) -> Option<(&str, &str)> {
    if s.is_char_boundary(mid) {
        Some(unsafe { (s.get_unchecked(0..mid), s.get_unchecked(mid..s.len())) })
    } else {
        None
    }
}

////////////////////
// parse::DiceSet //
////////////////////

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiceSet(BTreeMap<DieKind, u8>);

impl DiceSet {
    pub fn new() -> Self {
        Self(BTreeMap::new())
    }

    pub fn all_standard(ndice: u8) -> Self {
        Self(BTreeMap::from_iter([(DieKind::Standard, ndice)]))
    }

    fn ndice(&self) -> u8 {
        self.0.values().sum()
    }

    fn is_superset_of(&self, other: &Self) -> bool {
        other
            .0
            .iter()
            .all(|(o_kind, &o_count)| self.0.get(o_kind).copied().unwrap_or(0) >= o_count)
    }

    pub fn to_compact_form(&self) -> (DieKindTable, DieKindCounts) {
        let table = self.to_table();
        let counts = self.to_counts(&table);

        (table, counts)
    }

    pub fn to_table(&self) -> DieKindTable {
        let kinds = self.0.keys().copied().collect::<Vec<_>>();
        DieKindTable::from_die_kinds(&kinds)
    }

    pub fn to_counts(&self, table: &DieKindTable) -> DieKindCounts {
        self.0
            .iter()
            .map(|(&kind, &count)| {
                let kind_idx = table
                    .get_kind_idx(kind)
                    .expect("unknown DieKind not recognized in table");
                (kind_idx, count)
            })
            .collect()
    }

    pub fn validate_init_set(&self, rolled_set: &DiceSet) -> Result<(), String> {
        let ndice = self.ndice();
        if self.ndice() != 6 {
            return Err(format!(
                "initial die kinds set must contain exactly 6 dice: ndice: {}",
                ndice
            ));
        }
        if !self.is_superset_of(rolled_set) {
            return Err(
                format!("rolled dice can only contain dice from the initial die kinds set: init set: {}, rolled set: {}", self, rolled_set),
            );
        }

        Ok(())
    }
}

impl FromIterator<(DieKind, u8)> for DiceSet {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = (DieKind, u8)>,
    {
        Self(BTreeMap::from_iter(iter))
    }
}

impl FromStr for DiceSet {
    type Err = String;

    // [s:3,hk:2,o:1]

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut kinds = DiceSet::new();

        let s = s.trim_start_matches('[');
        let s = s.trim_end_matches(']');

        let splitters = &[',', ' ', '\n', '\t'];

        for kind_count_str in s.split(splitters).filter(|s| !s.is_empty()) {
            match kind_count_str.split_once(':') {
                Some((kind_str, count_str)) => {
                    let kind = DieKind::from_memnonic(kind_str)
                        .ok_or_else(|| format!("didn't recognize die kind: '{}'", kind_str))?;

                    if kinds.0.contains_key(&kind) {
                        return Err(format!(
                            "the dice set can't contain any duplicates: already contains kind: '{}'",
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

                    kinds.0.insert(kind, count);
                }
                None => return Err("".to_string()),
            }
        }

        let ndice = kinds.ndice();
        if ndice > 6 {
            return Err(format!(
                "too many dice in set! expected <= 6, got: {}",
                ndice
            ));
        }

        Ok(kinds)
    }
}

impl fmt::Display for DiceSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use itertools::Itertools;
        let pieces = self
            .0
            .iter()
            .map(|(kind, count)| format!("{}:{}", kind, count))
            .join(", ");
        write!(f, "[{}]", pieces)
    }
}

impl fmt::Debug for DiceSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

////////////////
// parse::Die //
////////////////

#[derive(Copy, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Die {
    face: u8,
    kind: DieKind,
}

impl Die {
    fn new(face: u8, kind: DieKind) -> Self {
        Self { face, kind }
    }

    fn into_compact_form(self, table: &DieKindTable) -> dice::Die {
        dice::Die::new(self.face, table.get_kind_idx(self.kind).unwrap())
    }

    fn from_compact_form(table: &DieKindTable, die: dice::Die) -> Self {
        Self::new(die.face(), table.get_kind(die.kind_idx()))
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
                    Some(Die::new(face, kind))
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

impl fmt::Display for Die {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.face, self.kind.as_memnonic())
    }
}

impl fmt::Debug for Die {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

////////////////////
// parse::DiceVec //
////////////////////

#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiceVec(Vec<Die>);

impl DiceVec {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn to_compact_form(&self, table: &DieKindTable) -> dice::DiceVec {
        let mut dice = self
            .0
            .iter()
            .map(|die| die.into_compact_form(table))
            .collect::<Vec<_>>();
        dice::DiceVec::from_slice(&mut dice)
    }

    pub fn from_compact_form(table: &DieKindTable, dice: dice::DiceVec) -> Self {
        Self(
            dice.into_iter_no_sentinel()
                .map(|die| Die::from_compact_form(table, die))
                .collect(),
        )
    }

    pub fn to_die_set(&self) -> DiceSet {
        use std::ops::AddAssign;
        let mut set = DiceSet::new();
        for die in &self.0 {
            let count = set.0.entry(die.kind).or_insert(0);
            count.add_assign(1);
        }
        set
    }

    pub fn to_string_clean(&self) -> String {
        use itertools::Itertools;
        self.0.iter().map(|die| format!("{}", die)).join(" ")
    }
}

impl FromStr for DiceVec {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut dice = Self::new();

        let s = s.trim_start_matches('[');
        let s = s.trim_end_matches(']');

        let splitters = &[',', ' ', '\n', '\t'];

        for die_str in s.split(splitters).filter(|s| !s.is_empty()) {
            if dice.0.len() >= 6 {
                return Err("too many dice".to_string());
            }

            let die = Die::from_str(die_str)?;
            dice.0.push(die);
        }

        Ok(dice)
    }
}

impl FromIterator<Die> for DiceVec {
    fn from_iter<T>(iter: T) -> Self
    where
        T: IntoIterator<Item = Die>,
    {
        Self(Vec::from_iter(iter))
    }
}

impl fmt::Display for DiceVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use itertools::Itertools;
        let pieces = self.0.iter().map(|die| format!("{}", die)).join(", ");
        write!(f, "[{}]", pieces)
    }
}

impl fmt::Debug for DiceVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

///////////////////
// parse::Action //
///////////////////

#[derive(Debug, Serialize, Deserialize)]
pub enum Action {
    Pass,
    Hold(DiceVec),
}

impl Action {
    pub fn from_compact_form(table: &DieKindTable, action: search::Action) -> Self {
        match action {
            search::Action::Pass => Self::Pass,
            search::Action::Roll(dice) => Self::Hold(DiceVec::from_compact_form(table, dice)),
        }
    }

    pub fn to_action_str(&self) -> &'static str {
        match self {
            Self::Pass => "Pass",
            Self::Hold(_) => "Hold",
        }
    }

    pub fn to_maybe_dice_str(&self) -> Option<String> {
        match self {
            Self::Pass => None,
            Self::Hold(dice) => Some(dice.to_string_clean()),
        }
    }

    pub fn to_string_parts(&self) -> (&'static str, Option<String>) {
        (self.to_action_str(), self.to_maybe_dice_str())
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (action_str, maybe_dice_str) = self.to_string_parts();
        f.write_str(action_str)?;
        if let Some(dice_str) = maybe_dice_str {
            f.write_str(" ")?;
            f.write_str(&dice_str)
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use claim::assert_err;

    #[test]
    fn test_dice_set_from_str() {
        use crate::dice::DieKind::{HeavenlyKingdomDie as hk, OddDie as o, Standard as s};

        assert_err!(DiceSet::from_str("[s:0]"));
        assert_err!(DiceSet::from_str("[s:7]"));
        assert_err!(DiceSet::from_str("[hk:2,o:1,s:7]"));
        assert_err!(DiceSet::from_str("[s:1,o:1,s:1]"));
        assert_eq!(
            DiceSet::from_iter([(s, 6)]),
            DiceSet::from_str("[s:6]").unwrap(),
        );
        assert_eq!(
            DiceSet::from_iter([(s, 3), (hk, 2), (o, 1)]),
            DiceSet::from_str("[s:3,hk:2,o:1]").unwrap(),
        );
        assert_eq!(
            DiceSet::from_iter([(s, 3), (hk, 2), (o, 1)]),
            DiceSet::from_str("[o:1,hk:2,s:3]").unwrap(),
        );
        assert_eq!(
            DiceSet::from_iter([(s, 3), (hk, 2), (o, 1)]),
            DiceSet::from_str("[ s:3, hk:2, o:1 ]").unwrap(),
        );
        assert_eq!(
            DiceSet::from_iter([(s, 3), (hk, 2), (o, 1)]),
            DiceSet::from_str(" s:3, hk:2, o:1 ").unwrap(),
        );
    }

    #[test]
    fn test_dice_vec_from_str() {
        use crate::dice::DieKind::{HeavenlyKingdomDie as hk, OddDie as o, Standard as s};

        assert_eq!(DiceVec(vec![]), DiceVec::from_str("").unwrap());
        assert_eq!(DiceVec(vec![]), DiceVec::from_str("[]").unwrap());
        assert_eq!(
            DiceVec(vec![Die::new(1, s), Die::new(2, hk), Die::new(3, o)]),
            DiceVec::from_str("[1, 2hk, 3o]").unwrap(),
        );
    }
}
