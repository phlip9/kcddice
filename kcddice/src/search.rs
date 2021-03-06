use crate::{
    dice::{DiceVec, DieKindCounts, DieKindTable},
    stats::p_rv1_le_rv2,
    total_cmp_f64, TotalSize,
};
use bytesize::ByteSize;
use ndarray::{s, Array1, Array2, ArrayView1};
use std::{borrow::Borrow, cell::Cell, cmp, collections::HashMap, hash::Hash, rc::Rc};

///////////
// Cache //
///////////

pub struct Cache<K, V> {
    store: HashMap<K, V>,
    hits: u32,
    misses: u32,
}

impl<K, V> Cache<K, V>
where
    K: Eq + Hash,
    V: Clone,
{
    fn new() -> Self {
        Self {
            store: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    fn peek_cache<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: Eq + Hash,
    {
        let out = self.store.get(key).cloned();
        if out.is_some() {
            self.hits += 1;
        } else {
            self.misses += 1;
        }
        out
    }

    fn fill_cache(&mut self, key: K, value: V) -> V {
        let out = value.clone();
        self.store.insert(key, value);
        out
    }

    pub fn cache_size(&self) -> usize {
        self.store.len()
    }

    pub fn cache_hits(&self) -> u32 {
        self.hits
    }

    pub fn cache_misses(&self) -> u32 {
        self.misses
    }

    pub fn cache_hit_rate(&self) -> f32 {
        let total_queries = self.hits + self.misses;
        if total_queries == 0 {
            0.0
        } else {
            (self.hits as f32) / (total_queries as f32)
        }
    }
}

impl<K, V> TotalSize for Cache<K, V>
where
    K: TotalSize,
    V: TotalSize,
{
    fn total_size(&self) -> usize {
        self.store.total_size() + self.hits.total_size() + self.misses.total_size()
    }
}

impl<K, V> Cache<K, V>
where
    K: TotalSize,
    V: TotalSize,
{
    pub fn cache_size_bytes(&self) -> ByteSize {
        ByteSize(self.total_size() as u64)
    }
}

////////////
// Action //
////////////

/// An `Action` represents any of the possible actions the player can take from a
/// given round [`State`]. Note that a "bust" is simply the absence of any possible
/// actions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Action {
    /// If there is at least one scoring die, the player can pass their turn,
    /// ending their turn and scoring their current round total + the currently
    /// rolled dice score.
    Pass,
    // TODO(philiphayes): rename to Hold?
    /// If there is at least one scoring die and at least 2 dice left, the player
    /// can choose to hold some non-empty subset of scoring dice and re-roll the
    /// rest. The held dice score is added to their current round total.
    Roll(DiceVec),
}

impl_total_size_static!(Action);

////////////////////////
// Evaluation Context //
////////////////////////

/// The `Context` is some additional evaluation state passed along while evaluating
/// `(state, action) -> expected value`. The evaluation should still be correct
/// even without the `Context`; rather, the `Context` should be used only for
/// optimizations or evaluation statistics.
pub struct Context {
    /// A mapping from kind index to DieKind
    kind_table: DieKindTable,
    /// All of our dice
    all_dice: DieKindCounts,
    /// A cache from normalized (State, Action) pairs to their expected value
    /// after evaluation.
    action_value_cache: Cache<NormalizedStateAction, f64>,
    score_distr_cache: Cache<NormalizedStateAction, ScorePMF>,
    actions_cache: Cache<DiceVec, Rc<Vec<Action>>>,
    actions_explored: u64,
    game_finished_prunes: Cell<u64>,
}

impl Context {
    pub fn new(kind_table: DieKindTable, all_dice: DieKindCounts) -> Self {
        Context {
            all_dice,
            kind_table,
            game_finished_prunes: Cell::new(0),
            action_value_cache: Cache::new(),
            score_distr_cache: Cache::new(),
            actions_cache: Cache::new(),
            actions_explored: 0,
        }
    }

    #[inline]
    pub fn actions_explored(&self) -> u64 {
        self.actions_explored
    }

    #[inline]
    fn inc_actions_explored(&mut self) {
        self.actions_explored += 1;
    }

    #[inline]
    fn game_finished_prunes(&self) -> u64 {
        self.game_finished_prunes.get()
    }

    #[inline]
    fn should_prune(&self, my_round_total: u16, target_score: u16) -> bool {
        if my_round_total >= target_score {
            self.game_finished_prunes
                .set(self.game_finished_prunes.get() + 1);
            true
        } else {
            false
        }
    }

    fn action_value_cache_mut(&mut self) -> &mut Cache<NormalizedStateAction, f64> {
        &mut self.action_value_cache
    }

    pub fn action_value_cache(&self) -> &Cache<NormalizedStateAction, f64> {
        &self.action_value_cache
    }

    fn score_distr_cache_mut(&mut self) -> &mut Cache<NormalizedStateAction, ScorePMF> {
        &mut self.score_distr_cache
    }

    pub fn score_distr_cache(&self) -> &Cache<NormalizedStateAction, ScorePMF> {
        &self.score_distr_cache
    }

    fn actions_cache_mut(&mut self) -> &mut Cache<DiceVec, Rc<Vec<Action>>> {
        &mut self.actions_cache
    }

    pub fn actions_cache(&self) -> &Cache<DiceVec, Rc<Vec<Action>>> {
        &self.actions_cache
    }

    #[inline]
    pub fn game_finished_rate(&self) -> f64 {
        (self.game_finished_prunes() as f64) / (self.actions_explored() as f64 + 1.0)
    }
}

////////////////////////
// Normalized Q-State //
////////////////////////

/// A normalized representation of a q-state / `(State, Action)` pair.
///
/// insight:  peek_cache(State {200, [1,1,5,X1,X2,X3]}, Roll[1,1,5])
///       and peek_cache(State {200, [1,1,5,Y1,Y2,Y3]}, Roll[1,1,5]) should hit
///       the same cache slot as long as all Xi and Yi aren't used by the action.
///
/// option 1: normalize the state representation only so _always_ non-scoring
///           dice get lumped into the 0-bucket and only affect the length.
///
/// option 2: normalize the (state, action) representation
///             (State {200, [1,1,5,2,2,3]}, Roll[1,1,5])
///           ??? StateAction { round_total=450, ndice_left=3 }
///
/// `NormalizedStateAction` is option 2.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct NormalizedStateAction {
    /// The round total after applying the action.
    pub my_round_total: u16,
    /// The total score we're trying to hit. If we meet or exceed this value,
    /// then we win.
    pub target_score: u16,
    /// The set of dice available to reroll after applying the action.
    pub dice_left: DieKindCounts,
}

impl_total_size_static!(NormalizedStateAction);

impl NormalizedStateAction {
    pub fn new(my_round_total: u16, target_score: u16, dice_left: DieKindCounts) -> Self {
        Self {
            my_round_total,
            target_score,
            dice_left,
        }
    }

    /// A pseudo initial state, before rolling the first set of dice.
    #[allow(unused)]
    pub(crate) fn init_state(target_score: u16, dice_kinds: DieKindCounts) -> Self {
        Self {
            my_round_total: 0,
            target_score,
            dice_left: dice_kinds,
        }
    }

    #[allow(unused)]
    fn from_state_action(all_dice: DieKindCounts, state: State, action: Action) -> Self {
        match action {
            Action::Pass => Self {
                my_round_total: state.my_round_total + state.rolled_dice.score(),
                target_score: state.target_score,
                dice_left: DieKindCounts::new(),
            },
            Action::Roll(held_dice) => Self::from_state_roll_action(all_dice, state, held_dice),
        }
    }

    #[inline]
    pub(crate) fn from_state_roll_action(
        all_dice: DieKindCounts,
        state: State,
        held_dice: DiceVec,
    ) -> Self {
        let dice_left = state.dice_left_after_hold(all_dice, held_dice);

        Self {
            // fold the held dice score into the round total
            my_round_total: state.my_round_total + held_dice.exact_score(),
            target_score: state.target_score,
            dice_left,
        }
    }

    #[inline]
    pub(crate) fn into_state(self, rolled_dice: DiceVec) -> State {
        State {
            my_round_total: self.my_round_total,
            target_score: self.target_score,
            rolled_dice,
        }
    }

    #[inline]
    fn is_pass(self) -> bool {
        self.dice_left.ndice() == 0
    }

    /// Return an `Iterator` over all possible states reachable after rolling the
    /// `ndice_left`, along with the conditional probability of reaching each
    /// state.
    #[inline]
    fn possible_roll_states(
        self,
        kind_table: &DieKindTable,
    ) -> impl Iterator<Item = (State, f64)> + '_ {
        self.dice_left
            .all_multisets()
            .into_iter()
            .map(move |next_roll| (self.into_state(next_roll), next_roll.p_roll(kind_table)))
    }

    /// Given a normalized `(State, Action)` pair, evaluate the expected value
    /// of applying the `Action` to the `State`, assuming we always choose
    /// subsequent actions by maximum expected value.
    pub fn expected_value(self, ctxt: &mut Context) -> f64 {
        ctxt.inc_actions_explored();

        if self.is_pass() {
            // expected value := P=1.0 * my_round_total
            return cmp::min(self.target_score, self.my_round_total) as f64;
        }

        // check if the cache already contains this normalized (State, Action)
        // pair.
        if let Some(action_value) = ctxt.action_value_cache_mut().peek_cache(&self) {
            return action_value;
        }

        // prune deep paths, low joint probability paths, and post-game finish
        // paths. if we pass any limit, just pretend this action always busts.
        if ctxt.should_prune(self.my_round_total, self.target_score) {
            return 0.0;
        }

        let mut expected_value = 0.0_f64;

        let kind_table = ctxt.kind_table;

        // for all possible dice rolls
        for (next_state, p_roll) in self.possible_roll_states(&kind_table) {
            // want to maximize expected value; choose action with
            // greatest expected value
            let best_action_value = next_state
                .actions_cached(ctxt)
                .iter()
                .map(|&next_action| {
                    Self::from_state_action(ctxt.all_dice, next_state, next_action)
                        .expected_value(ctxt)
                })
                .max_by(total_cmp_f64)
                .unwrap_or(0.0);

            expected_value += p_roll * best_action_value;
        }

        ctxt.action_value_cache_mut()
            .fill_cache(self, expected_value)
    }

    pub fn score_distribution(self, ctxt: &mut Context) -> ScorePMF {
        ctxt.inc_actions_explored();

        if self.is_pass() {
            let score = cmp::min(self.target_score, self.my_round_total);
            return ScorePMF::constant(score);
        }

        if let Some(pmf) = ctxt.score_distr_cache_mut().peek_cache(&self) {
            return pmf;
        }

        if ctxt.should_prune(self.my_round_total, self.target_score) {
            return ScorePMF::bust();
        }

        let mut score_pmf = ScorePMF::new();

        let kind_table = ctxt.kind_table;

        for (next_state, p_roll) in self.possible_roll_states(&kind_table) {
            let (_best_exp_value, best_score_distr) = next_state
                .actions_cached(ctxt)
                .iter()
                .map(|&next_action| {
                    let distr = Self::from_state_action(ctxt.all_dice, next_state, next_action)
                        .score_distribution(ctxt);
                    (distr.expected_value(), distr)
                })
                .max_by(|(v1, _), (v2, _)| total_cmp_f64(v1, v2))
                .unwrap_or_else(|| (0.0, ScorePMF::bust()));

            score_pmf.add_conditional_distr(p_roll, &best_score_distr);
        }

        ctxt.score_distr_cache_mut().fill_cache(self, score_pmf)
    }
}

/////////////////
// Round State //
/////////////////

/// A representation of the player's round state.
#[derive(Copy, Clone, Debug)]
pub struct State {
    /// The set of dice the player just rolled.
    pub rolled_dice: DiceVec,
    /// The player's current round score (accumulated from previous rolls in the turn).
    pub my_round_total: u16,
    /// The total score we're trying to hit. If we meet or exceed this value,
    /// then we win.
    pub target_score: u16,
}

impl State {
    pub fn new(my_round_total: u16, target_score: u16, rolled_dice: DiceVec) -> Self {
        Self {
            rolled_dice,
            my_round_total,
            target_score,
        }
    }

    pub(crate) fn actions(&self) -> Vec<Action> {
        if self.rolled_dice.is_bust() {
            // if this dice roll has no scores whatsoever, then there are no actions
            // (our turn has ended).
            Vec::new()
        } else {
            // we _can_ in fact hold all the dice, but they must all be scoring dice.
            let max_num_holds = self.rolled_dice.len();

            // TODO(philiphayes): this is probably no longer true with multiple
            // different die kinds.

            // for now just do the dumb thing and try all possible non-bust holds
            let possible_holds = (1..=max_num_holds)
                .flat_map(|ndice| self.rolled_dice.multisets_iter(ndice))
                .filter_map(|held_dice| {
                    if held_dice.is_valid_hold() {
                        Some(Action::Roll(held_dice))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();

            let mut actions_vec = possible_holds;

            // can always pass if we have some scores
            actions_vec.push(Action::Pass);

            actions_vec
        }
    }

    /// From the current round `State` return the complete set of possible
    /// `Action`s the player can take. Returns an empty set if the player has
    /// "busted".
    fn actions_cached(&self, ctxt: &mut Context) -> Rc<Vec<Action>> {
        // Luckily, the set of possible actions generated from a `State` is only
        // dependent on the currently rolled dice.
        if let Some(actions) = ctxt.actions_cache_mut().peek_cache(&self.rolled_dice) {
            return actions;
        }

        ctxt.actions_cache_mut()
            .fill_cache(self.rolled_dice, Rc::new(self.actions()))
    }

    /// Evaluate the expected value of applying this `Action` to the `State`,
    /// assuming we always choose subsequent actions by maximum expected value.
    #[inline]
    fn action_expected_value(&self, ctxt: &mut Context, action: Action) -> f64 {
        NormalizedStateAction::from_state_action(ctxt.all_dice, *self, action).expected_value(ctxt)
    }

    fn dice_left_after_hold(&self, all_dice: DieKindCounts, held_dice: DiceVec) -> DieKindCounts {
        let ndice_left = self.rolled_dice.len() - held_dice.len();

        if ndice_left == 0 {
            all_dice
        } else {
            let held_kinds = held_dice.into_die_kind_counts();
            let mut rolled_kinds = self.rolled_dice.into_die_kind_counts();

            rolled_kinds.sub_counts(held_kinds);
            rolled_kinds
        }
    }

    /// Evaluate the probability of "busting" immediately after applying the
    /// given `Action` to the current turn `State`.
    ///
    /// Fun fact: 3% chance to bust on your first roll : )
    fn action_p_bust(&self, ctxt: &Context, action: Action) -> f64 {
        match action {
            Action::Pass => 0.0,
            Action::Roll(held_dice) => {
                let available_kinds = self.dice_left_after_hold(ctxt.all_dice, held_dice);

                let mut p_bust = 0.0_f64;
                for next_roll in available_kinds.all_multisets() {
                    let p_roll = next_roll.p_roll(&ctxt.kind_table);

                    if next_roll.is_bust() {
                        p_bust += p_roll
                    }
                }
                p_bust
            }
        }
    }

    /// For each possible `Action` from this `State`, conditioned on choosing that
    /// action, what is the expected turn score and bust probability?
    pub fn actions_by_expected_value(&self, ctxt: &mut Context) -> Vec<ActionValue> {
        let mut actions_values = self
            .actions_cached(ctxt)
            .iter()
            .map(|&action| ActionValue {
                action,
                expected_value: self.action_expected_value(ctxt, action),
                p_bust: self.action_p_bust(ctxt, action),
            })
            .collect::<Vec<_>>();

        // sort by the expected turn score from highest to lowest.
        actions_values.sort_unstable_by(|av1, av2| {
            total_cmp_f64(&av1.expected_value, &av2.expected_value).reverse()
        });
        actions_values
    }
}

#[derive(Clone)]
pub struct ActionValue {
    pub action: Action,
    pub expected_value: f64,
    pub p_bust: f64,
}

impl ActionValue {
    pub fn to_row_cells(&self, dice_table: &DieKindTable) -> Vec<String> {
        let (action_str, dice_str) = match self.action {
            Action::Pass => ("pass", String::new()),
            Action::Roll(held_dice) => (
                "hold dice",
                format!(
                    "{}",
                    crate::parse::DiceVec::from_compact_form(dice_table, held_dice)
                ),
            ),
        };
        vec![
            action_str.to_owned(),
            dice_str,
            format!("{:0.1}", self.expected_value),
            format!("{:0.2}", self.p_bust),
        ]
    }
}

// HACK: can't seem to get around sycamore requiring PartialEq+Eq impl'd even
// though i'm using a keyed iterator???
impl cmp::PartialEq for ActionValue {
    fn eq(&self, other: &Self) -> bool {
        self.action == other.action
            && total_cmp_f64(&self.expected_value, &other.expected_value).is_eq()
            && total_cmp_f64(&self.p_bust, &other.p_bust).is_eq()
    }
}

impl cmp::Eq for ActionValue {}

///////////////
// Score PMF //
///////////////

#[derive(Clone, Debug)]
pub struct ScorePMF(HashMap<u16, f64>);

impl ScorePMF {
    #[inline]
    pub fn new() -> Self {
        Self(HashMap::new())
    }

    #[inline]
    pub fn constant(score: u16) -> Self {
        let mut pmf = HashMap::new();
        pmf.insert(score, 1.0);
        Self(pmf)
    }

    #[inline]
    pub fn bust() -> Self {
        Self::constant(0)
    }

    #[cfg(test)]
    pub fn from_counts_iter(
        num_trials: usize,
        counts: impl IntoIterator<Item = (u16, usize)>,
    ) -> Self {
        let n = num_trials as f64;
        Self(
            counts
                .into_iter()
                .map(|(score, count)| (score, (count as f64) / n))
                .collect(),
        )
    }

    pub fn into_vec(self) -> Vec<(u16, f64)> {
        self.0.into_iter().collect()
    }

    pub(crate) fn to_dense(&self, target_score: u16) -> Array1<f64> {
        let num_states = MarkovMatrix::num_states(target_score);

        let mut dense_pmf = Array1::zeros(num_states);

        for state_idx in 0..num_states {
            dense_pmf[state_idx] = self
                .0
                .get(&MarkovMatrix::i2s(state_idx))
                .copied()
                .unwrap_or(0.0);
        }

        dense_pmf
    }

    pub fn expected_value(&self) -> f64 {
        self.0
            .iter()
            .map(|(&score, &p_score)| p_score * (score as f64))
            .sum()
    }

    pub fn variance(&self) -> f64 {
        // Var(X) = E[(X - E[X])^2]
        //        = sum_x (Pr[X = x] * (x - E[X])^2)
        // mu = E[X]
        let mu = self.expected_value();

        self.0
            .iter()
            .map(|(&score, &p_score)| {
                let diff = (score as f64) - mu;
                p_score * diff * diff
            })
            .sum()
    }

    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn add_conditional_distr(&mut self, p_cond: f64, cond_distr: &Self) {
        use std::ops::AddAssign;
        for (&score, &p_score) in &cond_distr.0 {
            self.0
                .entry(score)
                .or_insert(0.0)
                .add_assign(p_cond * p_score);
        }
    }

    pub fn total_mass(&self) -> f64 {
        self.0.values().sum()
    }

    #[cfg(test)]
    pub fn compare_pmfs(&self, other: &Self, target_score: u16) {
        use crate::stats::kl_div_term;
        use ndarray::Zip;

        println!();
        println!("E[X_1] = {:<}", self.expected_value());
        println!("E[X_2] = {:<}", other.expected_value());
        println!();
        println!("stddev(X_1) = {:<}", self.stddev());
        println!("stddev(X_2) = {:<}", other.stddev());
        println!();
        println!("??? pmf1 = {:<}", self.total_mass());
        println!("??? pmf2 = {:<}", other.total_mass());
        println!();

        let mut table = tabular::Table::new("{:>}  {:<}  {:<}  {:<}  {:<}  {:<}");
        table.add_row(tabular::row!(
            "score",
            "pmf1",
            "pmf2",
            "|pmf1-pmf2|",
            "(pmf1-pmf2)^2",
            "pmf1 * ln(pmf1/pmf2)"
        ));

        let pmf1 = self.to_dense(target_score);
        let pmf2 = other.to_dense(target_score);
        assert_eq!(pmf1.len(), pmf2.len());

        let err = &pmf1 - &pmf2;
        let l1_diff = err.mapv(|x| x.abs());
        let err_sq = err.mapv(|x| x * x);
        let kl_terms = Zip::from(pmf1.view())
            .and(pmf2.view())
            .map_collect(|&p1_i, &p2_i| kl_div_term(p1_i, p2_i));

        for idx in 0..pmf1.len() {
            let score = MarkovMatrix::i2s(idx);
            let p_score_1 = pmf1[idx];
            let p_score_2 = pmf2[idx];
            let l1_diff = l1_diff[idx];
            let err_sq = err_sq[idx];
            let kl_term = kl_terms[idx];
            table.add_row(tabular::row!(
                score, p_score_1, p_score_2, l1_diff, err_sq, kl_term
            ));
        }

        println!(
            "max_x |p1(x)-p2(x)| = {:<}",
            l1_diff.iter().max_by(|x, y| total_cmp_f64(x, y)).unwrap()
        );
        println!("E[|pmf1 - pmf2|] = {:<}", l1_diff.mean().unwrap());
        println!("??? E[(pmf1 - pmf2)^2] = {:<}", err_sq.mean().unwrap().sqrt());
        println!("D_KL(pmf1 || pmf2) = {:<}", kl_terms.sum());

        println!("\n{table}");
    }
}

impl Default for ScorePMF {
    fn default() -> Self {
        Self::new()
    }
}

impl TotalSize for ScorePMF {
    fn total_size(&self) -> usize {
        self.0.total_size()
    }
}

#[derive(Debug)]
pub struct MarkovMatrix(Array2<f64>);

impl MarkovMatrix {
    /// score to state index
    #[inline]
    pub(crate) const fn s2i(score: u16) -> usize {
        (score / 50) as usize
    }

    /// state index to score
    #[inline]
    pub(crate) const fn i2s(state_idx: usize) -> u16 {
        (state_idx * 50) as u16
    }

    #[inline]
    pub(crate) const fn num_states(target_score: u16) -> usize {
        Self::s2i(target_score) + 1
    }

    pub fn from_optimal_policy(
        kind_table: DieKindTable,
        all_dice: DieKindCounts,
        target_score: u16,
    ) -> Self {
        let num_states = Self::num_states(target_score);

        let mut matrix = Array2::zeros((num_states, num_states));

        // compute the score distribution for each possible intermediate accumulated
        // turn score, then assemble into a matrix where each column x_i is the
        // score distribution pmf starting with turn score i*50.
        for turn_score_idx in Self::s2i(0)..Self::s2i(target_score) {
            let turn_score = Self::i2s(turn_score_idx);

            let mut ctxt = Context::new(kind_table, all_dice);
            let qstate = NormalizedStateAction::new(0, target_score - turn_score, all_dice);

            let score_pmf = qstate.score_distribution(&mut ctxt).to_dense(target_score);

            matrix
                .slice_mut(s![turn_score_idx.., turn_score_idx])
                .assign(&score_pmf.slice(s![..num_states - turn_score_idx]));
        }

        // remember to set the absorber column [0 0 .. 1] last (if we've won the
        // game, we stay in the win state forever).
        matrix[[num_states - 1, num_states - 1]] = 1.0;

        Self(matrix)
    }

    pub fn turns_to_win_cdf(&self, max_num_turns: usize) -> Array1<f64> {
        let n = self.0.nrows();
        let mut accum = Array2::<f64>::eye(n);

        let mut turns_cdf = Array1::<f64>::zeros(max_num_turns);

        for turn in 0..max_num_turns {
            // A^i+1 = A^i * A
            accum = accum.dot(&self.0);
            // P[win on turn i] = (A^i ?? [1 0 .. 0]^T)_{n-1} = (A^i)_{n-1,0}
            turns_cdf[turn] = accum[[n - 1, 0]];
        }

        turns_cdf
    }
}

#[cfg(test)]
pub mod prop {
    use super::*;
    use proptest::prelude::*;
    use std::ops::RangeInclusive;

    pub fn arb_score(range: impl Into<RangeInclusive<u16>>) -> impl Strategy<Value = u16> {
        let range = range.into();

        let idx_start = MarkovMatrix::s2i(*range.start());
        let idx_end = MarkovMatrix::s2i(*range.end());

        (idx_start..=idx_end).prop_map(MarkovMatrix::i2s)
    }
}

///////////
// Tests //
///////////

// #[cfg(test)]
// mod test {
//     use super::*;
//     use crate::{dice::Die, DEFAULT_TARGET_SCORE};
//     use approx::assert_relative_eq;
//     use std::str::FromStr;
//
//     #[test]
//     fn test_gen_actions() {
//         fn actions(rolled_dice: DiceVec) -> Vec<Action> {
//             let state = State::new(0, DEFAULT_TARGET_SCORE, rolled_dice);
//             let mut actions = state.actions();
//             actions.sort_unstable();
//             actions
//         }
//
//         macro_rules! dice {
//             () => {
//                 DiceVec::new()
//             };
//             ($($x:tt),+ $(,)?) => {
//                 DiceVec::from_iter([ $( Die::from_str(stringify!($x)).unwrap() ),+ ].into_iter())
//             };
//         }
//
//         // no scoring dice
//         assert_eq!(Vec::<Action>::new(), actions(dice![]));
//         assert_eq!(Vec::<Action>::new(), actions(dice![3]));
//         assert_eq!(Vec::<Action>::new(), actions(dice![3, 4]));
//         assert_eq!(Vec::<Action>::new(), actions(dice![3, 3, 6, 6]));
//
//         // with scoring dice
//         use super::Action::{Pass, Roll};
//         assert_eq!(vec![Pass, Roll(dice![1])], actions(dice![1, 3]));
//         assert_eq!(vec![Pass, Roll(dice![5])], actions(dice![2, 3, 5, 6]));
//         assert_eq!(
//             vec![Pass, Roll(dice![1]), Roll(dice![1, 5]), Roll(dice![5])],
//             actions(dice![1, 2, 3, 5, 6])
//         );
//         assert_eq!(
//             vec![Pass, Roll(dice![1]), Roll(dice![1, 1])],
//             actions(dice![1, 1, 3])
//         );
//         assert_eq!(
//             vec![Pass, Roll(dice![1]), Roll(dice![1, 1])],
//             actions(dice![1, 1, 3, 3])
//         );
//         assert_eq!(
//             vec![
//                 Pass,
//                 Roll(dice![1]),
//                 Roll(dice![1, 3, 3, 3]),
//                 Roll(dice![1, 3, 3, 3, 5]),
//                 Roll(dice![1, 5]),
//                 Roll(dice![3, 3, 3]),
//                 Roll(dice![3, 3, 3, 5]),
//                 Roll(dice![5]),
//             ],
//             actions(dice![1, 3, 3, 3, 5])
//         );
//
//         // should include hold (straight ++ 5) action
//         assert_eq!(
//             vec![
//                 Pass,
//                 Roll(dice![2, 3, 4, 5, 5, 6]),
//                 Roll(dice![2, 3, 4, 5, 6]),
//                 Roll(dice![5]),
//                 Roll(dice![5, 5]),
//             ],
//             actions(dice![2, 3, 4, 5, 5, 6]),
//         );
//
//         assert_eq!(
//             vec![
//                 Pass,
//                 Roll(dice![1]),
//                 Roll(dice![1, 1]),
//                 Roll(dice![1, 1, 2, 3, 4, 5]),
//                 Roll(dice![1, 1, 5]),
//                 Roll(dice![1, 2, 3, 4, 5]),
//                 Roll(dice![1, 5]),
//                 Roll(dice![5]),
//             ],
//             actions(dice![1, 1, 2, 3, 4, 5]),
//         );
//     }
// }
