use crate::{
    dice::{DiceVec, DieKindCounts, DieKindTable},
    search::{Action, Cache, Context, NormalizedStateAction, State},
};
use rand::{seq::SliceRandom, Rng};

pub struct GameState {
    max_score: u16,

    opponent_starting_dice_table: DieKindTable,
    opponent_starting_dice: DieKindCounts,

    our_starting_dice_table: DieKindTable,
    our_starting_dice: DieKindCounts,
}

#[derive(Copy, Clone)]
pub struct TotalState {
    opponent_total_score: u16,
    our_total_score: u16,
}

pub struct RoundState<'a> {
    game_state: &'a GameState,
    total_state: TotalState,
    our_qstate: NormalizedStateAction,
}

impl<'a> RoundState<'a> {
    fn init_round_state(game_state: &'a GameState, total_state: TotalState) -> Self {
        let our_qstate = NormalizedStateAction::init_state(
            game_state.max_score - total_state.our_total_score,
            game_state.our_starting_dice,
        );
        Self {
            game_state,
            total_state,
            our_qstate,
        }
    }

    #[inline]
    fn our_dice_table(&self) -> DieKindTable {
        self.game_state.our_starting_dice_table
    }

    #[inline]
    fn our_starting_dice(&self) -> DieKindCounts {
        self.game_state.our_starting_dice
    }

    #[inline]
    fn our_state(&self, rolled_dice: DiceVec) -> State {
        self.our_qstate.into_state(rolled_dice)
    }

    fn roll_dice<R: Rng>(&self, rng: &mut R) -> DiceVec {
        self.our_qstate
            .dice_left
            .roll_dice(rng, self.our_dice_table())
    }

    /// Applys an Agent's action to the current state. Returns `Some(round_total)`
    /// if the agent passed.
    fn apply_action(&mut self, state: State, action: Action) -> Option<u16> {
        match action {
            Action::Pass => Some(state.my_round_total + state.rolled_dice.score()),
            Action::Roll(held_dice) => {
                self.our_qstate = NormalizedStateAction::from_state_roll_action(
                    self.our_starting_dice(),
                    state,
                    held_dice,
                );
                None
            }
        }
    }
}

// agent

// top-level context/parameters

pub trait Agent {
    fn action(&mut self, round_state: &RoundState, state: State) -> Action;
}

pub struct AlwaysPassAgent;

impl Agent for AlwaysPassAgent {
    fn action(&mut self, _round_state: &RoundState, _state: State) -> Action {
        Action::Pass
    }
}

pub struct RandomActionAgent<R> {
    rng: R,
}

impl<R: Rng> Agent for RandomActionAgent<R> {
    fn action(&mut self, _round_state: &RoundState, state: State) -> Action {
        let actions = state.actions();
        *actions
            .choose(&mut self.rng)
            .expect("no actions available!")
    }
}

pub struct ExpectedValueMaxAgent {
    context: Context,
}

impl Agent for ExpectedValueMaxAgent {
    fn action(&mut self, _round_state: &RoundState, state: State) -> Action {
        state
            .actions_by_expected_value(&mut self.context)
            .first()
            .expect("no actions available!")
            .action
    }
}

//

enum RoundOutcome {}

struct RoundRunner<R> {
    rng: R,
}

impl<R: Rng> RoundRunner<R> {
    // return the current agent's round total after sampling a round.
    fn run_round<A: Agent>(
        &mut self,
        agent: &mut A,
        game_state: &GameState,
        total_state: TotalState,
    ) -> u16 {
        let mut round_state = RoundState::init_round_state(game_state, total_state);
        let mut niters = 0;

        loop {
            debug_assert!(niters <= 50);

            let rolled_dice = round_state.roll_dice(&mut self.rng);
            if rolled_dice.is_bust() {
                // agent went bust; agent earned no points this round.
                return 0;
            }

            // ask the agent what they want to do
            let new_state = round_state.our_qstate.into_state(rolled_dice);
            let agent_action = agent.action(&round_state, new_state);
            let maybe_passed = round_state.apply_action(new_state, agent_action);

            if let Some(round_total) = maybe_passed {
                // agent passed; add their round total to their total score
                // and return for the next agent to play.
                return round_total;
            }

            niters += 1;
        }
    }
}

/*

target_score: 4000
total_score: 0
dice_set: [s:6]

n = 100

SEM = 50.28169291079151
E[X_1] = 564.0533673921834
E[X_2] = 517
stddev(X_1) = 502.81692910791514
stddev(X_2) = 552.2327407896059
max_x |p1(x)-p2(x)| = 0.04784528958813103
E[|pmf1 - pmf2|] = 0.004152304527861793
√ E[(pmf1 - pmf2)^2] = 0.008406875015202535


n = 1_000

SEM = 15.900467420724281
E[X_1] = 564.0533673921834
E[X_2] = 547.0500000000001
stddev(X_1) = 502.81692910791526
stddev(X_2) = 480.82096200145014
max_x |p1(x)-p2(x)| = 0.009845289588131051
E[|pmf1 - pmf2|] = 0.0013660277220587007
√ E[(pmf1 - pmf2)^2] = 0.0025251326396472772


n = 10_000

E[X_1] = 564.0533673921834
E[X_2] = 558.1750000000001
max_x |p1(x)-p2(x)| = 0.004445289588131035
E[|pmf1 - pmf2|] = 0.0005431768959190096
√ E[(pmf1 - pmf2)^2] = 0.0009997623833912016


n = 20_000

SEM = 3.555452602676023
E[X_1] = 564.0533673921834
E[X_2] = 563.5749999999997
stddev(X_1) = 502.81692910791514
stddev(X_2) = 508.03195704109004
max_x |p1(x)-p2(x)| = 0.0029969947525471316
E[|pmf1 - pmf2|] = 0.0003983474818861177
√ E[(pmf1 - pmf2)^2] = 0.0007518747572107189


n = 100 (always pass agent)

E[X_1] = 564.0533673921838
E[X_2] = 338
max_x |p1(x)-p2(x)| = 0.17215471041186897
E[|pmf1 - pmf2|] = 0.015303956389566423
√ E[(pmf1 - pmf2)^2] = 0.038429660820065575


n = 20_000 (always pass agent)

E[X_1] = 564.0533673921835
E[X_2] = 397.65749999999997
max_x |p1(x)-p2(x)| = 0.15930471041186897
E[|pmf1 - pmf2|] = 0.01222738953480571
√ E[(pmf1 - pmf2)^2] = 0.03357450489391348


n = 100 (random action agent)

E[X_1] = 564.0533673921834
E[X_2] = 220
max_x |p1(x)-p2(x)| = 0.14784528958813106
E[|pmf1 - pmf2|] = 0.01147435623961885
√ E[(pmf1 - pmf2)^2] = 0.027718128373956496


n = 20_000 (random action agent)

E[X_1] = 564.0533673921836
E[X_2] = 245.73
max_x |p1(x)-p2(x)| = 0.18364528958813106
E[|pmf1 - pmf2|] = 0.01130877843815173
√ E[(pmf1 - pmf2)^2] = 0.028573495007785738

*/

#[cfg(test)]
mod test {
    use super::*;
    use crate::{
        parse::DiceSet,
        search::{MarkovMatrix, ScorePMF},
    };
    use claim::{assert_gt, assert_le};
    use rand::SeedableRng;
    use rand_xoshiro::Xoroshiro64Star;
    use std::{cmp, collections::HashMap, ops::AddAssign};

    // `stderr_mean` is the standard error of the mean (i.e., standard deviation
    // of the sample mean, possibly an estimate).
    //
    // `std_confidence` is in terms of standard sigmas from the standard mean 0.
    // it is a function of our tolerated failure probability, where we accidentally
    // sample a mean that is outside our confidence interval, despite the samples
    // actually coming from our population. as our tolerated failure probability
    // decreases, our required confidence bound increases.
    //
    // this function returns the maximum tolerated absolute deviation of the
    // observed sample mean from the true population mean, E = |μ - μ_x|.
    fn max_tolerated_abs_deviation(stderr_mean: f64, std_confidence: f64) -> f64 {
        stderr_mean * std_confidence
    }

    fn sample_emp_score_distr<A: Agent>(
        num_trials: usize,
        round_runner: &mut RoundRunner<impl Rng>,
        agent: &mut A,
        game_state: &GameState,
        total_state: TotalState,
    ) -> ScorePMF {
        let mut emp_score_counts = HashMap::<u16, usize>::new();
        for _ in 0..num_trials {
            let round_total = round_runner.run_round(agent, game_state, total_state);
            emp_score_counts
                .entry(round_total)
                .or_insert(0)
                .add_assign(1);
        }
        ScorePMF::from_counts_iter(num_trials, emp_score_counts)
    }

    #[test]
    fn test_run_round_mc_vs_score_distr() {
        let dice_set = DiceSet::all_standard(6);
        let (dice_table, starting_dice) = dice_set.to_compact_form();

        let target_score = 4000;

        let game_state = GameState {
            max_score: target_score,
            our_starting_dice: starting_dice,
            our_starting_dice_table: dice_table,
            // not relevant
            opponent_starting_dice: DieKindCounts::new(),
            opponent_starting_dice_table: DieKindTable::all_standard(),
        };
        let total_state = TotalState {
            our_total_score: 0,
            // not relevant
            opponent_total_score: 0,
        };

        // the true population score distribution
        let mut context = Context::new(dice_table, starting_dice);
        let qstate = NormalizedStateAction::new(0, target_score, starting_dice);
        let score_distr = qstate.score_distribution(&mut context);

        let mut round_runner = RoundRunner {
            rng: Xoroshiro64Star::seed_from_u64(0xd15c0),
        };

        // empirical score distribution for various agents
        let num_trials = 1_000;
        let expmax_score_distr = sample_emp_score_distr(
            num_trials,
            &mut round_runner,
            &mut ExpectedValueMaxAgent {
                context: Context::new(dice_table, starting_dice),
            },
            &game_state,
            total_state,
        );

        // we want to know whether the empirical sample score distr was drawn
        // from the expected population score distribution.
        //
        // for now, we will just compare the sample mean with the population mean
        // to check that they are not farther apart than our maximum deviation,
        // given our confidence bound.

        // the standard deviation of the sample mean statistic. since we have the
        // population distribution availble, we don't need an estimate.
        let stderr_mean = score_distr.stddev() / (num_trials as f64).sqrt();
        // corresponds with a confidence level of ≈99.99994%
        let std_confidence = 5.0;
        // if |μ - μ_x| <= max_abs_dev, then we can say the two means are (probably)
        // drawn from the same distribution. this decreases as we increase the
        // number of samples taken.
        let max_abs_dev = max_tolerated_abs_deviation(stderr_mean, std_confidence);

        let population_mean = score_distr.expected_value();
        let expmax_sample_mean = expmax_score_distr.expected_value();

        // the ExpectedValueMaxAgent sample mean should match population mean
        assert_le!((expmax_sample_mean - population_mean).abs(), max_abs_dev);
    }
}
