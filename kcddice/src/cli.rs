use crate::{
    dice::DieKindTable,
    parse,
    search::{
        p_rv1_lte_rv2, Action, ActionValue, Context, MarkovMatrix, NormalizedStateAction, State,
    },
    DEFAULT_MAX_SCORE, DEFAULT_TOTAL_SCORE,
};
use pico_args;
use serde::{Deserialize, Serialize};
use std::{fmt, str::FromStr};
use tabular::{row, Row, Table};
// For some reason, `std::time::Instant` doesn't just use `performance.now()` in
// wasm/web... this crate does that.
use trice::Instant;

///////////////////////////
// String parser helpers //
///////////////////////////

fn parse_req<T>(s: &str) -> Result<T, String>
where
    T: FromStr,
    T::Err: fmt::Display,
{
    T::from_str(s).map_err(|err| err.to_string())
}

fn parse_opt<T>(opt_s: Option<&str>) -> Result<Option<T>, String>
where
    T: FromStr,
    T::Err: fmt::Display,
{
    opt_s
        .map(T::from_str)
        .transpose()
        .map_err(|err| err.to_string())
}

//////////////////////
// CLI Args Wrapper //
//////////////////////

pub struct Args(pico_args::Arguments);

impl Args {
    pub fn new(inner: pico_args::Arguments) -> Self {
        Self(inner)
    }

    fn subcommand(&mut self) -> Result<Option<String>, String> {
        self.0.subcommand().map_err(|err| err.to_string())
    }

    fn opt_value(&mut self, keys: impl Into<pico_args::Keys>) -> Result<Option<String>, String> {
        self.0
            .opt_value_from_fn(keys, |s| Result::<_, pico_args::Error>::Ok(s.to_owned()))
            .map_err(|err| err.to_string())
    }

    fn free_value(&mut self) -> Result<String, String> {
        self.0
            .free_from_fn(|s| Result::<_, pico_args::Error>::Ok(s.to_owned()))
            .map_err(|err| err.to_string())
    }

    fn expect_finished(self) -> Result<(), String> {
        let remaining = self.0.finish();
        if !remaining.is_empty() {
            Err(format!("unexpected arguments left: '{:?}'", remaining))
        } else {
            Ok(())
        }
    }

    fn maybe_help(&mut self, usage: &str) {
        if self.0.contains(["-h", "--help"]) {
            print!("{}", usage);
            std::process::exit(0);
        }
    }
}

/////////////
// Metrics //
/////////////

#[derive(Clone, Default, PartialEq, Eq)]
pub struct Metrics(pub Vec<(String, String)>);

impl Metrics {
    pub fn new() -> Self {
        Self(Vec::new())
    }

    pub fn push(&mut self, label: impl Into<String>, value: impl Into<String>) -> &mut Self {
        self.0.push((label.into(), value.into()));
        self
    }

    pub fn to_table(&self) -> Table {
        let mut table = Table::new("{:>}  {:<}");

        for (label, value) in &self.0 {
            table.add_row(row!(label, value));
        }

        table
    }
}

///////////////////
// Command trait //
///////////////////

pub trait Command: Sized {
    const USAGE: &'static str;

    type Output: fmt::Display;

    fn try_from_cli_args(args: Args) -> Result<Self, String>;
    fn run(self) -> Result<Self::Output, String>;
}

///////////////////////
// BestActionCommand //
///////////////////////

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BestActionCommand {
    starting_dice: parse::DiceSet,
    total_score: u16,
    max_score: u16,
    round_score: u16,
    rolled_dice: parse::DiceVec,
}

impl BestActionCommand {
    pub fn try_from_str_args(
        starting_dice: Option<&str>,
        total_score: Option<&str>,
        max_score: Option<&str>,
        round_score: &str,
        rolled_dice: &str,
    ) -> Result<Self, String> {
        let cmd = Self {
            starting_dice: parse_opt(starting_dice)?
                .unwrap_or_else(|| parse::DiceSet::all_standard(6)),
            total_score: parse_opt(total_score)?.unwrap_or(DEFAULT_TOTAL_SCORE),
            max_score: parse_opt(max_score)?.unwrap_or(DEFAULT_MAX_SCORE),
            round_score: parse_req(round_score)?,
            rolled_dice: parse_req(rolled_dice)?,
        };

        if cmd.total_score >= cmd.max_score {
            return Err(format!(
                "the total score ({}) must be less than the max score ({})",
                cmd.total_score, cmd.max_score
            ));
        }

        cmd.starting_dice
            .validate_init_set(&cmd.rolled_dice.to_die_set())?;

        Ok(cmd)
    }
}

impl Command for BestActionCommand {
    const USAGE: &'static str = "\
kcddice best-action - compute the best action to take in a round

USAGE:
    kcddice best-action [option ...] <round-score> <rolled-dice>

EXAMPLES:
    kcddice best-action 550 [1,1,2,3,5,6]

OPTIONS:
    · --starting-dice / -s [s:n,hk:m,o:l,..] (default: [s:6])
      The set of dice we started the game with.
      For example, 3 Standard dice, 2 Heavenly Kingdom dice, and 1 Odd die would
      be formatted as `-s [s:3,hk:2,o:1]`

    · --total-score / -t score (default: 0)
      Our current total score, accumulated from previous rounds.

    · --max-score / -m score (default: 4000)
      The score needed to win the game.
";

    type Output = BestActionCommandOutput;

    fn try_from_cli_args(mut args: Args) -> Result<Self, String> {
        args.maybe_help(Self::USAGE);

        let starting_dice = args.opt_value(["-s", "--starting-dice"])?;
        let total_score = args.opt_value(["-t", "--total-score"])?;
        let max_score = args.opt_value(["-m", "--max-score"])?;
        let round_score = args.free_value()?;
        let rolled_dice = args.free_value()?;
        args.expect_finished()?;

        Self::try_from_str_args(
            starting_dice.as_deref(),
            total_score.as_deref(),
            max_score.as_deref(),
            &round_score,
            &rolled_dice,
        )
    }

    fn run(self) -> Result<Self::Output, String> {
        let (dice_table, starting_dice) = self.starting_dice.to_compact_form();
        let mut ctxt = Context::new(dice_table, starting_dice);

        let target_score = self.max_score - self.total_score;

        let rolled_dice = self.rolled_dice.to_compact_form(&dice_table);
        let state = State::new(self.round_score, target_score, rolled_dice);

        let start_time = Instant::now();
        let action_values = state.actions_by_expected_value(&mut ctxt);
        let action_values = ActionValuesTable(action_values);
        let search_duration = start_time.elapsed();

        let mut metrics = Metrics::new();

        metrics.push("search duration", format!("{:.2?}", search_duration));
        metrics.push("actions explored", ctxt.actions_explored().to_string());
        metrics.push(
            "game finished rate",
            format!("{:0.3}", ctxt.game_finished_rate()),
        );
        metrics.push(
            "action value cache size",
            format!(
                "{} ({})",
                ctxt.action_value_cache().cache_size(),
                ctxt.action_value_cache().cache_size_bytes(),
            ),
        );
        metrics.push(
            "action value cache hit rate",
            format!(
                "{:0.3} (h: {}, m: {})",
                ctxt.action_value_cache().cache_hit_rate(),
                ctxt.action_value_cache().cache_hits(),
                ctxt.action_value_cache().cache_misses(),
            ),
        );
        metrics.push(
            "actions cache size",
            format!(
                "{} ({})",
                ctxt.actions_cache().cache_size(),
                ctxt.actions_cache().cache_size_bytes(),
            ),
        );
        metrics.push(
            "actions cache hit rate",
            format!(
                "{:0.3} (h: {}, m: {})",
                ctxt.actions_cache().cache_hit_rate(),
                ctxt.actions_cache().cache_hits(),
                ctxt.actions_cache().cache_misses(),
            ),
        );

        Ok(BestActionCommandOutput {
            dice_table,
            action_values,
            metrics,
        })
    }
}

fn row_from_cells(cells: impl Iterator<Item = String>) -> Row {
    let mut row = Row::new();
    for cell in cells {
        row.add_cell(cell);
    }
    row
}

#[derive(Clone, Default)]
pub struct ActionValuesTable(pub Vec<ActionValue>);

impl ActionValuesTable {
    fn to_table(&self, dice_table: &DieKindTable) -> Table {
        let mut table = Table::new("{:>}  {:<}  {:>} {:>}").with_row(row!(
            format!("{:>12}", "action"),
            "held dice",
            "exp v",
            "pbust"
        ));

        let len = self.0.len();
        for action_value in self.0.iter().take(10) {
            let row = row_from_cells(action_value.to_row_cells(dice_table).into_iter());
            table.add_row(row);
        }

        // we only show the top 10 results, but display '...' to show that
        // there were more.
        if len > 10 {
            table.add_row(row!("...", format!("(+ {})", len - 10), "", ""));
        }

        table
    }
}

#[derive(Clone, Default)]
pub struct BestActionCommandOutput {
    pub dice_table: DieKindTable,
    pub action_values: ActionValuesTable,
    pub metrics: Metrics,
}

impl fmt::Display for BestActionCommandOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "\n{}\n{}",
            self.action_values.to_table(&self.dice_table),
            self.metrics.to_table()
        )
    }
}

///////////////////////
// ScoreDistrCommand //
///////////////////////

#[derive(Debug)]
pub struct ScoreDistrCommand {
    round_total: u16,
    target_score: u16,
    dice_left: parse::DiceSet,
    all_dice: parse::DiceSet,
}

impl ScoreDistrCommand {
    fn try_from_str_args(
        round_total: &str,
        target_score: &str,
        dice_left: &str,
        all_dice: Option<&str>,
    ) -> Result<Self, String> {
        let cmd = Self {
            round_total: parse_req(round_total)?,
            target_score: parse_req(target_score)?,
            dice_left: parse_req(dice_left)?,
            all_dice: parse_opt(all_dice)?.unwrap_or_else(|| parse::DiceSet::all_standard(6)),
        };

        cmd.all_dice.validate_init_set(&cmd.dice_left)?;

        Ok(cmd)
    }
}

impl Command for ScoreDistrCommand {
    const USAGE: &'static str = "\
kcddice score-distr - TODO

USAGE:
    kcddice score-distr [option ...] <round-total> <target-score> <dice-left>
";

    type Output = ScoreDistrCommandOutput;

    fn try_from_cli_args(mut args: Args) -> Result<Self, String> {
        args.maybe_help(Self::USAGE);

        let all_dice = args.opt_value(["-k", "--die-kinds"])?;
        let round_total = args.free_value()?;
        let target_score = args.free_value()?;
        let dice_left = args.free_value()?;
        args.expect_finished()?;

        Self::try_from_str_args(&round_total, &target_score, &dice_left, all_dice.as_deref())
    }

    fn run(self) -> Result<Self::Output, String> {
        let (dice_table, all_dice_counts) = self.all_dice.to_compact_form();
        let dice_left = self.dice_left.to_counts(&dice_table);
        let mut ctxt = Context::new(dice_table, all_dice_counts);

        let qstate = NormalizedStateAction::new(self.round_total, self.target_score, dice_left);

        let start_time = Instant::now();
        let score_pmf = qstate.score_distribution(&mut ctxt);
        let search_duration = start_time.elapsed();

        let mut dense_score_pmf = score_pmf.clone().into_vec();
        dense_score_pmf.sort_unstable_by(|(s1, _), (s2, _)| s1.cmp(s2));

        let mut metrics = Metrics::new();
        metrics.push("search duration", format!("{:.2?}", search_duration));
        metrics.push(
            "pmf expected value",
            format!("{:.3}", score_pmf.expected_value()),
        );
        metrics.push("pmf total mass", format!("{}", score_pmf.total_mass()));
        metrics.push("actions explored", ctxt.actions_explored().to_string());
        metrics.push(
            "game finished rate",
            format!("{:0.3}", ctxt.game_finished_rate()),
        );
        metrics.push(
            "score distr cache size",
            format!(
                "{} ({})",
                ctxt.score_distr_cache().cache_size(),
                ctxt.score_distr_cache().cache_size_bytes(),
            ),
        );
        metrics.push(
            "score distr cache hit rate",
            format!(
                "{:0.3} (h: {}, m: {})",
                ctxt.score_distr_cache().cache_hit_rate(),
                ctxt.score_distr_cache().cache_hits(),
                ctxt.score_distr_cache().cache_misses(),
            ),
        );
        metrics.push(
            "actions cache size",
            format!(
                "{} ({})",
                ctxt.actions_cache().cache_size(),
                ctxt.actions_cache().cache_size_bytes(),
            ),
        );
        metrics.push(
            "actions cache hit rate",
            format!(
                "{:0.3} (h: {}, m: {})",
                ctxt.actions_cache().cache_hit_rate(),
                ctxt.actions_cache().cache_hits(),
                ctxt.actions_cache().cache_misses(),
            ),
        );

        Ok(ScoreDistrCommandOutput {
            distr: dense_score_pmf,
            metrics,
        })
    }
}

pub struct ScoreDistrCommandOutput {
    distr: Vec<(u16, f64)>,
    metrics: Metrics,
}

impl fmt::Display for ScoreDistrCommandOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut table = Table::new("{:>}  {:<}");

        for (score, p_score) in &self.distr {
            table.add_row(row!(score, p_score));
        }

        write!(f, "\n{}\n{}", table, self.metrics.to_table())
    }
}

/////////////////////////
// MarkovMatrixCommand //
/////////////////////////

#[derive(Debug)]
pub struct MarkovMatrixCommand {
    target_score: u16,
    all_dice: parse::DiceSet,
}

impl MarkovMatrixCommand {
    fn try_from_str_args(target_score: &str, all_dice: Option<&str>) -> Result<Self, String> {
        let cmd = Self {
            target_score: parse_req(target_score)?,
            all_dice: parse_opt(all_dice)?.unwrap_or_else(|| parse::DiceSet::all_standard(6)),
        };

        cmd.all_dice.validate_init_set(&cmd.all_dice)?;

        Ok(cmd)
    }
}

impl Command for MarkovMatrixCommand {
    // TODO(philiphayes): fill out
    const USAGE: &'static str = "";

    type Output = MarkovMatrixCommandOutput;

    fn try_from_cli_args(mut args: Args) -> Result<Self, String> {
        args.maybe_help(Self::USAGE);

        let all_dice = args.opt_value(["-k", "--die-kinds"])?;
        let target_score = args.free_value()?;
        args.expect_finished()?;

        Self::try_from_str_args(&target_score, all_dice.as_deref())
    }

    fn run(self) -> Result<Self::Output, String> {
        let start_time = Instant::now();
        let (die_table, die_counts) = self.all_dice.to_compact_form();
        let matrix = MarkovMatrix::from_optimal_policy(die_table, die_counts, self.target_score);
        let search_duration = start_time.elapsed();

        let mut metrics = Metrics::new();
        metrics.push("search duration", format!("{:.2?}", search_duration));

        Ok(MarkovMatrixCommandOutput { matrix, metrics })
    }
}

pub struct MarkovMatrixCommandOutput {
    matrix: MarkovMatrix,
    metrics: Metrics,
}

impl fmt::Display for MarkovMatrixCommandOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\n{:?}\n{}", self.matrix, self.metrics.to_table())
    }
}

/////////////////////
// TurnsCdfCommand //
/////////////////////

#[derive(Debug)]
pub struct TurnsCdfCommand {
    target_score: u16,
    max_num_turns: usize,
    our_dice: parse::DiceSet,
    their_dice: parse::DiceSet,
}

impl TurnsCdfCommand {
    fn try_from_str_args(
        target_score: &str,
        max_num_turns: &str,
        our_dice: Option<&str>,
        their_dice: Option<&str>,
    ) -> Result<Self, String> {
        let cmd = Self {
            target_score: parse_req(target_score)?,
            max_num_turns: parse_req(max_num_turns)?,
            our_dice: parse_opt(our_dice)?.unwrap_or_else(|| parse::DiceSet::all_standard(6)),
            their_dice: parse_opt(their_dice)?.unwrap_or_else(|| parse::DiceSet::all_standard(6)),
        };

        cmd.our_dice.validate_init_set(&cmd.our_dice)?;
        cmd.their_dice.validate_init_set(&cmd.their_dice)?;

        Ok(cmd)
    }
}

impl Command for TurnsCdfCommand {
    // TODO(philiphayes): fill out
    const USAGE: &'static str = "";

    type Output = TurnsCdfCommandOutput;

    fn try_from_cli_args(mut args: Args) -> Result<Self, String> {
        args.maybe_help(Self::USAGE);

        let our_dice = args.opt_value(["-o", "--our-die-kinds"])?;
        let their_dice = args.opt_value(["-t", "--their-die-kinds"])?;
        let target_score = args.free_value()?;
        let max_num_turns = args.free_value()?;
        args.expect_finished()?;

        Self::try_from_str_args(
            &target_score,
            &max_num_turns,
            our_dice.as_deref(),
            their_dice.as_deref(),
        )
    }

    fn run(self) -> Result<Self::Output, String> {
        let start_time = Instant::now();

        let (our_die_table, our_die_counts) = self.our_dice.to_compact_form();
        let our_matrix =
            MarkovMatrix::from_optimal_policy(our_die_table, our_die_counts, self.target_score);
        let our_turns_cdf = our_matrix.turns_to_win_cdf(self.max_num_turns);

        let their_turns_cdf = if self.our_dice == self.their_dice {
            our_turns_cdf.clone()
        } else {
            let (their_die_table, their_die_counts) = self.their_dice.to_compact_form();
            let their_matrix = MarkovMatrix::from_optimal_policy(
                their_die_table,
                their_die_counts,
                self.target_score,
            );
            their_matrix.turns_to_win_cdf(self.max_num_turns)
        };

        let search_duration = start_time.elapsed();

        // Assuming we always go first and play optimally, what is our a priori
        // win probability against an opponent that also plays optimally, but
        // may have different dice?
        //
        // Assuming the our # turns to win is modeled as a r.v. X_1 ~ our_turns_cdf
        // and the opponent's is modeled by independent r.v. X_2 ~ their_turns_cdf,
        // the win probability is (mostly) the same as `Pr[X_1 <= X_2]`, i.e., the
        // probability we (independently) finish the game in less turns than the
        // opponent.
        //
        // This is useful for deciding how much to bet (e.g., following the
        // Kelley-Criterion).
        //
        // For the typical setup (target=4000, our dice=6 normal, their dice=6
        // normal), this is ≈0.562. The Kelley-Criterion optimal bet is then 12.5%
        // of your total wealth.
        let p_win = p_rv1_lte_rv2(our_turns_cdf.view(), their_turns_cdf.view());

        let mut metrics = Metrics::new();
        metrics.push("search_duration", format!("{:.2?}", search_duration));
        metrics.push("Pr[win]", format!("{}", p_win));

        Ok(TurnsCdfCommandOutput {
            max_num_turns: self.max_num_turns,
            our_turns_cdf: our_turns_cdf.to_vec(),
            their_turns_cdf: their_turns_cdf.to_vec(),
            metrics,
        })
    }
}

pub struct TurnsCdfCommandOutput {
    max_num_turns: usize,
    our_turns_cdf: Vec<f64>,
    their_turns_cdf: Vec<f64>,
    metrics: Metrics,
}

impl fmt::Display for TurnsCdfCommandOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut table = Table::new("{:>}  {:<}  {:<}");

        table.add_row(row!(
            format!("{:>8}", "turn"),
            "our turns CDF",
            "their turns CDF"
        ));

        for turn in 1..=self.max_num_turns {
            table.add_row(row!(
                turn,
                self.our_turns_cdf[turn - 1],
                self.their_turns_cdf[turn - 1]
            ));
        }

        write!(f, "\n{}\n{}", table, self.metrics.to_table())
    }
}

/////////////////
// BaseCommand //
/////////////////

#[derive(Debug)]
pub enum BaseCommand {
    BestAction(BestActionCommand),
    ScoreDistr(ScoreDistrCommand),
    MarkovMatrix(MarkovMatrixCommand),
    TurnsCdf(TurnsCdfCommand),
}

impl Command for BaseCommand {
    const USAGE: &'static str = "\
kcddice - A utility for optimally playing the Kingdom Come: Deliverance dice game!

USAGE:
    kcddice [option ...] <subcommand>

SUBCOMMANDS:
    · kcddice best-action - compute the best action to take in a round
    · kcddice score-distr - compute full score PMF given the score and number of dice left to roll
    · kcddice markov-matrix - TODO
    · kcddice turns-cdf - TODO
";

    type Output = String;

    fn try_from_cli_args(mut args: Args) -> Result<Self, String> {
        let maybe_subcommand = args.subcommand()?;

        match maybe_subcommand.as_deref() {
            Some("best-action") => Ok(Self::BestAction(BestActionCommand::try_from_cli_args(
                args,
            )?)),
            Some("score-distr") => Ok(Self::ScoreDistr(ScoreDistrCommand::try_from_cli_args(
                args,
            )?)),
            Some("markov-matrix") => Ok(Self::MarkovMatrix(
                MarkovMatrixCommand::try_from_cli_args(args)?,
            )),
            Some("turns-cdf") => Ok(Self::TurnsCdf(TurnsCdfCommand::try_from_cli_args(args)?)),
            Some(command) => Err(format!("'{}' is not a recognized command", command)),
            None => {
                args.maybe_help(Self::USAGE);
                Err("no subcommand specified".to_string())
            }
        }
    }

    fn run(self) -> Result<String, String> {
        match self {
            Self::BestAction(cmd) => cmd.run().map(|out| out.to_string()),
            Self::ScoreDistr(cmd) => cmd.run().map(|out| out.to_string()),
            Self::MarkovMatrix(cmd) => cmd.run().map(|out| out.to_string()),
            Self::TurnsCdf(cmd) => cmd.run().map(|out| out.to_string()),
        }
    }
}
