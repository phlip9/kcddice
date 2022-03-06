use futures_util::future::{self, Either};
use kcddice::{cli::BestActionCommand, parse};
use log::{debug, info, trace, warn};
use std::{str::FromStr, time::Duration};
use sycamore::prelude::*;
use web_sys::Event;

pub mod cmd_worker;

#[derive(Debug, Prop)]
pub struct AppProps<'a> {
    best_action_props: BestActionPageProps<'a>,
}

impl<'a> AppProps<'a> {
    pub fn init_placeholders(ctx: ScopeRef<'a>) -> Self {
        let starting_dice_str = "s:3 hk:2 o:1";

        let starting_dice_str = ctx.create_signal(starting_dice_str.to_string());
        let total_score_str = ctx.create_signal("1500".to_string());
        let max_score_str = ctx.create_signal("4000".to_string());
        let round_score_str = ctx.create_signal("550".to_string());
        let rolled_dice_str = ctx.create_signal("1 1hk 3 5hk 6 6o".to_string());

        let dice_vec = parse::DiceVec::from_str("1").unwrap();
        let best_action = parse::Action::Hold(dice_vec);
        let output_result = Ok(best_action);

        // let output_result = Err("rolled dice can only contain dice from the starting dice: start: [:6], rolled: [:3, hk:2, o:1]".to_owned());

        let best_action_state = BestActionOutputState {
            is_searching: create_rc_signal(false),
            output_result: create_rc_signal(output_result),
        };

        AppProps {
            best_action_props: BestActionPageProps {
                starting_dice_str,
                total_score_str,
                max_score_str,
                round_score_str,
                rolled_dice_str,
                best_action_state,
            },
        }
    }
}

#[component]
pub fn App<'a, G: Html>(ctx: ScopeRef<'a>, props: AppProps<'a>) -> View<G> {
    debug!("init App component: props: {:?}", props);

    view! { ctx,
        // central container
        div(class="page-wrapper") {
            Title {}

            main {
                section(id="inputs-page") {
                    BestActionPage(props.best_action_props)
                }
            }

            Info {}
        }
    }
}

#[component]
fn FleurRight<G: Html>(ctx: ScopeRef) -> View<G> {
    trace!("FleurRight: init");

    view! { ctx,
        i(dangerously_set_inner_html="<svg><use xlink:href=\"imgs/fleur-right.svg#fleur-right\"/></svg>")
    }
}

#[component]
fn FleurLeft<G: Html>(ctx: ScopeRef) -> View<G> {
    trace!("FleurLeft: init");

    view! { ctx,
        i(dangerously_set_inner_html="<svg><use xlink:href=\"imgs/fleur-left.svg#fleur-left\"/></svg>")
    }
}

#[component]
fn Title<G: Html>(ctx: ScopeRef) -> View<G> {
    trace!("Title: init");

    view! { ctx,
        header {
            h1(dangerously_set_inner_html="<svg><use xlink:href=\"imgs/logo-kcd.svg#logo-kcd\"/></svg>")
            h2 {
                FleurLeft {}
                "optimal dice strategy"
                FleurRight {}
            }
        }
    }
}

fn empty_str_to_opt(s: &str) -> Option<&str> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

#[derive(Clone, Debug, Prop)]
pub struct BestActionPageProps<'a> {
    starting_dice_str: &'a Signal<String>,
    total_score_str: &'a Signal<String>,
    max_score_str: &'a Signal<String>,
    round_score_str: &'a Signal<String>,
    rolled_dice_str: &'a Signal<String>,
    best_action_state: BestActionOutputState,
}

#[component]
fn BestActionPage<'a, G: Html>(ctx: ScopeRef<'a>, props: BestActionPageProps<'a>) -> View<G> {
    trace!("BestActionPage: init");

    let starting_dice_input = ctx.create_node_ref();
    let total_score_input = ctx.create_node_ref();
    let max_score_input = ctx.create_node_ref();
    let round_score_input = ctx.create_node_ref();
    let rolled_dice_input = ctx.create_node_ref();

    let best_action_submit = ctx.create_node_ref();

    let props_clone = props.clone();
    let handle_submit = move |event: Event| {
        trace!("BestActionPage::handle_submit: event: {:?}", event);
        event.prevent_default();
        event.stop_immediate_propagation();

        trace!("BestActionPage::handle_submit: is_searching.set(true)");
        props_clone.best_action_state.is_searching.set(true);

        let starting_dice_str = props_clone.starting_dice_str.get();
        let total_score_str = props_clone.total_score_str.get();
        let max_score_str = props_clone.max_score_str.get();
        let round_score_str = props_clone.round_score_str.get();
        let rolled_dice_str = props_clone.rolled_dice_str.get();

        trace!("BestActionPage::handle_submit: {starting_dice_str}, {total_score_str}/{max_score_str}, {round_score_str}, {rolled_dice_str}");

        let maybe_cmd = if round_score_str.is_empty() {
            Err("round score is required".to_owned())
        } else if rolled_dice_str.is_empty() {
            Err("rolled dice are required".to_owned())
        } else {
            BestActionCommand::try_from_str_args(
                empty_str_to_opt(starting_dice_str.as_str()),
                empty_str_to_opt(total_score_str.as_str()),
                empty_str_to_opt(max_score_str.as_str()),
                round_score_str.as_str(),
                rolled_dice_str.as_str(),
            )
        };

        match &maybe_cmd {
            Ok(cmd) => {
                debug!("BestActionPage::handle_submit: cmd: {cmd:?}");
            }
            Err(err) => {
                warn!("BestActionPage::handle_submit: invalid cmd args: {err}");
            }
        };

        let best_action_state = props_clone.best_action_state.clone();
        sycamore::futures::spawn_local(async move {
            // ensure the result only fulfills at least 0.35sec after we start,
            // to give the slide-up animation time to complete.
            let f_sleep = gloo_timers::future::sleep(Duration::from_secs_f32(0.35));
            let f_request = match maybe_cmd {
                Ok(cmd) => {
                    info!("BestActionPage::handle_submit::async: sending request to worker:\n{cmd:#?}");
                    Either::Left(crate::cmd_worker::CmdWorker::request(cmd))
                }
                Err(err) => Either::Right(future::ready(Err(err))),
            };
            let (_, result) = future::join(f_sleep, f_request).await;

            match &result {
                Ok(best_action) => {
                    debug!("BestActionPage::handle_submit::async: successful response from worker: {best_action}");
                }
                Err(err) => {
                    warn!(
                        "BestActionPage::handle_submit::async: error response from worker: {err}"
                    );
                }
            };

            best_action_state.is_searching.set(false);
            best_action_state.output_result.set(result);
        });
    };

    let best_action_state = props.best_action_state.clone();

    view! { ctx,
        form(
            id="best-action-form",
            on:submit=handle_submit,
        ) {
            hr(class="page-lines")

            div(class="input-wrapper") {
                label(for="starting-dice") { "starting dice" }
                input(
                    ref=starting_dice_input,
                    id="starting-dice",
                    name="starting-dice",
                    placeholder="s:6",
                    bind:value=props.starting_dice_str,
                )
            }
            div(class="input-wrapper") {
                label(for="total-score") { "total score" }
                input(
                    ref=total_score_input,
                    id="total-score",
                    name="total-score",
                    placeholder="0",
                    inputmode="numeric",
                    bind:value=props.total_score_str,
                )
                span(id="total-max-sep") { "/" }
                input(
                    ref=max_score_input,
                    id="max-score",
                    name="max-score",
                    placeholder="4000",
                    inputmode="numeric",
                    bind:value=props.max_score_str,
                )
            }
            div(class="input-wrapper") {
                label(for="round-score") { "round score" }
                input(
                    ref=round_score_input,
                    id="round-score",
                    name="round-score",
                    inputmode="numeric",
                    bind:value=props.round_score_str,
                )
            }
            div(class="input-wrapper") {
                label(for="rolled-dice") { "rolled dice" }
                input(
                    ref=rolled_dice_input,
                    id="rolled-dice",
                    name="rolled-dice",
                    bind:value=props.rolled_dice_str,
                )
            }

            hr(class="page-lines")

            button(
                ref=best_action_submit,
                id="best-action-submit",
                name="best-action-submit",
                value="best action",
                type="submit",
            ) {
                "best action"
            }
        }

        BestActionOutput(best_action_state)
    }
}

#[derive(Clone, Debug, Prop)]
pub struct BestActionOutputState {
    is_searching: RcSignal<bool>,
    output_result: RcSignal<Result<parse::Action, String>>,
}

#[component]
fn BestActionOutput<G: Html>(ctx: ScopeRef, state: BestActionOutputState) -> View<G> {
    trace!("BestActionOutput: state: {state:?}");

    let state_clone = state.clone();
    let output_class = ctx.create_memo(move || {
        if *state_clone.is_searching.get() {
            "searching"
        } else {
            ""
        }
    });

    let state_clone = state.clone();
    let main_str = ctx.create_memo(move || match state_clone.output_result.get().as_ref() {
        Ok(action) => action.to_action_str(),
        Err(_err) => "Error",
    });

    let state_clone = state.clone();
    let maybe_dice_str = ctx.create_memo(move || match state_clone.output_result.get().as_ref() {
        Ok(action) => action.to_maybe_dice_str(),
        Err(_err) => None,
    });

    let maybe_err_str = ctx.create_memo(move || match state.output_result.get().as_ref() {
        Ok(_action) => None,
        Err(err) => Some(err.clone()),
    });

    view! { ctx,
        section(id="output", class=output_class.get()) {
            div(id="output-wrapper-main") {
                FleurRight {}
                span(id="output-main") { (main_str.get()) }

                (if let Some(dice_str) = maybe_dice_str.get().as_ref().clone() {
                    view! { ctx,
                        span(id="output-dice") { (dice_str) }
                    }
                } else {
                    View::empty()
                })

                FleurLeft {}
            }

            div(id="output-wrapper-err") {
                (if let Some(err_str) = maybe_err_str.get().as_ref().clone() {
                    view! { ctx,
                        span(id="output-err") { (err_str) }
                    }
                } else {
                    View::empty()
                })
            }
        }
    }
}

#[component]
fn DiceMemnonicsTable<G: Html>(ctx: ScopeRef) -> View<G> {
    let mut die_kinds = kcddice::dice::DieKind::all().to_vec();
    die_kinds.sort_unstable_by_key(|die_kind| die_kind.as_memnonic());

    let entries = View::new_fragment(
        die_kinds
            .into_iter()
            .map(|die_kind| {
                let memnonic = if die_kind == kcddice::dice::DieKind::Standard {
                    "s"
                } else {
                    die_kind.as_memnonic()
                };
                let name = die_kind.as_human_readable();

                view! { ctx,
                    div(class="entry") {
                        span(class="memnonic") { (memnonic) }
                        span(class="name") { (name) }
                    }
                }
            })
            .collect::<Vec<_>>(),
    );

    view! { ctx,
        div(id="memnonics-table") {
            (entries)
        }
    }
}

#[component]
fn Info<G: Html>(ctx: ScopeRef) -> View<G> {
    view! { ctx,
        section(id="info-wrapper") {
            h3 {
                "Supported Dice"
            }

            DiceMemnonicsTable {}

            ul(id="links") {
                li { a(href="https://github.com/phlip9/kcddice#About") { "about" } }
                li { a(href="https://github.com/phlip9/kcddice") { "github" } }
                li { a(href="https://reddit.com/u/phlip9") { "reddit post" } }
            }
        }
    }
}
