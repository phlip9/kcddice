use futures_util::future;
use kcddice::{
    cli::{BestActionCommand, BestActionCommandOutput, Command, Metrics},
    dice::DieKindTable,
    parse,
    search::{Action, ActionValue},
};
use log::{debug, trace, warn};
use std::{cell::Cell, rc::Rc, str::FromStr, time::Duration};
use sycamore::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{Event, HtmlButtonElement, HtmlInputElement, KeyboardEvent};

pub mod cmd_worker;

#[derive(Clone, PartialEq, Eq)]
pub struct Todo {
    id: u32,
    title: String,
    completed: bool,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Filter {
    All,
    Active,
    Completed,
}

impl Default for Filter {
    fn default() -> Self {
        Self::All
    }
}

pub struct AppState {
    pub todos: RcSignal<Vec<RcSignal<Todo>>>,
    pub filter: RcSignal<Filter>,
    next_id: Cell<u32>,
}

impl AppState {
    fn todos_vec(&self) -> Vec<RcSignal<Todo>> {
        self.todos.get().as_ref().clone()
    }

    fn next_id(&self) -> u32 {
        let id = self.next_id.get();
        self.next_id.set(id + 1);
        id
    }

    fn add_todo(&self, title: String) {
        let mut todos_vec = self.todos_vec();
        todos_vec.push(create_rc_signal(Todo {
            id: self.next_id(),
            title,
            completed: false,
        }));

        self.todos.set(todos_vec);
    }

    fn remove_todo(&self, id: u32) {
        self.todos.set(
            self.todos_vec()
                .into_iter()
                .filter(|todo| todo.get().id != id)
                .collect(),
        )
    }

    fn todos_left(&self) -> usize {
        self.todos
            .get()
            .iter()
            .filter(|todo| !todo.get().completed)
            .count()
    }

    fn set_all_todos_state(&self, completed: bool) {
        for todo in self.todos.get().iter() {
            todo.set(Todo {
                completed,
                ..todo.get().as_ref().clone()
            });
        }
    }

    fn toggle_complete_all(&self) {
        let all_complete = self.todos_left() == 0;
        let set_complete = !all_complete;
        self.set_all_todos_state(set_complete);
    }

    fn clear_completed(&self) {
        self.todos.set(
            self.todos_vec()
                .into_iter()
                .filter(|todo| !todo.get().completed)
                .collect(),
        )
    }
}

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

        let best_action_state = BestActionOutputState {
            is_searching: create_rc_signal(false),
            best_action: create_rc_signal(best_action),
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
    // pub fn App<'a, G: Html>(ctx: ScopeRef<'a>) -> View<G> {
    debug!("init App component: props: {:?}", props);
    // debug!("init App component");

    // let app_state = AppState {
    //     todos: create_rc_signal(Vec::new()),
    //     filter: create_rc_signal(Filter::default()),
    //     next_id: Cell::new(0),
    // };
    //
    // ctx.provide_context(app_state);

    view! { ctx,
        // central container
        div(class="page-wrapper") {
            Title {}

            main {
                section(id="inputs-page") {
                    BestActionPage(props.best_action_props)
                    // BestActionPage {}
                }
            }
        }

        // build classic form { (label, input) }
        // bind input values
        // hook form submission, event.preventDefault
        // pull out input values from binds

        // BestAction {}

        // div(class="todomvc-wrapper") {
        //     section(class="todoapp") {
        //         Header {}
        //         List {}
        //         Footer {}
        //     }
        //     Copyright {}
        // }
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

// fn to_js_elt<G: GenericNode, T: JsCast>(node_ref: &NodeRef<G>) -> T {
//     node_ref
//         .get::<DomNode>()
//         .inner_element()
//         .dyn_into::<T>()
//         .unwrap()
// }
//
// fn to_input_elt<G: GenericNode>(node_ref: &NodeRef<G>) -> HtmlInputElement {
//     to_js_elt::<G, HtmlInputElement>(node_ref)
// }
//
// fn to_button_elt<G: GenericNode>(node_ref: &NodeRef<G>) -> HtmlButtonElement {
//     to_js_elt::<G, HtmlButtonElement>(node_ref)
// }

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
    // fn BestActionPage<'a, G: Html>(ctx: ScopeRef<'a>) -> View<G> {
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

        let maybe_cmd = BestActionCommand::try_from_str_args(
            empty_str_to_opt(starting_dice_str.as_str()),
            empty_str_to_opt(total_score_str.as_str()),
            empty_str_to_opt(max_score_str.as_str()),
            round_score_str.as_str(),
            rolled_dice_str.as_str(),
        );

        let cmd = match maybe_cmd {
            Ok(cmd) => {
                debug!("BestActionPage::handle_submit: cmd: {cmd:?}");
                cmd
            }
            Err(err) => {
                warn!("BestActionPage::handle_submit: invalid cmd args: {err}");
                return;
            }
        };

        let best_action_state = props_clone.best_action_state.clone();
        sycamore::futures::spawn_local(async move {
            trace!("BestActionPage::handle_submit::async: sending request to worker");

            // ensure the result only fulfills at least 0.35sec after we start,
            // to give the slide-up animation time to complete.
            let f_sleep = gloo_timers::future::sleep(Duration::from_secs_f32(0.35));
            let f_request = crate::cmd_worker::CmdWorker::request(cmd);
            let (_, result) = future::join(f_sleep, f_request).await;

            let best_action = match result {
                Ok(best_action) => {
                    debug!("BestActionPage::handle_submit::async: successful response from worker: {best_action}");
                    best_action
                }
                Err(err) => {
                    warn!(
                        "BestActionPage::handle_submit::async: error response from worker: {err}"
                    );
                    return;
                }
            };

            best_action_state.is_searching.set(false);
            best_action_state.best_action.set(best_action);
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
                    bind:value=props.starting_dice_str,
                )
            }
            div(class="input-wrapper") {
                label(for="total-score") { "total score" }
                input(
                    ref=total_score_input,
                    id="total-score",
                    name="total-score",
                    bind:value=props.total_score_str,
                )
                span(id="total-max-sep") { "/" }
                input(
                    ref=max_score_input,
                    id="max-score",
                    name="max-score",
                    bind:value=props.max_score_str,
                )
            }
            div(class="input-wrapper") {
                label(for="round-score") { "round score" }
                input(
                    ref=round_score_input,
                    id="round-score",
                    name="round-score",
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
    best_action: RcSignal<parse::Action>,
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
    let action_str = ctx.create_memo(move || state_clone.best_action.get().to_action_str());

    let state_clone = state.clone();
    let maybe_dice_str = ctx.create_memo(move || state_clone.best_action.get().to_maybe_dice_str());

    view! { ctx,
        section(id="output", class=output_class.get()) {
            FleurRight {}
            span(id="output-action") { (action_str.get()) }
            (if let Some(dice_str) = maybe_dice_str.get().as_ref().clone() {
                view! { ctx,
                    span(id="output-dice") { (dice_str) }
                }
            } else {
                View::empty()
            })
            FleurLeft {}
        }
    }
}

fn empty_to_opt(s: Rc<String>) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s.as_ref().clone())
    }
}

#[component]
fn MetricsList<'a, G: Html>(ctx: ScopeRef<'a>, metrics: &'a ReadSignal<Metrics>) -> View<G> {
    debug!("init MetricsList component");

    let metrics = ctx.create_memo(|| metrics.get().0.clone());

    view! { ctx,
        ul {
            Indexed {
                iterable: metrics,
                view: |ctx, metric_row| {
                    view! { ctx,
                        li {
                            span { (metric_row.0) }
                            span { (metric_row.1) }
                        }
                    }
                }
            }
        }
    }
}

#[derive(Prop)]
struct ActionValueRowProps {
    row: ActionValue,
    dice_table: DieKindTable,
}

#[component]
fn ActionValueRow<G: Html>(ctx: ScopeRef, props: ActionValueRowProps) -> View<G> {
    debug!("init ActionValueRow component");

    let cols = View::new_fragment(
        props
            .row
            .to_row_cells(&props.dice_table)
            .into_iter()
            .map(|col| {
                view! { ctx,
                    td {
                        (col)
                    }
                }
            })
            .collect::<Vec<_>>(),
    );

    view! { ctx,
        (cols)
    }
}

#[derive(Prop)]
struct ActionValuesTableProps<'a> {
    rows: &'a ReadSignal<Vec<ActionValue>>,
    dice_table: &'a ReadSignal<DieKindTable>,
}

#[component]
fn ActionValuesTable<'a, G: Html>(ctx: ScopeRef<'a>, props: ActionValuesTableProps<'a>) -> View<G> {
    debug!("init ActionValuesTable component");

    let dice_table = *props.dice_table.get();

    view! { ctx,
        table {
            Indexed {
                iterable: props.rows,
                view: move |ctx, row| {
                    view! { ctx,
                        tr {
                            ActionValueRow {
                                row: row,
                                dice_table: dice_table,
                            }
                        }
                    }
                },
            }
        }
    }
}

#[component]
fn BestAction<G: Html>(ctx: ScopeRef) -> View<G> {
    debug!("init BestAction component");

    let round_total_input = ctx.create_node_ref();
    let round_total_value = ctx.create_signal(String::new());

    let target_score_input = ctx.create_node_ref();
    let target_score_value = ctx.create_signal(String::new());

    let rolled_dice_input = ctx.create_node_ref();
    let rolled_dice_value = ctx.create_signal(String::new());

    let all_dice_input = ctx.create_node_ref();
    let all_dice_value = ctx.create_signal(String::new());

    let cmd_out = ctx.create_signal(BestActionCommandOutput::default());

    // hit enter in any input
    // -> parse inputs (for now)
    // -> submit job
    let handle_submit = |event: Event| {
        let event: KeyboardEvent = event.unchecked_into();
        let key = event.key();

        trace!("BestAction::handle_submit: event.key: {key}");

        if key != "Enter" {
            return;
        }

        let round_total = round_total_value.get().as_ref().clone();
        let target_score = empty_to_opt(target_score_value.get());
        let rolled_dice = rolled_dice_value.get().as_ref().clone();
        let all_dice = empty_to_opt(all_dice_value.get());

        debug!("BestAction::handle_submit: round_total: '{round_total}', target_score: {target_score:?}, rolled_dice: '{rolled_dice}', all_dice: {all_dice:?}");

        let maybe_cmd = BestActionCommand::try_from_str_args(
            all_dice.as_deref(),
            Some("0"),
            target_score.as_deref(),
            &round_total,
            &rolled_dice,
        );

        let cmd = match maybe_cmd {
            Ok(cmd) => {
                debug!("BestAction::handle_submit: cmd: {cmd:?}");
                cmd
            }
            Err(err) => {
                warn!("BestAction::handle_submit: invalid cmd args: {err}");
                return;
            }
        };

        let out = match cmd.run() {
            Ok(out) => {
                debug!("BestAction::handle_submit: out: {out}");
                out
            }
            Err(err) => {
                warn!("BestAction::handle_submit: error running cmd: {err}");
                return;
            }
        };

        cmd_out.set(out);
    };

    let rows = ctx.create_memo(|| cmd_out.get().action_values.0.clone());
    let dice_table = ctx.create_memo(|| cmd_out.get().dice_table);
    let metrics = ctx.create_memo(|| cmd_out.get().metrics.clone());

    view! { ctx,
        div(class="kcddice-inputs") {
            input(
                ref=target_score_input,
                placeholder="4000",
                bind:value=target_score_value,
                on:keyup=handle_submit,
            )

            input(
                ref=all_dice_input,
                placeholder="s:3 hk:2 o:1",
                bind:value=all_dice_value,
                on:keyup=handle_submit,
            )

            input(
                ref=round_total_input,
                placeholder="0",
                bind:value=round_total_value,
                on:keyup=handle_submit,
            )

            input(
                ref=rolled_dice_input,
                placeholder="1 1hk 2 3o 3 5hk",
                bind:value=rolled_dice_value,
                on:keyup=handle_submit,
            )
        }

        ActionValuesTable {
            rows: rows,
            dice_table: dice_table,
        }

        MetricsList(metrics)
    }
}

#[component]
fn Header<G: Html>(ctx: ScopeRef) -> View<G> {
    let app_state = ctx.use_context::<AppState>();
    let input_value = ctx.create_signal(String::new());
    let input_ref = ctx.create_node_ref();

    let handle_submit = |event: Event| {
        let event: KeyboardEvent = event.unchecked_into();

        if event.key() == "Enter" {
            let task = input_value.get().as_ref().trim().to_string();
            if !task.is_empty() {
                app_state.add_todo(task);
                input_value.set("".to_string());
                input_ref
                    .get::<DomNode>()
                    .unchecked_into::<HtmlInputElement>()
                    .set_value("");
            }
        }
    };

    view! { ctx,
        header(class="header") {
            h1 { "todos" }
            input(
                ref=input_ref,
                class="new-todo",
                placeholder="what needs to be done",
                bind:value=input_value,
                on:keyup=handle_submit,
            )
        }
    }
}

#[component]
fn Item<G: Html>(ctx: ScopeRef, todo: RcSignal<Todo>) -> View<G> {
    let app_state = ctx.use_context::<AppState>();

    let todo = ctx.create_ref(todo);

    let title = || todo.get().title.clone();
    let completed = ctx.create_selector(|| todo.get().completed);
    let id = todo.get().id;

    let editing = ctx.create_signal(false);
    let input_value = ctx.create_signal(String::new());
    let input_ref = ctx.create_node_ref();

    let handle_input = |event: Event| {
        let target: HtmlInputElement = event.target().unwrap().unchecked_into();
        input_value.set(target.value());
    };

    let toggle_completed = |_| {
        todo.set(Todo {
            completed: !todo.get().completed,
            ..todo.get().as_ref().clone()
        });
    };

    let handle_dblclick = move |_| {
        editing.set(true);
        input_ref
            .get::<DomNode>()
            .unchecked_into::<HtmlInputElement>()
            .focus()
            .unwrap();
        input_value.set(title());
    };

    let handle_blur = move || {
        editing.set(false);

        let value = input_value.get().trim().to_string();
        if value.is_empty() {
            app_state.remove_todo(id);
        } else {
            todo.set(Todo {
                title: value,
                ..todo.get().as_ref().clone()
            });
        }
    };

    let handle_submit = move |event: Event| {
        let event: KeyboardEvent = event.unchecked_into();
        match event.key().as_str() {
            "Enter" => handle_blur(),
            "Escape" => {
                input_ref
                    .get::<DomNode>()
                    .unchecked_into::<HtmlInputElement>()
                    .set_value(&title());
                editing.set(false);
            }
            _ => {}
        }
    };

    let handle_destroy = move |_| {
        app_state.remove_todo(id);
    };

    let checked = ctx.create_signal(false);
    ctx.create_effect(|| {
        checked.set(*completed.get());
    });

    let class = || {
        format!(
            "{} {}",
            if *completed.get() { "completed" } else { "" },
            if *editing.get() { "editing" } else { "" },
        )
    };

    view! { ctx,
        li(class=class()) {
            div(class="view") {
                input(
                    class="toggle",
                    type="checkbox",
                    on:input=toggle_completed,
                    bind:checked=checked,
                )

                label(on:dblclick=handle_dblclick) {
                    (title())
                }

                button(class="destroy", on:click=handle_destroy)
            }

            (if *editing.get() {
                view! { ctx,
                    input(
                        ref=input_ref,
                        class="edit",
                        value=todo.get().title.clone(),
                        on:blur=move |_| handle_blur(),
                        on:keyup=handle_submit,
                        on:input=handle_input,
                    )
                }
            } else {
                View::empty()
            })
        }
    }
}

#[component]
fn List<G: Html>(ctx: ScopeRef) -> View<G> {
    let app_state = ctx.use_context::<AppState>();
    let todos_left = ctx.create_selector(|| app_state.todos_left());

    let filtered_todos = ctx.create_memo(|| {
        app_state
            .todos
            .get()
            .iter()
            .filter(|todo| match *app_state.filter.get() {
                Filter::All => true,
                Filter::Active => !todo.get().completed,
                Filter::Completed => todo.get().completed,
            })
            .cloned()
            .collect::<Vec<_>>()
    });

    let checked = ctx.create_signal(false);
    ctx.create_effect(|| {
        checked.set(*todos_left.get() == 0);
    });

    view! { ctx,
        section(class="main") {
            input(
                id="toggle-all",
                class="toggle-all",
                type="checkbox",
                readonly=true,
                bind:checked=checked,
                on:input=|_| app_state.toggle_complete_all(),
            )
            label(for="toggle-all")

            ul(class="todo-list") {
                Keyed {
                    iterable: filtered_todos,
                    view: |ctx, todo| view! { ctx,
                        Item(todo)
                    },
                    key: |todo| todo.get().id,
                }
            }
        }
    }
}

#[component]
fn TodoFilter<G: Html>(ctx: ScopeRef, filter: Filter) -> View<G> {
    let app_state = ctx.use_context::<AppState>();
    let selected = move || filter == *app_state.filter.get();
    let set_filter = |filter| app_state.filter.set(filter);

    view! { ctx,
        li {
            a(
                class=if selected() { "selected "} else { "" },
                on:click=move |_| set_filter(filter),
            ) {
                (format!("{filter:?}"))
            }
        }
    }
}

#[component]
fn Footer<G: Html>(ctx: ScopeRef) -> View<G> {
    let app_state = ctx.use_context::<AppState>();

    let items_text = || match app_state.todos_left() {
        1 => "item",
        _ => "items",
    };

    let has_completed_todos =
        ctx.create_selector(|| app_state.todos_left() < app_state.todos.get().len());

    let handle_clear_completed = |_| app_state.clear_completed();

    view! { ctx,
        footer(class="footer") {
            span(class="todo-count") {
                strong { (app_state.todos_left()) }
                span { (format!(" {} left", items_text())) }
            }
            ul(class="filters") {
                TodoFilter(Filter::All)
                TodoFilter(Filter::Active)
                TodoFilter(Filter::Completed)
            }

            (if *has_completed_todos.get() {
                view! { ctx,
                    button(class="clear-completed", on:click=handle_clear_completed) {
                        "Clear completed"
                    }
                }
            } else {
                View::empty()
            })
        }
    }
}

#[component]
fn Copyright<G: Html>(ctx: ScopeRef) -> View<G> {
    view! { ctx,
        footer(class="info") {
            p { "Double click to edit a todo" }
        }
    }
}