use kcddice::{
    cli::{BestActionCommand, BestActionCommandOutput, Command, Metrics},
    dice::DieKindTable,
    parse,
    search::{Action, ActionValue},
};
use log::{debug, trace, warn};
use std::{cell::Cell, rc::Rc, str::FromStr};
use sycamore::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{Event, HtmlInputElement, KeyboardEvent};

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

#[component]
pub fn App<G: Html>(ctx: ScopeRef) -> View<G> {
    debug!("init App component");

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
                    BestActionPage {}
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
fn Title<G: Html>(ctx: ScopeRef) -> View<G> {
    view! { ctx,
        header {
            h1(dangerously_set_inner_html="<svg><use xlink:href=\"imgs/logo-kcd.svg#logo-kcd\"/></svg>")
            h2 {
                i(dangerously_set_inner_html="<svg><use xlink:href=\"imgs/fleur-left.svg#fleur-left\"/></svg>")
                "optimal dice strategy"
                i(dangerously_set_inner_html="<svg><use xlink:href=\"imgs/fleur-right.svg#fleur-right\"/></svg>")
            }
        }
    }
}

#[component]
fn BestActionPage<G: Html>(ctx: ScopeRef) -> View<G> {
    let dice_set = parse::DiceSet::from_str("s:3 hk:2 o:1").unwrap();
    let dice_table = dice_set.to_table();
    let dice_vec = parse::DiceVec::from_str("1").unwrap();
    let dice_vec = dice_vec.to_compact_form(&dice_table);
    let action = Action::Roll(dice_vec);
    let maybe_output = Some((action, dice_table));
    let maybe_output = ctx.create_signal(maybe_output);

    view! { ctx,
        form(id="best-action-form") {
            hr(class="page-lines")

            div(class="input-wrapper") {
                label(for="starting-dice") { "starting dice" }
                input(id="starting-dice", name="starting-dice", value="s:3 hk:2 o:1")
            }
            div(class="input-wrapper") {
                label(for="total-score") { "total score" }
                input(id="total-score", name="total-score", value="1500")
                span(id="total-max-sep") { "/" }
                input(id="max-score", name="max-score", value="4000")
            }
            div(class="input-wrapper") {
                label(for="round-score") { "round score" }
                input(id="round-score", name="round-score", value="550")
            }
            div(class="input-wrapper") {
                label(for="rolled-dice") { "rolled dice" }
                input(id="rolled-dice", name="rolled-dice", value="1 1hk 3 5hk 6 6o")
            }

            hr(class="page-lines")

            button(id="best-action-submit", name="best-action-submit", value="best action", type="submit") {
                "best action"
            }
        }

        BestActionOutput(maybe_output)
    }
}

#[component]
fn BestActionOutput<'a, G: Html>(
    ctx: ScopeRef<'a>,
    maybe_output: &'a ReadSignal<Option<(Action, DieKindTable)>>,
) -> View<G> {
    let fleur_right = view! { ctx,
        i(dangerously_set_inner_html="<svg><use xlink:href=\"imgs/fleur-right.svg#fleur-right\"/></svg>")
    };
    let fleur_left = view! { ctx,
        i(dangerously_set_inner_html="<svg><use xlink:href=\"imgs/fleur-left.svg#fleur-left\"/></svg>")
    };

    let (output_class, output_view) = match maybe_output.get().as_ref() {
        None => ("searching", View::empty()),
        Some((Action::Pass, _)) => {
            let output_view = view! { ctx,
                (fleur_right)
                span(id="output-action") { "Pass" }
                (fleur_left)
            };
            ("", output_view)
        }
        Some((Action::Roll(dice), dice_table)) => {
            let dice = parse::DiceVec::from_compact_form(dice_table, *dice);
            let dice_str = dice.to_string_clean();

            let output_view = view! { ctx,
                (fleur_right)
                span(id="output-action") { "Hold" }
                span(id="output-dice") { (dice_str) }
                (fleur_left)
            };

            ("", output_view)
        }
    };

    view! { ctx,
        section(id="output", class=output_class) {
            (output_view)
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
            &round_total,
            target_score.as_deref(),
            &rolled_dice,
            all_dice.as_deref(),
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
