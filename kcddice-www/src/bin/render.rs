use kcddice_www::App;
use std::{
    io::{self, Read, Write},
    time::Instant,
};
use sycamore::prelude::*;

// 1. Reads html file from stdin.
// 2. Looks for a `<!-- HYDRATION-STATION -->` comment.
// 3. If it's present, replaces it with the main pre-rendered sycamore app
//    and dumps that to stdout.
fn main() {
    eprintln!("kcddice-www::render: start");
    let start = Instant::now();

    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .expect("Failed to read from stdin");

    let mut parts = input.split("<!-- HYDRATION-STATION -->").fuse();

    let (pre, post, end) = (parts.next(), parts.next(), parts.next());

    let (output, did_something) = match (pre, post, end) {
        // empty or no marker; do nothing and just return the input.
        (None, _, _) | (Some(_), None, _) => (input, false),
        // found a marker. we'll pre-render the page and insert it where the
        // marker was.
        (Some(pre), Some(post), None) => {
            let prerender = sycamore::render_to_string(|ctx| {
                view! { ctx,
                    App {}
                }
            });

            ([pre, &prerender, post].join("\n"), true)
        }
        (Some(_), Some(_), Some(_)) => panic!("too many hydration markers"),
    };

    io::stdout()
        .write_all(output.as_bytes())
        .expect("Failed to write to stdout");

    eprintln!(
        "kcddice-www::render: end, did something: {:?}, time: {:?}",
        did_something,
        start.elapsed(),
    );
}
