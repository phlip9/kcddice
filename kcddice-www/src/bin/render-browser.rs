use kcddice_www::{App, AppProps};
use log::trace;
use sycamore::prelude::*;

fn main() {
    console_error_panic_hook::set_once();
    // console_log::init_with_level(log::Level::Debug).expect("Failed to init logger");
    console_log::init_with_level(log::Level::Trace).expect("Failed to init logger");

    trace!("render-browser: init");

    // we need to mount to an inner element b/c `trunk` puts some preload
    // stuff in the body, which gets overwritten if we just render to body.
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let root = document.get_element_by_id("sycamore-mount").unwrap();

    // let root = web_sys::window()
    //     .unwrap()
    //     .document()
    //     .unwrap()
    //     .query_selector("#sycamore-mount")
    //     .unwrap()
    //     .unwrap();

    // sycamore::hydrate_to(
    sycamore::render_to(
        |ctx| {
            view! { ctx,
                App(AppProps::init_placeholders(ctx))
                // App {}
            }
        },
        &root,
    );
}
