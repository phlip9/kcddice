use kcddice_www::App;
use log::debug;
use sycamore::prelude::*;

fn main() {
    console_error_panic_hook::set_once();
    // console_log::init_with_level(log::Level::Debug).expect("Failed to init logger");
    console_log::init_with_level(log::Level::Trace).expect("Failed to init logger");

    debug!("main");

    // we need to mount to an inner element b/c `trunk` puts some preload
    // stuff in the body, which gets overwritten if we just render to body.
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let mount = document.get_element_by_id("sycamore-mount").unwrap();

    sycamore::hydrate_to(
        |ctx| {
            view! { ctx,
                App {}
            }
        },
        &mount,
    );
}
