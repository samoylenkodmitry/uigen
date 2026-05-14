#![forbid(unsafe_code)]

fn main() {
    #[cfg(feature = "logging")]
    let _ = env_logger::try_init();

    cranamp::create_desktop_app().run_windows(cranamp::winamp::WinampStandaloneApp);
}
