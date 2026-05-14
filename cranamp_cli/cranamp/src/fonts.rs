/// Cranpose/glyphon need at least one application-provided face on wasm and
/// mobile targets because system font discovery is unavailable or disabled.
const LIBERATION_SANS_REGULAR: &[u8] = include_bytes!("../assets/fonts/LiberationSans-Regular.ttf");

pub const APP_FONTS: &[&[u8]] = &[LIBERATION_SANS_REGULAR];
