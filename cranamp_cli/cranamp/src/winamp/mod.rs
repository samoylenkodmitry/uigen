//! Winamp skin renderer and Cranamp player surface.
//!
//! UI is intentionally split into per-control composables instead of a
//! monolithic draw pass so interactions and sprite mapping stay explicit.

#![allow(non_snake_case)]

mod skin;
mod sprites;

use std::cell::Cell;
use std::collections::HashSet;
use std::rc::Rc;
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use cranpose::{
    rememberWindowState, WindowAttachPolicy, WindowConfig, WindowGroup, WindowId,
    WindowModifierExt, WindowMoveMode, WindowNode, WindowResizeDirection, WindowState,
};
use cranpose_core::{self, MutableState};
use cranpose_foundation::text::{TextFieldState, TextRange};
use cranpose_foundation::PointerButton;
use cranpose_ui::text::{AnnotatedString, FontFamily, ParagraphStyle, TextOverflow, TextUnit};
#[cfg(target_os = "android")]
use cranpose_ui::BoxWithConstraints;
#[cfg(target_os = "android")]
use cranpose_ui::BoxWithConstraintsScope;
use cranpose_ui::{
    composable, current_density, BasicText, BasicTextField, Box, BoxSpec, Button, ButtonSpec,
    Canvas, Color, Column, ColumnSpec, Modifier, Point, PointerEventKind, PointerInputScope, Size,
    SpanStyle, Text, TextStyle,
};
use cranpose_ui_graphics::{Brush, ImageBitmap, Rect};

#[cfg(target_os = "android")]
use crate::android_bridge::{self, AndroidBridgeResult, AndroidLoadMode};
use crate::audio::{self, Track};
use skin::{load_skin, SkinPalette, VisColor, WinampSkin};
use sprites::*;
#[cfg(all(feature = "web", target_arch = "wasm32"))]
use wasm_bindgen::{JsCast, JsValue};

#[cfg(not(target_arch = "wasm32"))]
fn run_native_io<T>(work: impl FnOnce() -> T + Send + 'static, on_ui: impl FnOnce(T) + 'static)
where
    T: Send + 'static,
{
    let Some(runtime) = cranpose_core::current_runtime_handle() else {
        let value = work();
        on_ui(value);
        return;
    };
    let Some(continuation_id) = runtime.register_ui_cont(on_ui) else {
        return;
    };
    let dispatcher = runtime.dispatcher();
    std::thread::spawn(move || {
        let value = work();
        dispatcher.post_invoke(continuation_id, value);
    });
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlaybackState {
    Stopped,
    Playing,
    Paused,
}

#[derive(Clone, Debug)]
struct WinampState {
    closed: bool,
    playback: PlaybackState,
    shuffle: bool,
    repeat: bool,
    eq_visible: bool,
    playlist_visible: bool,
    eq_enabled: bool,
    eq_auto: bool,
    eq_values: [f32; 11],
    skin_path: Option<String>,
    shuffle_order: Vec<usize>,
    playlist_scroll: f32,
    playlist_visible_rows: usize,
    volume: f32,
    balance: f32,
    position: f32,
    elapsed_seconds: f32,
    duration_seconds: Option<f32>,
    title_marquee_phase: f32,
    status: String,
    playlist: Rc<Vec<Track>>,
    current_index: Option<usize>,
    selected_indices: Vec<usize>,
    selection_anchor: Option<usize>,
    playlist_last_click_index: Option<usize>,
    playlist_last_click_ms: u64,
    playlist_search_visible: bool,
    playlist_search_query: String,
    playlist_search_revision: u64,
}

impl PartialEq for WinampState {
    fn eq(&self, other: &Self) -> bool {
        self.closed == other.closed
            && self.playback == other.playback
            && self.shuffle == other.shuffle
            && self.repeat == other.repeat
            && self.eq_visible == other.eq_visible
            && self.playlist_visible == other.playlist_visible
            && self.eq_enabled == other.eq_enabled
            && self.eq_auto == other.eq_auto
            && self.eq_values == other.eq_values
            && self.skin_path == other.skin_path
            && self.shuffle_order == other.shuffle_order
            && self.playlist_scroll == other.playlist_scroll
            && self.playlist_visible_rows == other.playlist_visible_rows
            && self.volume == other.volume
            && self.balance == other.balance
            && self.position == other.position
            && self.elapsed_seconds == other.elapsed_seconds
            && self.duration_seconds == other.duration_seconds
            && self.title_marquee_phase == other.title_marquee_phase
            && self.status == other.status
            && Rc::ptr_eq(&self.playlist, &other.playlist)
            && self.current_index == other.current_index
            && self.selected_indices == other.selected_indices
            && self.selection_anchor == other.selection_anchor
            && self.playlist_last_click_index == other.playlist_last_click_index
            && self.playlist_last_click_ms == other.playlist_last_click_ms
            && self.playlist_search_visible == other.playlist_search_visible
            && self.playlist_search_query == other.playlist_search_query
            && self.playlist_search_revision == other.playlist_search_revision
    }
}

impl Default for WinampState {
    fn default() -> Self {
        Self {
            closed: false,
            playback: PlaybackState::Stopped,
            shuffle: false,
            repeat: false,
            eq_visible: true,
            playlist_visible: true,
            eq_enabled: true,
            eq_auto: false,
            eq_values: DEFAULT_EQ_VALUES,
            skin_path: None,
            shuffle_order: Vec::new(),
            playlist_scroll: 0.0,
            playlist_visible_rows: DEFAULT_PLAYLIST_VISIBLE_ROWS,
            volume: 0.72,
            balance: 0.5,
            position: 0.0,
            elapsed_seconds: 0.0,
            duration_seconds: None,
            title_marquee_phase: 0.0,
            status: "Stopped".to_string(),
            playlist: Rc::new(Vec::new()),
            current_index: None,
            selected_indices: Vec::new(),
            selection_anchor: None,
            playlist_last_click_index: None,
            playlist_last_click_ms: 0,
            playlist_search_visible: false,
            playlist_search_query: String::new(),
            playlist_search_revision: 0,
        }
    }
}

fn initial_winamp_state() -> WinampState {
    let mut state = load_saved_player_state()
        .map(restore_saved_player_state)
        .unwrap_or_default();
    if state.playlist.is_empty() {
        let tracks = audio::demo_playlist_tracks();
        if !tracks.is_empty() {
            state.current_index = Some(0);
            state.status = format!("Loaded {} Demo Track(s)", tracks.len());
            set_playlist_tracks(&mut state, tracks);
            set_playlist_selection(&mut state, [0]);
        }
    }
    refresh_shuffle_order(&mut state);
    state
}

fn set_playlist_tracks(state: &mut WinampState, tracks: Vec<Track>) {
    state.playlist = Rc::new(tracks);
}

fn playlist_tracks_mut(state: &mut WinampState) -> &mut Vec<Track> {
    Rc::make_mut(&mut state.playlist)
}

thread_local! {
    static PLAYLIST_SCROLL_DRAG_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

fn set_playlist_scroll_drag_active(active: bool) {
    PLAYLIST_SCROLL_DRAG_ACTIVE.with(|dragging| dragging.set(active));
}

fn playlist_scroll_drag_active() -> bool {
    PLAYLIST_SCROLL_DRAG_ACTIVE.with(Cell::get)
}

#[derive(Clone, Copy, PartialEq)]
enum WinampDragTarget {
    Inline(MutableState<Point>),
    #[cfg(any(target_os = "android", all(feature = "web", target_arch = "wasm32")))]
    Fixed(Point),
    NativeGroup,
}

#[derive(Clone, Copy, PartialEq)]
enum WinampCloseAction {
    SetStatus,
    CloseApp,
}

#[derive(Clone, Copy, PartialEq)]
enum WinampWindowSize {
    Fixed(Size),
    State(WindowState),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlaylistFooterMenu {
    Add,
    Remove,
    Select,
    Misc,
    List,
}

#[derive(Clone, Copy, PartialEq)]
struct PlaylistScrollHandles {
    thumb: MutableState<f32>,
    entries: MutableState<f32>,
    dragging: MutableState<bool>,
}

struct PlaylistVisibleTextCache {
    playlist: Option<Rc<Vec<Track>>>,
    width_bits: u32,
    start: usize,
    rows: usize,
    current_index: Option<usize>,
    text: Rc<AnnotatedString>,
    line_count: usize,
}

impl Default for PlaylistVisibleTextCache {
    fn default() -> Self {
        Self {
            playlist: None,
            width_bits: 0,
            start: usize::MAX,
            rows: 0,
            current_index: None,
            text: Rc::new(AnnotatedString::default()),
            line_count: 1,
        }
    }
}

#[derive(Clone, Copy)]
struct PlaylistMenuItem {
    label: &'static str,
    action: fn(MutableState<WinampState>),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct PlaylistClickModifiers {
    shift: bool,
    ctrl: bool,
}

impl PlaylistClickModifiers {
    fn any(self) -> bool {
        self.shift || self.ctrl
    }
}

const MAIN_TITLE_DRAG_HIT_AREA: SpriteRect = (16.0, 0.0, 228.0, 14.0);
const EQ_TITLE_DRAG_HIT_AREA: SpriteRect = (0.0, 0.0, 264.0, 14.0);
const MAIN_SKIN_CHOOSER_HIT_AREA: SpriteRect = (249.0, 79.0, 26.0, 33.0);
const CRANAMP_WINAMP_MAIN_TITLE: &str = "Cranamp Winamp";
const CRANAMP_WINAMP_EQUALIZER_TITLE: &str = "Cranamp Winamp Equalizer";
const CRANAMP_WINAMP_PLAYLIST_TITLE: &str = "Cranamp Winamp Playlist";
const WINAMP_DEFAULT_SCREEN_POSITION: Point = Point { x: 140.0, y: 120.0 };
const TITLE_MARQUEE_CHARS_PER_SECOND: f32 = 2.0;
const PLAYLIST_DOUBLE_CLICK_MS: u64 = 500;
const DEFAULT_PLAYLIST_VISIBLE_ROWS: usize = 19;
#[cfg(not(target_arch = "wasm32"))]
const PLAYLIST_THUMB_SCROLL_FRAME_MS: u64 = 16;
const PLAYLIST_SCROLL_HIT_PAD_X: f32 = 8.0;
const DEFAULT_EQ_VALUES: [f32; 11] = [0.5; 11];
const WINAMP_SYSTEM_FONT_SIZE: f32 = 8.25;
const WINAMP_SYSTEM_LINE_HEIGHT: f32 = 10.0;
const WINAMP_SYSTEM_TEXT_Y_ADJUST: f32 = -2.25;
const WINAMP_SYSTEM_MEASURE_CHAR_WIDTH: f32 = 4.95;
const WINAMP_PLAYLIST_FONT_SIZE: f32 = 9.25;
const WINAMP_PLAYLIST_LINE_HEIGHT: f32 = 11.0;
const WINAMP_PLAYLIST_TEXT_Y_ADJUST: f32 = -2.5;
const WINAMP_PLAYLIST_ROW_CHAR_WIDTH: f32 = 4.2;
const WINAMP_PLAYLIST_TEXT_X: f32 = 4.0;
const WINAMP_PLAYLIST_TEXT_Y: f32 = 1.0;
const WINAMP_PLAYLIST_SELECTION_HEIGHT: f32 = 9.0;
const WINAMP_PLAYLIST_SELECTION_Y_OFFSET: f32 = -2.0;
const WINAMP_SYSTEM_MARQUEE_CHAR_WIDTH: f32 = 4.125;

#[derive(Clone, Copy, PartialEq)]
struct SystemTextMetrics {
    font_size: f32,
    line_height: f32,
    y_adjust: f32,
}

const WINAMP_SYSTEM_TEXT_METRICS: SystemTextMetrics = SystemTextMetrics {
    font_size: WINAMP_SYSTEM_FONT_SIZE,
    line_height: WINAMP_SYSTEM_LINE_HEIGHT,
    y_adjust: WINAMP_SYSTEM_TEXT_Y_ADJUST,
};

const WINAMP_PLAYLIST_TEXT_METRICS: SystemTextMetrics = SystemTextMetrics {
    font_size: WINAMP_PLAYLIST_FONT_SIZE,
    line_height: WINAMP_PLAYLIST_LINE_HEIGHT,
    y_adjust: WINAMP_PLAYLIST_TEXT_Y_ADJUST,
};

fn srgb_to_linear_u8(channel: u8) -> f32 {
    let s = channel as f32 / 255.0;
    if s <= 0.04045 {
        s / 12.92
    } else {
        ((s + 0.055) / 1.055).powf(2.4)
    }
}

fn srgb_color(rgba: [u8; 4]) -> Color {
    Color(
        srgb_to_linear_u8(rgba[0]),
        srgb_to_linear_u8(rgba[1]),
        srgb_to_linear_u8(rgba[2]),
        rgba[3] as f32 / 255.0,
    )
}

#[derive(Clone, Copy, PartialEq)]
struct ControlRect {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    scale: f32,
}

impl ControlRect {
    fn new(x: f32, y: f32, width: f32, height: f32, scale: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
            scale,
        }
    }

    fn scaled_width(self) -> f32 {
        scaled(self.width, self.scale)
    }

    fn scaled_height(self) -> f32 {
        scaled(self.height, self.scale)
    }

    fn scaled_x(self) -> f32 {
        scaled(self.x, self.scale)
    }

    fn scaled_y(self) -> f32 {
        scaled(self.y, self.scale)
    }

    fn contains_scaled_point(self, point: Point) -> bool {
        point.x >= 0.0
            && point.x <= self.scaled_width()
            && point.y >= 0.0
            && point.y <= self.scaled_height()
    }
}

type WinampSkinState = MutableState<Result<WinampSkin, String>>;

impl WinampWindowSize {
    fn get(self) -> Size {
        match self {
            Self::Fixed(size) => size,
            Self::State(state) => state.size(),
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) struct WinampTabState {
    player: MutableState<WinampState>,
    detached: MutableState<bool>,
    inline_windows: WinampInlineWindowStates,
    peer_windows: WinampPeerWindowStates,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct WinampInlineWindowStates {
    main: MutableState<Point>,
    equalizer: MutableState<Point>,
    playlist: MutableState<Point>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
struct WinampPeerWindowStates {
    main: WindowState,
    equalizer: WindowState,
    playlist: WindowState,
}

#[derive(Clone, Copy)]
struct WinampWindowPlacement {
    title: &'static str,
    initial_position: WinampInitialWindowPosition,
    state: WindowState,
}

#[derive(Clone, Copy)]
#[allow(dead_code)]
enum WinampInitialWindowPosition {
    Host(Point),
    Screen(Point),
}

#[composable]
pub(crate) fn remember_winamp_tab_state() -> WinampTabState {
    let peer_windows = WinampPeerWindowStates {
        main: rememberWindowState(MAIN_WIDTH, MAIN_HEIGHT),
        equalizer: rememberWindowState(EQ_WIDTH, EQ_HEIGHT),
        playlist: rememberWindowState(PLAYLIST_WIDTH, PLAYLIST_HEIGHT),
    };
    remember_saved_window_config(peer_windows);

    WinampTabState {
        player: cranpose_core::useState(initial_winamp_state),
        detached: cranpose_core::useState(native_winamp_windows_available),
        inline_windows: WinampInlineWindowStates {
            main: cranpose_core::useState(|| Point::new(26.0, 22.0)),
            equalizer: cranpose_core::useState(|| Point::new(26.0, 142.0)),
            playlist: cranpose_core::useState(|| Point::new(26.0, 262.0)),
        },
        peer_windows,
    }
}

#[composable]
pub(crate) fn WinampTab(tab_state: WinampTabState) {
    let scale = ui_scale();
    let state = tab_state.player;
    let native_available = native_winamp_windows_available();
    let detached = native_available && tab_state.detached.get();
    let snapshot = state.get();
    let skin_state = remember_winamp_skin(state);
    WinampRuntimeEffects(state, tab_state.peer_windows, skin_state);
    let skin = match skin_state.get() {
        Ok(skin) => skin,
        Err(error) => {
            WinampSkinError(error);
            return;
        }
    };

    Column(
        Modifier::empty()
            .fill_max_size()
            .padding(10.0)
            .background(Color(0.05, 0.06, 0.08, 1.0))
            .rounded_corners(12.0),
        ColumnSpec::default(),
        move || {
            Text(
                format!(
                    "{} | pos {:>3.0}% vol {:>3.0}% bal {:>3.0}%",
                    snapshot.status,
                    snapshot.position * 100.0,
                    snapshot.volume * 100.0,
                    snapshot.balance * 100.0,
                ),
                Modifier::empty().padding(8.0),
                TextStyle::default(),
            );

            if native_available {
                DockToggleButton(tab_state.detached, detached);
            }

            if !detached {
                WinampInlineStage(
                    skin.clone(),
                    state,
                    skin_state,
                    tab_state.inline_windows,
                    scale,
                );
            } else {
                WinampNativeWindows(
                    skin.clone(),
                    state,
                    skin_state,
                    tab_state.inline_windows,
                    tab_state.peer_windows,
                    scale,
                    snapshot.clone(),
                );
            }
        },
    );
}

fn remember_winamp_skin(_state: MutableState<WinampState>) -> WinampSkinState {
    let skin_state = cranpose_core::useState(bundled_skin);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let skin_path = _state.get_non_reactive().skin_path;
        cranpose_core::remember(move || {
            if let Some(path) = skin_path {
                load_skin_file_background(
                    _state,
                    skin_state,
                    std::path::PathBuf::from(path),
                    false,
                    None,
                    false,
                );
            }
        });
    }
    skin_state
}

#[cfg(not(target_arch = "wasm32"))]
fn load_skin_file_background(
    state: MutableState<WinampState>,
    skin_state: WinampSkinState,
    path: std::path::PathBuf,
    persist_path: bool,
    status_on_success: Option<String>,
    show_error_in_status: bool,
) {
    let path_for_work = path.clone();
    let saved_skin_path = persist_path.then(|| path.to_string_lossy().to_string());
    run_native_io(
        move || load_skin_file(&path_for_work),
        move |result| match result {
            Ok(skin) => {
                skin_state.set(Ok(skin));
                if saved_skin_path.is_some() || status_on_success.is_some() {
                    state.update(|s| {
                        if let Some(path) = saved_skin_path {
                            s.skin_path = Some(path);
                        }
                        if let Some(status) = status_on_success {
                            s.status = status;
                        }
                    });
                }
            }
            Err(error) => {
                if show_error_in_status {
                    state.update(|s| s.status = format!("Skin Load Failed: {error}"));
                }
            }
        },
    );
}

fn bundled_skin() -> Result<WinampSkin, String> {
    let wsz = include_bytes!("../../assets/winamp.wsz");
    load_skin(wsz).map_err(|err| format!("{err:#}"))
}

#[cfg(not(target_arch = "wasm32"))]
fn load_skin_file(path: &std::path::Path) -> Result<WinampSkin, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{error}"))?;
    load_skin(&bytes).map_err(|err| format!("{err:#}"))
}

#[composable]
fn WinampRuntimeEffects(
    state: MutableState<WinampState>,
    peer_windows: WinampPeerWindowStates,
    skin_state: WinampSkinState,
) {
    PlaybackProgressEffect(state);
    AndroidPickerEffect(state, skin_state);
    WebSurfaceSizeEffect(state);
    PlayerStatePersistence(state);
    NativeWindowPersistence(peer_windows);
}

#[cfg(target_os = "android")]
#[composable]
fn AndroidPickerEffect(state: MutableState<WinampState>, skin_state: WinampSkinState) {
    cranpose_core::LaunchedEffectAsync!((), move |scope| {
        Box::pin(async move {
            let clock = scope.runtime().frame_clock();
            loop {
                if !scope.is_active() {
                    break;
                }
                let _ = clock.next_frame().await;
                if !scope.is_active() {
                    break;
                }
                handle_android_bridge_results(state, skin_state);
            }
        })
    });
}

#[cfg(not(target_os = "android"))]
#[composable]
fn AndroidPickerEffect(_state: MutableState<WinampState>, _skin_state: WinampSkinState) {}

#[cfg(all(feature = "web", target_arch = "wasm32"))]
#[composable]
fn WebSurfaceSizeEffect(state: MutableState<WinampState>) {
    let snapshot = state.get();
    cranpose_core::SideEffect(move || {
        let layout = web_stacked_layout(&snapshot);
        publish_web_surface_size(
            stacked_surface_width(layout),
            stacked_surface_height(&snapshot, layout),
        );
    });
}

#[cfg(not(all(feature = "web", target_arch = "wasm32")))]
#[composable]
fn WebSurfaceSizeEffect(_state: MutableState<WinampState>) {}

#[cfg(all(feature = "web", target_arch = "wasm32"))]
fn publish_web_surface_size(width: f32, height: f32) {
    let Some(window) = web_sys::window() else {
        return;
    };
    let Ok(callback) = js_sys::Reflect::get(&window, &JsValue::from_str("cranampApplySurfaceSize"))
        .and_then(|value| value.dyn_into::<js_sys::Function>().map_err(Into::into))
    else {
        return;
    };
    let _ = callback.call2(
        &window,
        &JsValue::from_f64(width as f64),
        &JsValue::from_f64(height as f64),
    );
}

#[composable]
fn PlaybackProgressEffect(state: MutableState<WinampState>) {
    cranpose_core::LaunchedEffectAsync!((), move |scope| {
        Box::pin(async move {
            let clock = scope.runtime().frame_clock();
            loop {
                if !scope.is_active() {
                    break;
                }
                let _ = clock.next_frame().await;
                if !scope.is_active() {
                    break;
                }
                sync_playback_progress(state);
            }
        })
    });
}

#[composable]
fn PlayerStatePersistence(state: MutableState<WinampState>) {
    let last_saved = cranpose_core::remember(|| None::<SavedPlayerState>);
    let last_key = cranpose_core::remember(|| None::<SavedPlayerStateKey>);
    let snapshot = state.get();
    let key = SavedPlayerStateKey::from_state(&snapshot);
    let config = if last_key.with(|last| last.as_ref() == Some(&key)) {
        None
    } else {
        let config = SavedPlayerState::from_state(&snapshot);
        Some((key, config))
    };
    cranpose_core::SideEffect(move || {
        let Some((key, config)) = config else { return };
        last_saved.update(|last| {
            if last.as_ref() != Some(&config) {
                save_player_state_background(config.clone());
                *last = Some(config);
            }
            last_key.replace(Some(key));
        });
    });
}

fn sync_playback_progress(state: MutableState<WinampState>) {
    let snapshot = state.get_non_reactive();
    if snapshot.playback != PlaybackState::Playing || playlist_scroll_drag_active() {
        return;
    }

    match audio::playback_progress() {
        Ok(Some(progress)) => {
            let finished = progress.finished;
            state.update(|s| {
                let duration = progress.duration_seconds.or(s.duration_seconds);
                let elapsed = normalized_elapsed_seconds(progress.elapsed_seconds, duration);
                s.elapsed_seconds = elapsed;
                s.duration_seconds = duration;
                s.position = progress_fraction(elapsed, duration);
                s.title_marquee_phase = elapsed * TITLE_MARQUEE_CHARS_PER_SECOND;
            });
            if finished {
                advance_finished_track(state);
            }
        }
        Ok(None) => {}
        Err(error) => state.update(|s| s.status = error),
    }
}

fn normalized_elapsed_seconds(elapsed: f32, duration: Option<f32>) -> f32 {
    let elapsed = elapsed.max(0.0);
    let Some(duration) = duration.filter(|duration| *duration > 0.0) else {
        return elapsed;
    };
    elapsed.min(duration)
}

fn progress_fraction(elapsed: f32, duration: Option<f32>) -> f32 {
    duration
        .filter(|duration| *duration > 0.0)
        .map(|duration| (elapsed / duration).clamp(0.0, 1.0))
        .unwrap_or(0.0)
}

#[composable]
pub fn WinampFullscreenApp() {
    WinampSurfaceApp();
}

#[composable]
pub fn WinampWidgetApp() {
    #[cfg(all(feature = "web", target_arch = "wasm32"))]
    {
        WinampWebStackedApp();
    }

    #[cfg(not(all(feature = "web", target_arch = "wasm32")))]
    {
        WinampSurfaceApp();
    }
}

#[cfg(target_os = "android")]
#[composable]
pub fn WinampAndroidApp() {
    let tab_state = remember_winamp_tab_state();
    let skin_state = remember_winamp_skin(tab_state.player);
    WinampRuntimeEffects(tab_state.player, tab_state.peer_windows, skin_state);
    let skin = match skin_state.get() {
        Ok(skin) => skin,
        Err(error) => {
            WinampSkinError(error);
            return;
        }
    };

    Box(
        Modifier::empty()
            .fill_max_size()
            .clip_to_bounds()
            .background(Color(0.02, 0.02, 0.03, 1.0)),
        BoxSpec::default(),
        move || {
            let skin_for_stack = skin.clone();
            BoxWithConstraints(Modifier::empty().fill_max_size(), move |scope| {
                let snapshot = tab_state.player.get();
                let layout = resizable_stacked_layout(
                    scope.max_width().0,
                    scope.max_height().0,
                    android_bridge::content_top_inset_dp(),
                    android_bridge::content_bottom_inset_dp(),
                    &snapshot,
                    ui_scale(),
                );
                WinampStackedStage(skin_for_stack.clone(), tab_state.player, skin_state, layout);
            });
        },
    );
}

#[cfg(all(feature = "web", target_arch = "wasm32"))]
#[composable]
pub fn WinampWebStackedApp() {
    let tab_state = remember_winamp_tab_state();
    let skin_state = remember_winamp_skin(tab_state.player);
    WinampRuntimeEffects(tab_state.player, tab_state.peer_windows, skin_state);
    let skin = match skin_state.get() {
        Ok(skin) => skin,
        Err(error) => {
            WinampSkinError(error);
            return;
        }
    };

    Box(
        Modifier::empty()
            .fill_max_size()
            .clip_to_bounds()
            .background(Color(0.02, 0.02, 0.03, 1.0)),
        BoxSpec::default(),
        move || {
            let skin_for_stack = skin.clone();
            let snapshot = tab_state.player.get();
            WinampStackedStage(
                skin_for_stack,
                tab_state.player,
                skin_state,
                web_stacked_layout(&snapshot),
            );
        },
    );
}

#[composable]
fn WinampSurfaceApp() {
    let tab_state = remember_winamp_tab_state();
    let skin_state = remember_winamp_skin(tab_state.player);
    WinampRuntimeEffects(tab_state.player, tab_state.peer_windows, skin_state);
    let skin = match skin_state.get() {
        Ok(skin) => skin,
        Err(error) => {
            WinampSkinError(error);
            return;
        }
    };

    Box(
        Modifier::empty()
            .fill_max_size()
            .clip_to_bounds()
            .background(Color(0.02, 0.02, 0.03, 1.0)),
        BoxSpec::default(),
        move || {
            WinampInlineStage(
                skin.clone(),
                tab_state.player,
                skin_state,
                tab_state.inline_windows,
                ui_scale(),
            );
        },
    );
}

#[composable]
fn WinampSkinError(error: String) {
    Column(
        Modifier::empty().padding(16.0),
        ColumnSpec::default(),
        move || {
            Text(
                "Failed to load Winamp skin",
                Modifier::empty(),
                TextStyle::default(),
            );
            Text(error.clone(), Modifier::empty(), TextStyle::default());
        },
    );
}

#[composable]
fn DockToggleButton(detached_state: MutableState<bool>, detached: bool) {
    Button(
        Modifier::empty()
            .padding(8.0)
            .background(Color(0.18, 0.34, 0.58, 1.0))
            .rounded_corners(8.0)
            .padding(8.0),
        ButtonSpec::default(),
        move || {
            detached_state.set(!detached_state.get_non_reactive());
        },
        move || {
            Text(
                if detached { "Dock" } else { "Undock" },
                Modifier::empty(),
                TextStyle::default(),
            );
        },
    );
}

#[composable]
fn WinampInlineStage(
    skin: WinampSkin,
    state: MutableState<WinampState>,
    skin_state: WinampSkinState,
    windows: WinampInlineWindowStates,
    scale: f32,
) {
    Box(
        Modifier::empty()
            .fill_max_size()
            .clip_to_bounds()
            .background(Color(0.02, 0.02, 0.03, 1.0))
            .rounded_corners(8.0),
        BoxSpec::default(),
        move || {
            MainWindow(
                skin.clone(),
                state,
                skin_state,
                WinampDragTarget::Inline(windows.main),
                WinampCloseAction::SetStatus,
                scale,
            );

            if state.get().eq_visible {
                EqualizerWindow(
                    skin.clone(),
                    state,
                    WinampDragTarget::Inline(windows.equalizer),
                    scale,
                );
            }

            if state.get().playlist_visible {
                PlaylistWindow(
                    skin.pledit.clone(),
                    skin.palette,
                    skin.display_text_color,
                    state,
                    WinampDragTarget::Inline(windows.playlist),
                    WinampWindowSize::Fixed(Size::new(PLAYLIST_WIDTH, PLAYLIST_HEIGHT)),
                    scale,
                );
            }
        },
    );
}

#[cfg(any(target_os = "android", all(feature = "web", target_arch = "wasm32")))]
#[composable]
fn WinampStackedStage(
    skin: WinampSkin,
    state: MutableState<WinampState>,
    skin_state: WinampSkinState,
    layout: AndroidStackedLayout,
) {
    let snapshot = state.get();
    let scale = layout.scale;
    let mut y = 0.0;
    let main_y = y;
    y += MAIN_HEIGHT;
    let equalizer_y = y;
    if snapshot.eq_visible {
        y += EQ_HEIGHT;
    }
    let playlist_y = y;
    if snapshot.playlist_visible {
        y += layout.playlist_height;
    }

    Box(
        Modifier::empty()
            .size_points(scaled(MAIN_WIDTH, scale), scaled(y, scale))
            .clip_to_bounds()
            .background(Color(0.02, 0.02, 0.03, 1.0))
            .offset(
                snap_to_pixel(layout.content_left_inset),
                snap_to_pixel(layout.content_top_inset),
            ),
        BoxSpec::default(),
        move || {
            MainWindow(
                skin.clone(),
                state,
                skin_state,
                WinampDragTarget::Fixed(Point::new(0.0, main_y)),
                WinampCloseAction::SetStatus,
                scale,
            );

            if snapshot.eq_visible {
                EqualizerWindow(
                    skin.clone(),
                    state,
                    WinampDragTarget::Fixed(Point::new(0.0, equalizer_y)),
                    scale,
                );
            }

            if snapshot.playlist_visible {
                PlaylistWindow(
                    skin.pledit.clone(),
                    skin.palette,
                    skin.display_text_color,
                    state,
                    WinampDragTarget::Fixed(Point::new(0.0, playlist_y)),
                    WinampWindowSize::Fixed(Size::new(
                        scaled(PLAYLIST_WIDTH, scale),
                        scaled(layout.playlist_height, scale),
                    )),
                    scale,
                );
            }

            #[cfg(target_os = "android")]
            AndroidWindowMoveTarget(MAIN_WIDTH, y, scale, snapshot.clone(), layout);
        },
    );
}

#[cfg(any(target_os = "android", all(feature = "web", target_arch = "wasm32")))]
#[derive(Clone, Copy, Debug, PartialEq)]
struct AndroidStackedLayout {
    scale: f32,
    playlist_height: f32,
    content_left_inset: f32,
    content_top_inset: f32,
}

#[cfg(any(target_os = "android", all(feature = "web", target_arch = "wasm32")))]
fn resizable_stacked_layout(
    available_width: f32,
    available_height: f32,
    content_top_inset: f32,
    content_bottom_inset: f32,
    snapshot: &WinampState,
    fallback_scale: f32,
) -> AndroidStackedLayout {
    let width_scale = if available_width.is_finite() && available_width > 0.0 {
        available_width / MAIN_WIDTH
    } else {
        fallback_scale
    };

    let base_height = MAIN_HEIGHT + if snapshot.eq_visible { EQ_HEIGHT } else { 0.0 };
    let content_top_inset = if available_height.is_finite() && available_height > 0.0 {
        content_top_inset.clamp(0.0, available_height)
    } else {
        content_top_inset.max(0.0)
    };
    let content_bottom_inset = if available_height.is_finite() && available_height > 0.0 {
        content_bottom_inset.clamp(0.0, available_height)
    } else {
        content_bottom_inset.max(0.0)
    };
    let vertical_insets = if available_height.is_finite() && available_height > 0.0 {
        (content_top_inset + content_bottom_inset).min(available_height)
    } else {
        content_top_inset + content_bottom_inset
    };
    let available_content_height = if available_height.is_finite() && available_height > 0.0 {
        (available_height - vertical_insets).max(0.0)
    } else {
        base_height + PLAYLIST_HEIGHT
    };
    let minimum_skin_height = base_height
        + if snapshot.playlist_visible {
            playlist_min_height()
        } else {
            0.0
        };
    let height_scale = if available_content_height > 0.0 && minimum_skin_height > 0.0 {
        available_content_height / minimum_skin_height
    } else {
        fallback_scale
    };
    let scale = width_scale.min(height_scale).clamp(0.5, 4.0);
    let content_left_inset = if available_width.is_finite() && available_width > 0.0 {
        ((available_width - MAIN_WIDTH * scale).max(0.0)) * 0.5
    } else {
        0.0
    };
    let playlist_height = if snapshot.playlist_visible {
        let available_skin_height = if available_height.is_finite() && available_height > 0.0 {
            available_content_height / scale.max(f32::EPSILON)
        } else {
            base_height + PLAYLIST_HEIGHT
        };
        (available_skin_height - base_height).max(playlist_min_height())
    } else {
        PLAYLIST_HEIGHT
    };

    AndroidStackedLayout {
        scale,
        playlist_height,
        content_left_inset,
        content_top_inset,
    }
}

#[cfg(all(feature = "web", target_arch = "wasm32"))]
fn web_stacked_layout(snapshot: &WinampState) -> AndroidStackedLayout {
    let fallback_height = stacked_skin_height(snapshot, PLAYLIST_HEIGHT);
    let available =
        web_current_surface_size().unwrap_or_else(|| Size::new(MAIN_WIDTH, fallback_height));
    resizable_stacked_layout(
        available.width,
        available.height,
        0.0,
        0.0,
        snapshot,
        ui_scale(),
    )
}

#[cfg(all(feature = "web", target_arch = "wasm32"))]
fn web_current_surface_size() -> Option<Size> {
    let window = web_sys::window()?;
    let callback = js_sys::Reflect::get(&window, &JsValue::from_str("cranampCurrentSurfaceSize"))
        .ok()?
        .dyn_into::<js_sys::Function>()
        .ok()?;
    let size = callback.call0(&window).ok()?;
    let width = js_sys::Reflect::get(&size, &JsValue::from_str("width"))
        .ok()?
        .as_f64()? as f32;
    let height = js_sys::Reflect::get(&size, &JsValue::from_str("height"))
        .ok()?
        .as_f64()? as f32;
    (width.is_finite() && width > 0.0 && height.is_finite() && height > 0.0)
        .then_some(Size::new(width, height))
}

#[cfg(all(feature = "web", target_arch = "wasm32"))]
fn stacked_skin_height(snapshot: &WinampState, playlist_height: f32) -> f32 {
    MAIN_HEIGHT
        + if snapshot.eq_visible { EQ_HEIGHT } else { 0.0 }
        + if snapshot.playlist_visible {
            playlist_height
        } else {
            0.0
        }
}

#[cfg(all(feature = "web", target_arch = "wasm32"))]
fn stacked_surface_width(layout: AndroidStackedLayout) -> f32 {
    MAIN_WIDTH * layout.scale
}

#[cfg(all(feature = "web", target_arch = "wasm32"))]
fn stacked_surface_height(snapshot: &WinampState, layout: AndroidStackedLayout) -> f32 {
    stacked_skin_height(snapshot, layout.playlist_height) * layout.scale
}

#[cfg(target_os = "android")]
#[composable]
fn AndroidWindowMoveTarget(
    width: f32,
    height: f32,
    scale: f32,
    snapshot: WinampState,
    layout: AndroidStackedLayout,
) {
    let drag_start = cranpose_core::useState(|| None::<Point>);
    let drag_blocked = cranpose_core::useState(|| false);
    let moving = cranpose_core::useState(|| false);
    Box(
        Modifier::empty()
            .size_points(scaled(width, scale), scaled(height, scale))
            .pointer_input((), move |scope: PointerInputScope| {
                let snapshot = snapshot.clone();
                async move {
                    scope
                        .await_pointer_event_scope(|await_scope| async move {
                            loop {
                                let event = await_scope.await_pointer_event().await;
                                match event.kind {
                                    PointerEventKind::Down => {
                                        let interactive = android_stacked_interactive_area_contains(
                                            event.position,
                                            &snapshot,
                                            layout,
                                        );
                                        drag_start.set((!interactive).then_some(event.position));
                                        drag_blocked.set(interactive);
                                        moving.set(false);
                                    }
                                    PointerEventKind::Move => {
                                        if drag_blocked.get() {
                                            continue;
                                        }
                                        if !event.buttons.contains(PointerButton::Primary) {
                                            drag_start.set(None);
                                            moving.set(false);
                                            continue;
                                        }
                                        if moving.get() {
                                            event.consume();
                                            continue;
                                        }
                                        if let Some(start) = drag_start.get() {
                                            let dx = event.position.x - start.x;
                                            let dy = event.position.y - start.y;
                                            let threshold = scaled(3.0, scale).max(3.0);
                                            if dx * dx + dy * dy >= threshold * threshold {
                                                let started = android_bridge::start_window_move(
                                                    event.position.x + layout.content_left_inset,
                                                    event.position.y + layout.content_top_inset,
                                                );
                                                if started {
                                                    moving.set(true);
                                                    event.consume();
                                                } else {
                                                    drag_start.set(None);
                                                    drag_blocked.set(true);
                                                }
                                            }
                                        }
                                    }
                                    PointerEventKind::Up | PointerEventKind::Cancel => {
                                        drag_start.set(None);
                                        drag_blocked.set(false);
                                        moving.set(false);
                                    }
                                    PointerEventKind::Scroll
                                    | PointerEventKind::Enter
                                    | PointerEventKind::Exit => {}
                                }
                            }
                        })
                        .await;
                }
            }),
        BoxSpec::default(),
        || {},
    );
}

#[cfg(target_os = "android")]
fn android_stacked_interactive_area_contains(
    position: Point,
    snapshot: &WinampState,
    layout: AndroidStackedLayout,
) -> bool {
    let scale = layout.scale.max(f32::EPSILON);
    let x = position.x / scale;
    let mut y = position.y / scale;

    if y < MAIN_HEIGHT {
        return main_interactive_area_contains(x, y);
    }
    y -= MAIN_HEIGHT;

    if snapshot.eq_visible {
        if y < EQ_HEIGHT {
            return equalizer_interactive_area_contains(x, y);
        }
        y -= EQ_HEIGHT;
    }

    if snapshot.playlist_visible {
        return playlist_interactive_area_contains(x, y, PLAYLIST_WIDTH, layout.playlist_height);
    }

    false
}

#[cfg(target_os = "android")]
fn main_interactive_area_contains(x: f32, y: f32) -> bool {
    let rects = [
        rect_at(POS_OPTIONS_BUTTON, MAIN_OPTIONS_BUTTON),
        rect_at(POS_MINIMIZE_BUTTON, MAIN_MINIMIZE_BUTTON),
        rect_at(POS_SHADE_BUTTON, MAIN_SHADE_BUTTON),
        rect_at(POS_CLOSE_BUTTON, MAIN_CLOSE_BUTTON),
        rect(POS_POSBAR.0, POS_POSBAR.1, POSBAR_BG.2, POSBAR_BG.3),
        rect(
            POS_VOLUME.0,
            POS_VOLUME.1,
            VOLUME_BG_WIDTH,
            VOLUME_BG_HEIGHT,
        ),
        rect(
            POS_BALANCE.0,
            POS_BALANCE.1,
            BALANCE_BG_WIDTH,
            BALANCE_BG_HEIGHT,
        ),
        rect(POS_CBUTTONS.0, POS_CBUTTONS.1, 114.0, PREV_BUTTON.3),
        rect_at(POS_EJECT, EJECT_BUTTON),
        rect_at(POS_SHUFFLE, SHUFFLE_OFF),
        rect_at(POS_REPEAT, REPEAT_OFF),
        rect_at(POS_EQ_BUTTON, EQ_BUTTON_OFF),
        rect_at(POS_PL_BUTTON, PL_BUTTON_OFF),
        MAIN_SKIN_CHOOSER_HIT_AREA,
    ];
    rects.iter().any(|rect| rect_contains(*rect, x, y))
}

#[cfg(target_os = "android")]
fn equalizer_interactive_area_contains(x: f32, y: f32) -> bool {
    let button_rects = [
        rect_at(POS_EQ_CLOSE_BUTTON, EQ_CLOSE_BUTTON),
        rect_at(POS_EQ_ON_BUTTON, EQ_ON_BUTTON_OFF),
        rect_at(POS_EQ_AUTO_BUTTON, EQ_AUTO_BUTTON_OFF),
        rect_at(POS_EQ_PRESETS_BUTTON, EQ_PRESETS_BUTTON),
    ];
    button_rects.iter().any(|rect| rect_contains(*rect, x, y))
        || EQ_SLIDER_XS.iter().copied().any(|slider_x| {
            rect_contains(
                rect(
                    slider_x,
                    EQ_SLIDER_BG_Y,
                    EQ_SLIDER_BG.2,
                    EQ_SLIDER_TRACK_HEIGHT,
                ),
                x,
                y,
            )
        })
}

#[cfg(target_os = "android")]
fn playlist_interactive_area_contains(x: f32, y: f32, width: f32, height: f32) -> bool {
    let right_x = width - PLAYLIST_RIGHT_TILE.2;
    let bottom_y = height - PLAYLIST_BOTTOM_LEFT_CORNER.3;
    let list_width = (right_x - PLAYLIST_LIST_BG.0).max(1.0);
    let list_height = (bottom_y - PLAYLIST_LIST_BG.1).max(1.0);
    let scroll_track_x = width - 15.0;

    rect_contains(
        rect(
            PLAYLIST_LIST_BG.0,
            PLAYLIST_LIST_BG.1,
            list_width,
            list_height,
        ),
        x,
        y,
    ) || rect_contains(
        rect(
            scroll_track_x,
            PLAYLIST_LIST_BG.1,
            PLAYLIST_SCROLL_TRACK.2,
            list_height,
        ),
        x,
        y,
    ) || [
        offset_rect(PLAYLIST_ADD_BUTTON_HIT_AREA, 0.0, bottom_y),
        offset_rect(PLAYLIST_REM_BUTTON_HIT_AREA, 0.0, bottom_y),
        offset_rect(PLAYLIST_SEL_BUTTON_HIT_AREA, 0.0, bottom_y),
        offset_rect(PLAYLIST_MISC_BUTTON_HIT_AREA, 0.0, bottom_y),
        offset_rect(PLAYLIST_LIST_BUTTON_HIT_AREA, 0.0, bottom_y),
        offset_rect(PLAYLIST_PREV_BUTTON_HIT_AREA, 0.0, bottom_y),
        offset_rect(PLAYLIST_PLAY_BUTTON_HIT_AREA, 0.0, bottom_y),
        offset_rect(PLAYLIST_PAUSE_BUTTON_HIT_AREA, 0.0, bottom_y),
        offset_rect(PLAYLIST_STOP_BUTTON_HIT_AREA, 0.0, bottom_y),
        offset_rect(PLAYLIST_NEXT_BUTTON_HIT_AREA, 0.0, bottom_y),
        offset_rect(PLAYLIST_EJECT_BUTTON_HIT_AREA, 0.0, bottom_y),
    ]
    .iter()
    .any(|rect| rect_contains(*rect, x, y))
}

#[cfg(target_os = "android")]
fn rect_at(position: (f32, f32), sprite: SpriteRect) -> SpriteRect {
    rect(position.0, position.1, sprite.2, sprite.3)
}

#[cfg(target_os = "android")]
fn offset_rect(rect: SpriteRect, x: f32, y: f32) -> SpriteRect {
    (rect.0 + x, rect.1 + y, rect.2, rect.3)
}

#[cfg(target_os = "android")]
fn rect(x: f32, y: f32, width: f32, height: f32) -> SpriteRect {
    (x, y, width, height)
}

#[cfg(target_os = "android")]
fn rect_contains(rect: SpriteRect, x: f32, y: f32) -> bool {
    x >= rect.0 && x <= rect.0 + rect.2 && y >= rect.1 && y <= rect.1 + rect.3
}

#[composable]
fn WinampNativeWindows(
    skin: WinampSkin,
    state: MutableState<WinampState>,
    skin_state: WinampSkinState,
    inline_windows: WinampInlineWindowStates,
    peer_windows: WinampPeerWindowStates,
    scale: f32,
    snapshot: WinampState,
) {
    WindowGroup("cranamp-winamp", winamp_attach_policy(), move || {
        WindowNode(
            winamp_main_window_id(),
            winamp_window_config(WinampWindowPlacement {
                title: CRANAMP_WINAMP_MAIN_TITLE,
                initial_position: WinampInitialWindowPosition::Host(inline_windows.main.get()),
                state: peer_windows.main,
            }),
            {
                let skin = skin.clone();
                move || {
                    MainWindow(
                        skin.clone(),
                        state,
                        skin_state,
                        WinampDragTarget::NativeGroup,
                        WinampCloseAction::SetStatus,
                        scale,
                    );
                }
            },
        );

        if snapshot.eq_visible {
            WindowNode(
                winamp_equalizer_window_id(),
                winamp_window_config(WinampWindowPlacement {
                    title: CRANAMP_WINAMP_EQUALIZER_TITLE,
                    initial_position: WinampInitialWindowPosition::Host(
                        inline_windows.equalizer.get(),
                    ),
                    state: peer_windows.equalizer,
                }),
                {
                    let skin = skin.clone();
                    move || {
                        EqualizerWindow(skin.clone(), state, WinampDragTarget::NativeGroup, scale);
                    }
                },
            );
        }

        if snapshot.playlist_visible {
            WindowNode(
                winamp_playlist_window_id(),
                winamp_window_config(WinampWindowPlacement {
                    title: CRANAMP_WINAMP_PLAYLIST_TITLE,
                    initial_position: WinampInitialWindowPosition::Host(
                        inline_windows.playlist.get(),
                    ),
                    state: peer_windows.playlist,
                })
                .with_resizable(true)
                .with_min_size(
                    scaled(PLAYLIST_WIDTH, scale),
                    scaled(PLAYLIST_HEIGHT, scale),
                ),
                {
                    let pledit = skin.pledit.clone();
                    let palette = skin.palette;
                    let display_text_color = skin.display_text_color;
                    move || {
                        PlaylistWindow(
                            pledit.clone(),
                            palette,
                            display_text_color,
                            state,
                            WinampDragTarget::NativeGroup,
                            WinampWindowSize::State(peer_windows.playlist),
                            scale,
                        );
                    }
                },
            );
        }
    });
}

#[composable]
pub fn WinampStandaloneApp() {
    let state = cranpose_core::useState(initial_winamp_state);
    let peer_windows = WinampPeerWindowStates {
        main: rememberWindowState(MAIN_WIDTH, MAIN_HEIGHT),
        equalizer: rememberWindowState(EQ_WIDTH, EQ_HEIGHT),
        playlist: rememberWindowState(PLAYLIST_WIDTH, PLAYLIST_HEIGHT),
    };
    remember_saved_window_config(peer_windows);
    let skin_state = remember_winamp_skin(state);
    WinampRuntimeEffects(state, peer_windows, skin_state);
    let snapshot = state.get();
    if snapshot.closed {
        return;
    }
    let skin = match skin_state.get() {
        Ok(skin) => skin,
        Err(error) => {
            WinampSkinError(error);
            return;
        }
    };

    WindowGroup("cranamp-winamp", winamp_attach_policy(), move || {
        WindowNode(
            winamp_main_window_id(),
            winamp_window_config(WinampWindowPlacement {
                title: CRANAMP_WINAMP_MAIN_TITLE,
                initial_position: WinampInitialWindowPosition::Screen(default_main_position()),
                state: peer_windows.main,
            }),
            {
                let skin = skin.clone();
                move || {
                    MainWindow(
                        skin.clone(),
                        state,
                        skin_state,
                        WinampDragTarget::NativeGroup,
                        WinampCloseAction::CloseApp,
                        ui_scale(),
                    );
                }
            },
        );

        if snapshot.eq_visible {
            WindowNode(
                winamp_equalizer_window_id(),
                winamp_window_config(WinampWindowPlacement {
                    title: CRANAMP_WINAMP_EQUALIZER_TITLE,
                    initial_position: WinampInitialWindowPosition::Screen(
                        default_equalizer_position(),
                    ),
                    state: peer_windows.equalizer,
                }),
                {
                    let skin = skin.clone();
                    move || {
                        EqualizerWindow(
                            skin.clone(),
                            state,
                            WinampDragTarget::NativeGroup,
                            ui_scale(),
                        );
                    }
                },
            );
        }

        if snapshot.playlist_visible {
            WindowNode(
                winamp_playlist_window_id(),
                winamp_window_config(WinampWindowPlacement {
                    title: CRANAMP_WINAMP_PLAYLIST_TITLE,
                    initial_position: WinampInitialWindowPosition::Screen(
                        default_playlist_position(),
                    ),
                    state: peer_windows.playlist,
                })
                .with_resizable(true)
                .with_min_size(PLAYLIST_WIDTH, PLAYLIST_HEIGHT),
                {
                    let pledit = skin.pledit.clone();
                    let palette = skin.palette;
                    let display_text_color = skin.display_text_color;
                    move || {
                        PlaylistWindow(
                            pledit.clone(),
                            palette,
                            display_text_color,
                            state,
                            WinampDragTarget::NativeGroup,
                            WinampWindowSize::State(peer_windows.playlist),
                            ui_scale(),
                        );
                    }
                },
            );
        }
    });
}

#[composable]
fn MainWindow(
    skin: WinampSkin,
    state: MutableState<WinampState>,
    skin_state: WinampSkinState,
    drag_target: WinampDragTarget,
    close_action: WinampCloseAction,
    scale: f32,
) {
    let snapshot = state.get();

    Box(
        winamp_window_modifier(MAIN_WIDTH, MAIN_HEIGHT, scale, drag_target),
        BoxSpec::default(),
        move || {
            Sprite(skin.main.clone(), MAIN_WINDOW, 0.0, 0.0, scale);
            Sprite(
                skin.titlebar.clone(),
                MAIN_TITLE_BAR_SELECTED,
                0.0,
                0.0,
                scale,
            );

            WindowDragHandle(drag_target, MAIN_TITLE_DRAG_HIT_AREA, scale);

            {
                let state_click = state;
                let skin_click = skin_state;
                PressableSprite(
                    skin.titlebar.clone(),
                    MAIN_OPTIONS_BUTTON,
                    MAIN_OPTIONS_BUTTON_SELECTED,
                    POS_OPTIONS_BUTTON.0,
                    POS_OPTIONS_BUTTON.1,
                    scale,
                    move || {
                        open_skin_file(state_click, skin_click);
                    },
                );
            }
            {
                let state_click = state;
                PressableSprite(
                    skin.titlebar.clone(),
                    MAIN_MINIMIZE_BUTTON,
                    MAIN_MINIMIZE_BUTTON_SELECTED,
                    POS_MINIMIZE_BUTTON.0,
                    POS_MINIMIZE_BUTTON.1,
                    scale,
                    move || {
                        state_click.update(|s| s.status = "Minimize".to_string());
                    },
                );
            }
            {
                let state_click = state;
                PressableSprite(
                    skin.titlebar.clone(),
                    MAIN_SHADE_BUTTON,
                    MAIN_SHADE_BUTTON_SELECTED,
                    POS_SHADE_BUTTON.0,
                    POS_SHADE_BUTTON.1,
                    scale,
                    move || {
                        state_click.update(|s| s.status = "Shade".to_string());
                    },
                );
            }
            {
                let state_click = state;
                PressableSprite(
                    skin.titlebar.clone(),
                    MAIN_CLOSE_BUTTON,
                    MAIN_CLOSE_BUTTON_SELECTED,
                    POS_CLOSE_BUTTON.0,
                    POS_CLOSE_BUTTON.1,
                    scale,
                    move || {
                        state_click.update(|s| match close_action {
                            WinampCloseAction::SetStatus => {
                                s.status = "Close".to_string();
                            }
                            WinampCloseAction::CloseApp => {
                                s.closed = true;
                                s.status = "Closed".to_string();
                            }
                        });
                    },
                );
            }

            let status_sprite = match snapshot.playback {
                PlaybackState::Stopped => STATUS_STOPPED,
                PlaybackState::Playing => STATUS_PLAYING,
                PlaybackState::Paused => STATUS_PAUSED,
            };
            Sprite(
                skin.playpaus.clone(),
                status_sprite,
                POS_STATUS.0,
                POS_STATUS.1,
                scale,
            );

            Visualizer(
                snapshot.playback == PlaybackState::Playing,
                audio::visualizer_bands(),
                skin.viscolor,
                scale,
            );

            let digits = time_digits(snapshot.elapsed_seconds);
            for (i, digit) in digits.iter().enumerate() {
                let pos = POS_TIME_DIGITS[i];
                Sprite(
                    skin.numbers.clone(),
                    digit_rect(*digit),
                    pos.0,
                    pos.1,
                    scale,
                );
            }

            SystemWinampText(
                marquee_system_text(
                    main_display_title(&snapshot),
                    MAIN_TRACK_TEXT_WIDTH,
                    snapshot.title_marquee_phase,
                ),
                POS_MAIN_TRACK_TEXT.0,
                POS_MAIN_TRACK_TEXT.1,
                MAIN_TRACK_TEXT_WIDTH,
                WINAMP_SYSTEM_LINE_HEIGHT,
                scale,
                skin.display_text_color,
            );
            MainMetaReadouts(snapshot.clone(), scale, skin.display_text_color);

            Sprite(
                skin.monoster.clone(),
                MONO_OFF,
                POS_MONO.0,
                POS_MONO.1,
                scale,
            );
            Sprite(
                skin.monoster.clone(),
                if snapshot.current_index.is_some() {
                    STEREO_ON
                } else {
                    STEREO_OFF
                },
                POS_STEREO.0,
                POS_STEREO.1,
                scale,
            );

            Sprite(
                skin.posbar.clone(),
                POSBAR_BG,
                POS_POSBAR.0,
                POS_POSBAR.1,
                scale,
            );
            let posbar_pressed = cranpose_core::useState(|| false);
            let position_drag = cranpose_core::useState(|| None::<f32>);
            let display_position = position_drag.get().unwrap_or(snapshot.position);
            let position_thumb_x = slider_thumb_x(display_position, POSBAR_BG.2, POSBAR_THUMB.2);
            let posbar_thumb_sprite = if posbar_pressed.get() {
                POSBAR_THUMB_ACTIVE
            } else {
                POSBAR_THUMB
            };
            Sprite(
                skin.posbar.clone(),
                posbar_thumb_sprite,
                POS_POSBAR.0 + position_thumb_x,
                POS_POSBAR.1,
                scale,
            );
            {
                let position_drag_change = position_drag;
                let position_drag_commit = position_drag;
                let snapshot_duration = snapshot.duration_seconds;
                DragSlider(
                    ControlRect::new(POS_POSBAR.0, POS_POSBAR.1, POSBAR_BG.2, POSBAR_BG.3, scale),
                    move |fraction| {
                        position_drag_change.set(Some(fraction));
                    },
                    move |dragging| {
                        posbar_pressed.set(dragging);
                        if !dragging {
                            let Some(fraction) = position_drag_commit.get_non_reactive() else {
                                return;
                            };
                            position_drag_commit.set(None);
                            state.update(|s| {
                                s.position = fraction;
                                if let Some(duration) = s.duration_seconds.or(snapshot_duration) {
                                    s.elapsed_seconds = duration * fraction.clamp(0.0, 1.0);
                                }
                            });
                            if let Err(error) = audio::seek_fraction(fraction) {
                                state.update(|s| s.status = error);
                            }
                        }
                    },
                );
            }

            TransportButtons(skin.cbuttons.clone(), state, scale);

            let volume_pressed = cranpose_core::useState(|| false);
            let volume_drag = cranpose_core::useState(|| None::<f32>);
            let display_volume = volume_drag.get().unwrap_or(snapshot.volume);
            let vol_frame = slider_frame(display_volume, VOLUME_FRAMES);
            Sprite(
                skin.volume.clone(),
                (
                    0.0,
                    vol_frame as f32 * VOLUME_BG_STRIDE,
                    VOLUME_BG_WIDTH,
                    VOLUME_BG_HEIGHT,
                ),
                POS_VOLUME.0,
                POS_VOLUME.1,
                scale,
            );
            let volume_thumb_x = slider_thumb_x(display_volume, VOLUME_BG_WIDTH, VOLUME_THUMB.2);
            let volume_thumb_sprite = if volume_pressed.get() {
                VOLUME_THUMB_ACTIVE
            } else {
                VOLUME_THUMB
            };
            Sprite(
                skin.volume.clone(),
                volume_thumb_sprite,
                POS_VOLUME.0 + volume_thumb_x,
                POS_VOLUME.1 + 1.0,
                scale,
            );
            {
                let volume_drag_change = volume_drag;
                let volume_drag_commit = volume_drag;
                DragSlider(
                    ControlRect::new(
                        POS_VOLUME.0,
                        POS_VOLUME.1,
                        VOLUME_BG_WIDTH,
                        VOLUME_BG_HEIGHT,
                        scale,
                    ),
                    move |fraction| {
                        volume_drag_change.set(Some(fraction));
                        if let Err(error) = audio::set_volume(fraction) {
                            state.update(|s| s.status = error);
                        }
                    },
                    move |dragging| {
                        volume_pressed.set(dragging);
                        if !dragging {
                            let Some(fraction) = volume_drag_commit.get_non_reactive() else {
                                return;
                            };
                            volume_drag_commit.set(None);
                            state.update(|s| {
                                s.volume = fraction;
                            });
                        }
                    },
                );
            }

            let balance_pressed = cranpose_core::useState(|| false);
            let balance_drag = cranpose_core::useState(|| None::<f32>);
            let display_balance = balance_drag.get().unwrap_or(snapshot.balance);
            let bal_frame = slider_frame(display_balance, BALANCE_FRAMES);
            Sprite(
                skin.balance.clone(),
                (
                    BALANCE_BG_X,
                    bal_frame as f32 * BALANCE_BG_STRIDE,
                    BALANCE_BG_WIDTH,
                    BALANCE_BG_HEIGHT,
                ),
                POS_BALANCE.0,
                POS_BALANCE.1,
                scale,
            );
            let balance_thumb_x =
                slider_thumb_x(display_balance, BALANCE_BG_WIDTH, BALANCE_THUMB.2);
            let balance_thumb_sprite = if balance_pressed.get() {
                BALANCE_THUMB_ACTIVE
            } else {
                BALANCE_THUMB
            };
            Sprite(
                skin.balance.clone(),
                balance_thumb_sprite,
                POS_BALANCE.0 + balance_thumb_x,
                POS_BALANCE.1 + 1.0,
                scale,
            );
            {
                let balance_drag_change = balance_drag;
                let balance_drag_commit = balance_drag;
                DragSlider(
                    ControlRect::new(
                        POS_BALANCE.0,
                        POS_BALANCE.1,
                        BALANCE_BG_WIDTH,
                        BALANCE_BG_HEIGHT,
                        scale,
                    ),
                    move |fraction| {
                        balance_drag_change.set(Some(fraction));
                    },
                    move |dragging| {
                        balance_pressed.set(dragging);
                        if !dragging {
                            let Some(fraction) = balance_drag_commit.get_non_reactive() else {
                                return;
                            };
                            balance_drag_commit.set(None);
                            state.update(|s| s.balance = fraction);
                        }
                    },
                );
            }

            let shuffle_normal = if snapshot.shuffle {
                SHUFFLE_ON
            } else {
                SHUFFLE_OFF
            };
            let shuffle_pressed = if snapshot.shuffle {
                SHUFFLE_ON_ACTIVE
            } else {
                SHUFFLE_OFF_ACTIVE
            };
            {
                let state_click = state;
                PressableSprite(
                    skin.shufrep.clone(),
                    shuffle_normal,
                    shuffle_pressed,
                    POS_SHUFFLE.0,
                    POS_SHUFFLE.1,
                    scale,
                    move || {
                        state_click.update(|s| {
                            s.shuffle = !s.shuffle;
                            if s.shuffle {
                                refresh_shuffle_order(s);
                            } else {
                                s.shuffle_order.clear();
                            }
                            s.status = if s.shuffle {
                                "Shuffle On".to_string()
                            } else {
                                "Shuffle Off".to_string()
                            };
                        });
                    },
                );
            }

            let repeat_normal = if snapshot.repeat {
                REPEAT_ON
            } else {
                REPEAT_OFF
            };
            let repeat_pressed = if snapshot.repeat {
                REPEAT_ON_ACTIVE
            } else {
                REPEAT_OFF_ACTIVE
            };
            {
                let state_click = state;
                PressableSprite(
                    skin.shufrep.clone(),
                    repeat_normal,
                    repeat_pressed,
                    POS_REPEAT.0,
                    POS_REPEAT.1,
                    scale,
                    move || {
                        state_click.update(|s| {
                            s.repeat = !s.repeat;
                            s.status = if s.repeat {
                                "Repeat On".to_string()
                            } else {
                                "Repeat Off".to_string()
                            };
                        });
                    },
                );
            }

            let eq_normal = if snapshot.eq_visible {
                EQ_BUTTON_ON
            } else {
                EQ_BUTTON_OFF
            };
            let eq_pressed = if snapshot.eq_visible {
                EQ_BUTTON_ON_ACTIVE
            } else {
                EQ_BUTTON_OFF_ACTIVE
            };
            {
                let state_click = state;
                PressableSprite(
                    skin.shufrep.clone(),
                    eq_normal,
                    eq_pressed,
                    POS_EQ_BUTTON.0,
                    POS_EQ_BUTTON.1,
                    scale,
                    move || {
                        state_click.update(|s| {
                            s.eq_visible = !s.eq_visible;
                            s.status = if s.eq_visible {
                                "Equalizer Shown".to_string()
                            } else {
                                "Equalizer Hidden".to_string()
                            };
                        });
                    },
                );
            }

            let pl_normal = if snapshot.playlist_visible {
                PL_BUTTON_ON
            } else {
                PL_BUTTON_OFF
            };
            let pl_pressed = if snapshot.playlist_visible {
                PL_BUTTON_ON_ACTIVE
            } else {
                PL_BUTTON_OFF_ACTIVE
            };
            {
                let state_click = state;
                PressableSprite(
                    skin.shufrep.clone(),
                    pl_normal,
                    pl_pressed,
                    POS_PL_BUTTON.0,
                    POS_PL_BUTTON.1,
                    scale,
                    move || {
                        state_click.update(|s| {
                            s.playlist_visible = !s.playlist_visible;
                            s.status = if s.playlist_visible {
                                "Playlist Shown".to_string()
                            } else {
                                "Playlist Hidden".to_string()
                            };
                        });
                    },
                );
            }

            {
                let state_click = state;
                let skin_click = skin_state;
                ClickTarget(
                    MAIN_SKIN_CHOOSER_HIT_AREA.0,
                    MAIN_SKIN_CHOOSER_HIT_AREA.1,
                    MAIN_SKIN_CHOOSER_HIT_AREA.2,
                    MAIN_SKIN_CHOOSER_HIT_AREA.3,
                    scale,
                    move || {
                        open_skin_file(state_click, skin_click);
                    },
                );
            }
        },
    );
}

#[composable]
fn EqualizerWindow(
    skin: WinampSkin,
    state: MutableState<WinampState>,
    drag_target: WinampDragTarget,
    scale: f32,
) {
    let snapshot = state.get();

    Box(
        winamp_window_modifier(EQ_WIDTH, EQ_HEIGHT, scale, drag_target),
        BoxSpec::default(),
        move || {
            Sprite(skin.eqmain.clone(), EQ_WINDOW, 0.0, 0.0, scale);
            Sprite(skin.eqmain.clone(), EQ_TITLE_BAR_SELECTED, 0.0, 0.0, scale);
            Sprite(
                skin.eqmain.clone(),
                EQ_GRAPH_BG,
                POS_EQ_GRAPH_BG.0,
                POS_EQ_GRAPH_BG.1,
                scale,
            );
            Sprite(
                skin.eqmain.clone(),
                EQ_PREAMP_LINE,
                POS_EQ_PREAMP_LINE.0,
                POS_EQ_PREAMP_LINE.1,
                scale,
            );
            if snapshot.eq_enabled {
                EqCurve(snapshot.eq_values, skin.display_text_color, scale);
            }

            WindowDragHandle(drag_target, EQ_TITLE_DRAG_HIT_AREA, scale);

            {
                let state_click = state;
                PressableSprite(
                    skin.eqmain.clone(),
                    EQ_CLOSE_BUTTON,
                    EQ_CLOSE_BUTTON_SELECTED,
                    POS_EQ_CLOSE_BUTTON.0,
                    POS_EQ_CLOSE_BUTTON.1,
                    scale,
                    move || {
                        state_click.update(|s| {
                            s.eq_visible = false;
                            s.status = "Equalizer Hidden".to_string();
                        });
                    },
                );
            }

            let eq_on_normal = if snapshot.eq_enabled {
                EQ_ON_BUTTON_ON
            } else {
                EQ_ON_BUTTON_OFF
            };
            let eq_on_pressed = if snapshot.eq_enabled {
                EQ_ON_BUTTON_ON_SELECTED
            } else {
                EQ_ON_BUTTON_OFF_SELECTED
            };
            {
                let state_click = state;
                PressableSprite(
                    skin.eqmain.clone(),
                    eq_on_normal,
                    eq_on_pressed,
                    POS_EQ_ON_BUTTON.0,
                    POS_EQ_ON_BUTTON.1,
                    scale,
                    move || {
                        state_click.update(|s| {
                            s.eq_enabled = !s.eq_enabled;
                            s.status = if s.eq_enabled {
                                "EQ On".to_string()
                            } else {
                                "EQ Off".to_string()
                            };
                        });
                    },
                );
            }

            let eq_auto_normal = if snapshot.eq_auto {
                EQ_AUTO_BUTTON_ON
            } else {
                EQ_AUTO_BUTTON_OFF
            };
            let eq_auto_pressed = if snapshot.eq_auto {
                EQ_AUTO_BUTTON_ON_SELECTED
            } else {
                EQ_AUTO_BUTTON_OFF_SELECTED
            };
            {
                let state_click = state;
                PressableSprite(
                    skin.eqmain.clone(),
                    eq_auto_normal,
                    eq_auto_pressed,
                    POS_EQ_AUTO_BUTTON.0,
                    POS_EQ_AUTO_BUTTON.1,
                    scale,
                    move || {
                        state_click.update(|s| {
                            s.eq_auto = !s.eq_auto;
                            s.status = if s.eq_auto {
                                "EQ Auto On".to_string()
                            } else {
                                "EQ Auto Off".to_string()
                            };
                        });
                    },
                );
            }

            {
                let state_click = state;
                PressableSprite(
                    skin.eqmain.clone(),
                    EQ_PRESETS_BUTTON,
                    EQ_PRESETS_BUTTON_SELECTED,
                    POS_EQ_PRESETS_BUTTON.0,
                    POS_EQ_PRESETS_BUTTON.1,
                    scale,
                    move || {
                        state_click.update(|s| {
                            s.eq_values = DEFAULT_EQ_VALUES;
                            s.status = "EQ Reset".to_string();
                        });
                    },
                );
            }

            for (index, slider_x) in EQ_SLIDER_XS.iter().copied().enumerate() {
                let thumb_x = EQ_THUMB_XS[index];
                let eq_pressed = cranpose_core::useState(|| false);
                let eq_drag = cranpose_core::useState(|| None::<f32>);
                let value = eq_drag.get().unwrap_or(snapshot.eq_values[index]);
                let thumb_y = EQ_SLIDER_BG_Y
                    + vertical_slider_thumb_y(value, EQ_SLIDER_TRACK_HEIGHT, EQ_SLIDER_THUMB.3);
                let eq_thumb_sprite = if eq_pressed.get() {
                    EQ_SLIDER_THUMB_SELECTED
                } else {
                    EQ_SLIDER_THUMB
                };

                Sprite(
                    skin.eqmain.clone(),
                    eq_slider_bg_rect(value),
                    slider_x,
                    EQ_SLIDER_BG_Y,
                    scale,
                );
                Sprite(
                    skin.eqmain.clone(),
                    eq_thumb_sprite,
                    thumb_x,
                    thumb_y + EQ_SLIDER_THUMB_Y_OFFSET,
                    scale,
                );

                let eq_drag_change = eq_drag;
                let eq_drag_commit = eq_drag;
                VerticalDragSlider(
                    ControlRect::new(
                        slider_x,
                        EQ_SLIDER_BG_Y,
                        EQ_SLIDER_BG.2,
                        EQ_SLIDER_TRACK_HEIGHT,
                        scale,
                    ),
                    true,
                    move |fraction| {
                        eq_drag_change.set(Some(fraction));
                    },
                    move |dragging| {
                        eq_pressed.set(dragging);
                        if !dragging {
                            let Some(fraction) = eq_drag_commit.get_non_reactive() else {
                                return;
                            };
                            eq_drag_commit.set(None);
                            state.update(|s| {
                                s.eq_values[index] = fraction;
                            });
                        }
                    },
                );
            }
        },
    );
}

#[composable]
fn PlaylistWindow(
    pledit: ImageBitmap,
    palette: SkinPalette,
    display_text_color: [u8; 4],
    state: MutableState<WinampState>,
    drag_target: WinampDragTarget,
    window_size: WinampWindowSize,
    scale: f32,
) {
    let snapshot = state.get();
    let footer_menu = cranpose_core::useState(|| None::<PlaylistFooterMenu>);
    let playlist_scroll_state = cranpose_core::useState(|| snapshot.playlist_scroll);
    let playlist_entries_scroll_state = cranpose_core::useState(|| snapshot.playlist_scroll);
    let playlist_scroll_dragging = cranpose_core::useState(|| false);
    {
        let local_scroll = playlist_scroll_state;
        let local_entries_scroll = playlist_entries_scroll_state;
        let local_dragging = playlist_scroll_dragging;
        let saved_scroll = snapshot.playlist_scroll;
        cranpose_core::SideEffect(move || {
            if !local_dragging.get_non_reactive() {
                if (local_scroll.get_non_reactive() - saved_scroll).abs() > f32::EPSILON {
                    local_scroll.set(saved_scroll);
                }
                if (local_entries_scroll.get_non_reactive() - saved_scroll).abs() > f32::EPSILON {
                    local_entries_scroll.set(saved_scroll);
                }
            }
        });
    }
    let search_field =
        cranpose_core::remember(|| TextFieldState::new("")).with(|field| field.clone());
    let window_size = window_size.get();
    let skin_scale = scale.max(f32::EPSILON);
    let width = (window_size.width / skin_scale).max(PLAYLIST_WIDTH);
    let height = (window_size.height / skin_scale).max(playlist_min_height());
    let right_x = width - PLAYLIST_RIGHT_TILE.2;
    let bottom_y = height - PLAYLIST_BOTTOM_LEFT_CORNER.3;
    let list_width = (right_x - PLAYLIST_LIST_BG.0).max(1.0);
    let list_height = (bottom_y - PLAYLIST_LIST_BG.1).max(1.0);
    let title_min_x = PLAYLIST_TOP_LEFT_CORNER.2;
    let title_max_x = (width - PLAYLIST_TOP_RIGHT_CORNER.2 - PLAYLIST_TITLE_BAR.2).max(title_min_x);
    let title_x = ((width - PLAYLIST_TITLE_BAR.2) * 0.5).clamp(title_min_x, title_max_x);
    let scroll_track_x = width - 15.0;

    Box(
        winamp_window_modifier(width, height, scale, drag_target),
        BoxSpec::default(),
        move || {
            Box(
                Modifier::empty()
                    .size_points(scaled(list_width, scale), scaled(list_height, scale))
                    .absolute_offset(
                        scaled(PLAYLIST_LIST_BG.0, scale),
                        scaled(PLAYLIST_LIST_BG.1, scale),
                    )
                    .background(Color(0.0, 0.0, 0.0, 1.0)),
                BoxSpec::default(),
                || {},
            );

            Sprite(pledit.clone(), PLAYLIST_TOP_LEFT_CORNER, 0.0, 0.0, scale);
            StretchSprite(
                pledit.clone(),
                PLAYLIST_TOP_TILE,
                PLAYLIST_TOP_LEFT_CORNER.2,
                0.0,
                width - PLAYLIST_TOP_LEFT_CORNER.2 - PLAYLIST_TOP_RIGHT_CORNER.2,
                PLAYLIST_TOP_TILE.3,
                scale,
            );
            Sprite(pledit.clone(), PLAYLIST_TITLE_BAR, title_x, 0.0, scale);
            Sprite(
                pledit.clone(),
                PLAYLIST_TOP_RIGHT_CORNER,
                width - PLAYLIST_TOP_RIGHT_CORNER.2,
                0.0,
                scale,
            );

            StretchSprite(
                pledit.clone(),
                PLAYLIST_LEFT_TILE,
                0.0,
                PLAYLIST_TOP_LEFT_CORNER.3,
                PLAYLIST_LEFT_TILE.2,
                bottom_y - PLAYLIST_TOP_LEFT_CORNER.3,
                scale,
            );
            StretchSprite(
                pledit.clone(),
                PLAYLIST_RIGHT_TILE,
                right_x,
                PLAYLIST_TOP_RIGHT_CORNER.3,
                PLAYLIST_RIGHT_TILE.2,
                bottom_y - PLAYLIST_TOP_RIGHT_CORNER.3,
                scale,
            );

            StretchSprite(
                pledit.clone(),
                PLAYLIST_BOTTOM_LEFT_CORNER,
                0.0,
                bottom_y,
                width - PLAYLIST_BOTTOM_RIGHT_CORNER.2,
                PLAYLIST_BOTTOM_LEFT_CORNER.3,
                scale,
            );
            Sprite(
                pledit.clone(),
                PLAYLIST_BOTTOM_RIGHT_CORNER,
                width - PLAYLIST_BOTTOM_RIGHT_CORNER.2,
                bottom_y,
                scale,
            );
            PlaylistScrollbar(
                pledit.clone(),
                scroll_track_x,
                list_height,
                state,
                PlaylistScrollHandles {
                    thumb: playlist_scroll_state,
                    entries: playlist_entries_scroll_state,
                    dragging: playlist_scroll_dragging,
                },
                scale,
            );

            PlaylistEntries(
                palette,
                state,
                snapshot.clone(),
                playlist_entries_scroll_state,
                list_width,
                list_height,
                scale,
            );
            PlaylistFooterReadouts(snapshot.clone(), bottom_y, scale, display_text_color);
            PlaylistFooterControls(state, footer_menu, bottom_y, scale);
            if snapshot.playlist_search_visible {
                PlaylistSearchOverlay(
                    palette,
                    state,
                    snapshot.clone(),
                    search_field.clone(),
                    list_width,
                    scale,
                );
            }

            if let Some(menu) = footer_menu.get() {
                PlaylistMenu(
                    palette,
                    state,
                    footer_menu,
                    menu,
                    PlaylistMenuLayout {
                        window_width: width,
                        bottom_y,
                        scale,
                    },
                );
            }

            PlaylistWheelScrollTarget(
                state,
                PLAYLIST_LIST_BG.0,
                PLAYLIST_LIST_BG.1,
                list_width,
                list_height,
                scale,
            );
            WindowDragHandle(drag_target, (0.0, 0.0, width, PLAYLIST_DRAG_AREA.3), scale);
            WindowResizeHandle(
                drag_target,
                WindowResizeDirection::SouthEast,
                width - 16.0,
                height - 16.0,
                16.0,
                16.0,
                scale,
            );
        },
    );
}

#[composable]
fn PlaylistScrollbar(
    pledit: ImageBitmap,
    scroll_track_x: f32,
    list_height: f32,
    state: MutableState<WinampState>,
    playlist_scroll: PlaylistScrollHandles,
    scale: f32,
) {
    let scroll = playlist_scroll.thumb.get();
    let scroll_y = PLAYLIST_LIST_BG.1
        + vertical_slider_thumb_y_down(scroll, list_height, PLAYLIST_SCROLL_HANDLE.3);
    let scroll_pressed = cranpose_core::useState(|| false);
    let scroll_handle_sprite = if scroll_pressed.get() {
        PLAYLIST_SCROLL_HANDLE_SELECTED
    } else {
        PLAYLIST_SCROLL_HANDLE
    };
    Sprite(
        pledit,
        scroll_handle_sprite,
        scroll_track_x,
        scroll_y,
        scale,
    );
    PlaylistScrollbarInput(
        scroll_track_x,
        list_height,
        state,
        playlist_scroll,
        scroll_pressed,
        scale,
    );
}

#[composable]
fn PlaylistScrollbarInput(
    scroll_track_x: f32,
    list_height: f32,
    state: MutableState<WinampState>,
    playlist_scroll: PlaylistScrollHandles,
    scroll_pressed: MutableState<bool>,
    scale: f32,
) {
    #[cfg(not(target_arch = "wasm32"))]
    let last_thumb_scroll_update = cranpose_core::remember(|| None::<Instant>);
    let initial_scroll = playlist_scroll.thumb.get_non_reactive();
    let latest_scroll = cranpose_core::remember(move || initial_scroll);
    {
        let scroll_change = playlist_scroll.thumb;
        let entries_scroll_change = playlist_scroll.entries;
        let scroll_commit = playlist_scroll.thumb;
        let entries_scroll_commit = playlist_scroll.entries;
        let drag_state = playlist_scroll.dragging;
        let latest_scroll_change = latest_scroll.clone();
        let latest_scroll_commit = latest_scroll;
        #[cfg(not(target_arch = "wasm32"))]
        let last_thumb_scroll_change = last_thumb_scroll_update.clone();
        #[cfg(not(target_arch = "wasm32"))]
        let last_thumb_scroll_drag = last_thumb_scroll_update;
        VerticalDragSlider(
            ControlRect::new(
                scroll_track_x - PLAYLIST_SCROLL_HIT_PAD_X,
                PLAYLIST_LIST_BG.1,
                PLAYLIST_SCROLL_TRACK.2 + PLAYLIST_SCROLL_HIT_PAD_X * 2.0,
                list_height,
                scale,
            ),
            false,
            move |fraction| {
                latest_scroll_change.update(|latest| *latest = fraction);
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let now = Instant::now();
                    let should_update_thumb = last_thumb_scroll_change.with(|last| {
                        last.as_ref().is_none_or(|last| {
                            now.duration_since(*last)
                                >= Duration::from_millis(PLAYLIST_THUMB_SCROLL_FRAME_MS)
                        })
                    });
                    if should_update_thumb {
                        last_thumb_scroll_change.update(|last| *last = Some(now));
                        scroll_change.set(fraction);
                        entries_scroll_change.set(fraction);
                    }
                }
                #[cfg(target_arch = "wasm32")]
                {
                    scroll_change.set(fraction);
                    entries_scroll_change.set(fraction);
                }
            },
            move |dragging| {
                scroll_pressed.set(dragging);
                drag_state.set(dragging);
                set_playlist_scroll_drag_active(dragging);
                if dragging {
                    #[cfg(not(target_arch = "wasm32"))]
                    last_thumb_scroll_drag.update(|last| *last = None);
                } else {
                    let fraction = latest_scroll_commit.with(|latest| *latest);
                    scroll_commit.set(fraction);
                    entries_scroll_commit.set(fraction);
                    state.update(|s| {
                        s.playlist_scroll = fraction;
                    });
                }
            },
        );
    }
}

#[composable]
fn PlaylistEntries(
    palette: SkinPalette,
    state: MutableState<WinampState>,
    snapshot: WinampState,
    playlist_scroll: MutableState<f32>,
    list_width: f32,
    list_height: f32,
    scale: f32,
) {
    let playlist_scroll = playlist_scroll.get();
    Box(
        Modifier::empty()
            .size_points(scaled(list_width, scale), scaled(list_height, scale))
            .absolute_offset(
                scaled(PLAYLIST_LIST_BG.0, scale),
                scaled(PLAYLIST_LIST_BG.1, scale),
            )
            .clip_to_bounds(),
        BoxSpec::default(),
        move || {
            let row_height = WINAMP_PLAYLIST_LINE_HEIGHT;
            let max_rows = playlist_visible_row_capacity(list_height);
            let x = WINAMP_PLAYLIST_TEXT_X;
            let y = WINAMP_PLAYLIST_TEXT_Y;

            cranpose_core::SideEffect(move || {
                if state.get_non_reactive().playlist_visible_rows != max_rows {
                    state.update(|s| s.playlist_visible_rows = max_rows);
                }
            });

            if snapshot.playlist.is_empty() {
                PlaylistWinampText(
                    snapshot.status.clone(),
                    x,
                    y,
                    (list_width - 8.0).max(1.0),
                    row_height,
                    scale,
                    palette.normal,
                );
                return;
            }

            let max_start = snapshot.playlist.len().saturating_sub(max_rows);
            let start = ((playlist_scroll * max_start as f32).round() as usize).min(max_start);
            let line_width = (list_width - 8.0).max(1.0);
            let visible_text_cache = cranpose_core::remember(PlaylistVisibleTextCache::default);
            let (visible_text, visible_line_count) = visible_text_cache.update(|cache| {
                let width_bits = line_width.to_bits();
                let stale_playlist = cache
                    .playlist
                    .as_ref()
                    .is_none_or(|playlist| !Rc::ptr_eq(playlist, &snapshot.playlist));
                if stale_playlist
                    || cache.width_bits != width_bits
                    || cache.start != start
                    || cache.rows != max_rows
                    || cache.current_index != snapshot.current_index
                {
                    let text = build_playlist_visible_text(
                        snapshot.playlist.as_slice(),
                        start,
                        max_rows,
                        snapshot.current_index,
                        line_width,
                    );
                    cache.playlist = Some(Rc::clone(&snapshot.playlist));
                    cache.width_bits = width_bits;
                    cache.start = start;
                    cache.rows = max_rows;
                    cache.current_index = snapshot.current_index;
                    cache.line_count =
                        visible_playlist_line_count(snapshot.playlist.len(), start, max_rows);
                    cache.text = Rc::new(AnnotatedString::from(text));
                }
                (Rc::clone(&cache.text), cache.line_count)
            });

            let selection_height = WINAMP_PLAYLIST_SELECTION_HEIGHT.min(row_height).max(1.0);
            for row in start..start + visible_line_count {
                if snapshot.selected_indices.contains(&row) {
                    let row_offset = (row - start) as f32 * row_height;
                    let selection_y = y
                        + row_offset
                        + (row_height - selection_height) * 0.5
                        + WINAMP_PLAYLIST_SELECTION_Y_OFFSET;
                    FilledRect(
                        2.0,
                        selection_y,
                        list_width - 4.0,
                        selection_height,
                        scale,
                        srgb_color(palette.selected_bg),
                    );
                }
            }

            PlaylistWinampTextBlock(
                visible_text,
                SystemTextBox {
                    x,
                    y,
                    width: line_width,
                    height: row_height * visible_line_count as f32,
                    scale,
                },
                palette.normal,
                visible_line_count,
            );
            for (row, track) in snapshot
                .playlist
                .iter()
                .enumerate()
                .skip(start)
                .take(max_rows)
            {
                let current = snapshot.current_index == Some(row);
                let row_offset = (row - start) as f32 * row_height;
                let row_y = y + row_offset;
                if current {
                    let duration = playlist_duration_text(track.duration_seconds);
                    let title = if snapshot.playback == PlaybackState::Playing {
                        marquee_system_text(
                            track.display_title().to_string(),
                            line_width,
                            snapshot.title_marquee_phase,
                        )
                    } else {
                        track.display_title().to_string()
                    };
                    let line = playlist_row_line_text(title, duration.as_deref(), line_width);
                    PlaylistWinampText(
                        line,
                        x,
                        row_y,
                        line_width,
                        row_height,
                        scale,
                        palette.current,
                    );
                }

                {
                    let state_click = state;
                    ClickTarget(0.0, row_offset, list_width, row_height, scale, move || {
                        handle_playlist_row_click(state_click, row);
                    });
                }
            }
        },
    );
}

fn playlist_min_height() -> f32 {
    #[cfg(target_os = "android")]
    {
        145.0
    }
    #[cfg(not(target_os = "android"))]
    {
        PLAYLIST_HEIGHT
    }
}

#[composable]
fn PlaylistFooterReadouts(
    snapshot: WinampState,
    bottom_y: f32,
    scale: f32,
    display_text_color: [u8; 4],
) {
    let summary = playlist_footer_summary(&snapshot);
    SystemWinampText(
        summary,
        132.0,
        bottom_y + 10.0,
        72.0,
        WINAMP_SYSTEM_LINE_HEIGHT,
        scale,
        display_text_color,
    );

    let elapsed = format_duration_compact(snapshot.elapsed_seconds);
    let elapsed_width = system_text_width(&elapsed);
    let elapsed_x = 221.0 - elapsed_width;
    SystemWinampText(
        elapsed,
        elapsed_x,
        bottom_y + 24.0,
        elapsed_width + 2.0,
        WINAMP_SYSTEM_LINE_HEIGHT,
        scale,
        display_text_color,
    );
}

#[composable]
fn PlaylistSearchOverlay(
    palette: SkinPalette,
    state: MutableState<WinampState>,
    snapshot: WinampState,
    search_field: TextFieldState,
    list_width: f32,
    scale: f32,
) {
    let x = PLAYLIST_LIST_BG.0 + 14.0;
    let y = PLAYLIST_LIST_BG.1 + 34.0;
    let width = (list_width - 28.0).clamp(150.0, 260.0);
    let height = 40.0;
    let query = search_field.text();
    let last_revision = cranpose_core::useState(|| 0u64);

    {
        let state_for_sync = state;
        let field_for_sync = search_field.clone();
        let snapshot_query = snapshot.playlist_search_query.clone();
        let snapshot_revision = snapshot.playlist_search_revision;
        let query_for_sync = query.clone();
        cranpose_core::SideEffect(move || {
            if last_revision.get() != snapshot_revision {
                set_text_field_text(&field_for_sync, &snapshot_query);
                last_revision.set(snapshot_revision);
                return;
            }
            if query_for_sync != state_for_sync.get_non_reactive().playlist_search_query {
                state_for_sync.update(|s| {
                    s.playlist_search_query = query_for_sync.clone();
                    apply_playlist_search_filter_in_state(s, &query_for_sync);
                    s.status = format!("Selected {} Match(es)", s.selected_indices.len());
                });
            }
        });
    }

    FilledRect(x, y, width, height, scale, Color(0.01, 0.015, 0.012, 1.0));
    FilledRect(x, y, width, 1.0, scale, Color(0.30, 0.42, 0.50, 1.0));
    FilledRect(
        x,
        y + height - 1.0,
        width,
        1.0,
        scale,
        Color(0.12, 0.20, 0.24, 1.0),
    );
    SystemWinampText(
        "SEARCH".to_string(),
        x + 6.0,
        y + 5.0,
        44.0,
        WINAMP_SYSTEM_LINE_HEIGHT,
        scale,
        palette.normal,
    );
    SystemWinampText(
        "CLOSE".to_string(),
        x + width - 33.0,
        y + 5.0,
        30.0,
        WINAMP_SYSTEM_LINE_HEIGHT,
        scale,
        palette.normal,
    );

    FilledRect(
        x + 6.0,
        y + 18.0,
        width - 12.0,
        14.0,
        scale,
        Color(0.0, 0.0, 0.0, 1.0),
    );
    BasicTextField(
        search_field,
        Modifier::empty()
            .size_points(scaled(width - 16.0, scale), scaled(12.0, scale))
            .absolute_offset(scaled(x + 8.0, scale), scaled(y + 18.0, scale)),
        TextStyle::from_span_style(SpanStyle {
            color: Some(Color(0.72, 0.86, 0.94, 1.0)),
            font_size: TextUnit::Sp(10.0),
            ..SpanStyle::default()
        }),
    );

    {
        let state_click = state;
        ClickTarget(x + width - 38.0, y, 38.0, 15.0, scale, move || {
            state_click.update(|s| {
                s.playlist_search_visible = false;
            });
        });
    }
}

#[composable]
fn PlaylistFooterControls(
    state: MutableState<WinampState>,
    footer_menu: MutableState<Option<PlaylistFooterMenu>>,
    bottom_y: f32,
    scale: f32,
) {
    {
        let menu_state = footer_menu;
        PlaylistFooterClickTarget(PLAYLIST_ADD_BUTTON_HIT_AREA, bottom_y, scale, move || {
            toggle_playlist_footer_menu(menu_state, PlaylistFooterMenu::Add);
        });
    }
    {
        let menu_state = footer_menu;
        PlaylistFooterClickTarget(PLAYLIST_REM_BUTTON_HIT_AREA, bottom_y, scale, move || {
            toggle_playlist_footer_menu(menu_state, PlaylistFooterMenu::Remove);
        });
    }
    {
        let menu_state = footer_menu;
        PlaylistFooterClickTarget(PLAYLIST_SEL_BUTTON_HIT_AREA, bottom_y, scale, move || {
            toggle_playlist_footer_menu(menu_state, PlaylistFooterMenu::Select);
        });
    }
    {
        let menu_state = footer_menu;
        PlaylistFooterClickTarget(PLAYLIST_MISC_BUTTON_HIT_AREA, bottom_y, scale, move || {
            toggle_playlist_footer_menu(menu_state, PlaylistFooterMenu::Misc);
        });
    }
    {
        let menu_state = footer_menu;
        PlaylistFooterClickTarget(PLAYLIST_LIST_BUTTON_HIT_AREA, bottom_y, scale, move || {
            toggle_playlist_footer_menu(menu_state, PlaylistFooterMenu::List);
        });
    }
    {
        let state_click = state;
        let menu_state = footer_menu;
        PlaylistFooterClickTarget(PLAYLIST_PREV_BUTTON_HIT_AREA, bottom_y, scale, move || {
            menu_state.set(None);
            previous_track(state_click);
        });
    }
    {
        let state_click = state;
        let menu_state = footer_menu;
        PlaylistFooterClickTarget(PLAYLIST_PLAY_BUTTON_HIT_AREA, bottom_y, scale, move || {
            menu_state.set(None);
            play_or_resume(state_click);
        });
    }
    {
        let state_click = state;
        let menu_state = footer_menu;
        PlaylistFooterClickTarget(PLAYLIST_PAUSE_BUTTON_HIT_AREA, bottom_y, scale, move || {
            menu_state.set(None);
            pause_playback(state_click);
        });
    }
    {
        let state_click = state;
        let menu_state = footer_menu;
        PlaylistFooterClickTarget(PLAYLIST_STOP_BUTTON_HIT_AREA, bottom_y, scale, move || {
            menu_state.set(None);
            stop_playback(state_click);
        });
    }
    {
        let state_click = state;
        let menu_state = footer_menu;
        PlaylistFooterClickTarget(PLAYLIST_NEXT_BUTTON_HIT_AREA, bottom_y, scale, move || {
            menu_state.set(None);
            next_track(state_click);
        });
    }
    {
        let state_click = state;
        let menu_state = footer_menu;
        PlaylistFooterClickTarget(PLAYLIST_EJECT_BUTTON_HIT_AREA, bottom_y, scale, move || {
            menu_state.set(None);
            open_audio_files(state_click);
        });
    }
}

fn toggle_playlist_footer_menu(
    menu_state: MutableState<Option<PlaylistFooterMenu>>,
    menu: PlaylistFooterMenu,
) {
    menu_state.update(|open| {
        *open = if *open == Some(menu) {
            None
        } else {
            Some(menu)
        };
    });
}

#[composable]
fn PlaylistFooterClickTarget(
    area: SpriteRect,
    bottom_y: f32,
    scale: f32,
    on_click: impl Fn() + 'static,
) {
    let is_pressed = cranpose_core::useState(|| false);
    let area = ControlRect::new(area.0, bottom_y + area.1, area.2, area.3, scale);
    if is_pressed.get() {
        FilledRect(
            area.x,
            area.y,
            area.width,
            area.height,
            area.scale,
            Color(0.0, 0.0, 0.0, 0.4),
        );
    }
    PressableClickArea(area, is_pressed, on_click);
}

#[composable]
fn PressableClickArea(
    area: ControlRect,
    is_pressed: MutableState<bool>,
    on_click: impl Fn() + 'static,
) {
    let on_click = Rc::new(on_click);
    let w = area.scaled_width();
    let h = area.scaled_height();

    Box(
        Modifier::empty()
            .size_points(w, h)
            .absolute_offset(area.scaled_x(), area.scaled_y())
            .pointer_input((), {
                move |scope: PointerInputScope| {
                    let on_click = on_click.clone();
                    async move {
                        scope
                            .await_pointer_event_scope(|await_scope| async move {
                                loop {
                                    let event = await_scope.await_pointer_event().await;
                                    match event.kind {
                                        PointerEventKind::Down => {
                                            is_pressed.set(true);
                                            event.consume();
                                        }
                                        PointerEventKind::Move
                                            if is_pressed.get()
                                                && !event
                                                    .buttons
                                                    .contains(PointerButton::Primary) =>
                                        {
                                            is_pressed.set(false);
                                        }
                                        PointerEventKind::Up => {
                                            let was_pressed = is_pressed.get();
                                            is_pressed.set(false);
                                            let inside = area.contains_scaled_point(event.position);
                                            if was_pressed && inside {
                                                on_click();
                                            }
                                            event.consume();
                                        }
                                        PointerEventKind::Cancel => {
                                            is_pressed.set(false);
                                        }
                                        _ => {}
                                    }
                                }
                            })
                            .await;
                    }
                }
            }),
        BoxSpec::default(),
        || {},
    );
}

#[composable]
fn PlaylistWheelScrollTarget(
    state: MutableState<WinampState>,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    scale: f32,
) {
    Box(
        Modifier::empty()
            .size_points(scaled(width, scale), scaled(height, scale))
            .absolute_offset(scaled(x, scale), scaled(y, scale))
            .pointer_input((), move |scope: PointerInputScope| async move {
                scope
                    .await_pointer_event_scope(|await_scope| async move {
                        loop {
                            let event = await_scope.await_pointer_event().await;
                            if event.kind == PointerEventKind::Scroll {
                                let rows = if event.scroll_delta.y < 0.0 { 3 } else { -3 };
                                scroll_playlist_by_rows(state, rows);
                                event.consume();
                            }
                        }
                    })
                    .await;
            }),
        BoxSpec::default(),
        || {},
    );
}

fn playlist_footer_summary(state: &WinampState) -> String {
    let current = state
        .current_index
        .and_then(|index| state.playlist.get(index))
        .and_then(|track| state.duration_seconds.or(track.duration_seconds))
        .map(format_duration_compact)
        .unwrap_or_else(|| "0:00".to_string());
    let total = playlist_total_duration_seconds(state.playlist.as_slice())
        .map(format_duration_compact)
        .unwrap_or_else(|| "0:00".to_string());

    format!("{current}/{total}")
}

fn playlist_total_duration_seconds(playlist: &[Track]) -> Option<f32> {
    let mut total = 0.0;
    let mut found = false;
    for track in playlist {
        if let Some(duration) = track.duration_seconds.filter(|duration| *duration > 0.0) {
            total += duration;
            found = true;
        }
    }
    found.then_some(total)
}

fn playlist_duration_text(duration_seconds: Option<f32>) -> Option<String> {
    duration_seconds
        .filter(|duration| *duration > 0.0)
        .map(format_duration_compact)
}

fn visible_playlist_line_count(total_rows: usize, start: usize, max_rows: usize) -> usize {
    total_rows.saturating_sub(start).min(max_rows).max(1)
}

fn playlist_visible_row_capacity(list_height: f32) -> usize {
    ((list_height - WINAMP_PLAYLIST_TEXT_Y) / WINAMP_PLAYLIST_LINE_HEIGHT)
        .floor()
        .max(1.0) as usize
}

fn build_playlist_visible_text(
    playlist: &[Track],
    start: usize,
    max_rows: usize,
    current_index: Option<usize>,
    width: f32,
) -> String {
    let mut text = String::new();
    for (line, (row, track)) in playlist
        .iter()
        .enumerate()
        .skip(start)
        .take(max_rows)
        .enumerate()
    {
        if line > 0 {
            text.push('\n');
        }
        if current_index == Some(row) {
            continue;
        }
        let duration = playlist_duration_text(track.duration_seconds);
        text.push_str(&playlist_row_line_text(
            track.display_title().to_string(),
            duration.as_deref(),
            width,
        ));
    }
    text
}

fn playlist_row_line_text(title: String, duration: Option<&str>, width: f32) -> String {
    let max_chars = (width / WINAMP_PLAYLIST_ROW_CHAR_WIDTH).floor().max(1.0) as usize;
    let Some(duration) = duration else {
        return truncate_chars(&title, max_chars);
    };
    let duration_chars = duration.chars().count();
    if max_chars <= duration_chars + 1 {
        return truncate_chars(&title, max_chars);
    }

    let title_chars = max_chars - duration_chars - 1;
    let title = truncate_chars(&title, title_chars);
    let title_len = title.chars().count();
    let padding = title_chars.saturating_sub(title_len) + 1;
    format!("{title}{}{duration}", " ".repeat(padding))
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        text.chars().take(max_chars).collect()
    }
}

fn format_duration_compact(seconds: f32) -> String {
    let seconds = seconds.max(0.0).round() as u32;
    let hours = seconds / 3600;
    let minutes = (seconds / 60) % 60;
    let seconds = seconds % 60;
    if hours > 0 {
        format!("{hours}:{minutes:02}:{seconds:02}")
    } else {
        format!("{minutes}:{seconds:02}")
    }
}

fn system_text_width(text: &str) -> f32 {
    text.chars().count() as f32 * WINAMP_SYSTEM_MEASURE_CHAR_WIDTH
}

#[derive(Clone, Copy, PartialEq)]
struct PlaylistMenuLayout {
    window_width: f32,
    bottom_y: f32,
    scale: f32,
}

#[composable]
fn PlaylistMenu(
    palette: SkinPalette,
    state: MutableState<WinampState>,
    menu_open: MutableState<Option<PlaylistFooterMenu>>,
    menu: PlaylistFooterMenu,
    layout: PlaylistMenuLayout,
) {
    let items = playlist_footer_menu_items(menu);
    let width = playlist_footer_menu_width(menu);
    let row_height = 14.0;
    let height = row_height * items.len() as f32;
    let x = playlist_footer_menu_x(menu, layout.window_width, width);
    let y = (layout.bottom_y - height - 3.0).max(PLAYLIST_TOP_LEFT_CORNER.3);
    let scale = layout.scale;

    FilledRect(x, y, width, height, scale, Color(0.01, 0.015, 0.012, 1.0));
    FilledRect(x, y, width, 1.0, scale, Color(0.30, 0.42, 0.50, 1.0));

    for (index, item) in items.iter().copied().enumerate() {
        let row_y = y + row_height * index as f32;
        if index > 0 {
            FilledRect(x, row_y, width, 1.0, scale, Color(0.12, 0.20, 0.24, 1.0));
        }

        SystemWinampText(
            item.label.to_string(),
            x + 5.0,
            row_y + 4.0,
            width - 10.0,
            row_height - 4.0,
            scale,
            palette.normal,
        );

        let state_click = state;
        let menu_state = menu_open;
        ClickTarget(x, row_y, width, row_height, scale, move || {
            menu_state.set(None);
            (item.action)(state_click);
        });
    }
}

fn playlist_footer_menu_items(menu: PlaylistFooterMenu) -> Vec<PlaylistMenuItem> {
    match menu {
        PlaylistFooterMenu::Add => vec![
            PlaylistMenuItem {
                label: "ADD FILE",
                action: add_audio_files,
            },
            PlaylistMenuItem {
                label: "ADD FOLDER",
                action: add_audio_folder,
            },
        ],
        PlaylistFooterMenu::Remove => vec![
            PlaylistMenuItem {
                label: "REM ALL",
                action: remove_all_tracks,
            },
            PlaylistMenuItem {
                label: "DUPLICATES",
                action: remove_duplicate_tracks,
            },
            PlaylistMenuItem {
                label: "SELECTED",
                action: remove_selected_tracks,
            },
            PlaylistMenuItem {
                label: "UNSELECTED",
                action: remove_unselected_tracks,
            },
        ],
        PlaylistFooterMenu::Select => vec![
            PlaylistMenuItem {
                label: "NONE",
                action: select_no_tracks,
            },
            PlaylistMenuItem {
                label: "ALL",
                action: select_all_tracks,
            },
            PlaylistMenuItem {
                label: "SEARCH",
                action: select_search_matches,
            },
            PlaylistMenuItem {
                label: "INVERT",
                action: invert_track_selection,
            },
        ],
        PlaylistFooterMenu::Misc => vec![
            PlaylistMenuItem {
                label: "SORT TITLE",
                action: sort_playlist_by_title,
            },
            PlaylistMenuItem {
                label: "SORT ARTIST",
                action: sort_playlist_by_artist,
            },
            PlaylistMenuItem {
                label: "SORT FILE",
                action: sort_playlist_by_file_name,
            },
            PlaylistMenuItem {
                label: "SORT PATH",
                action: sort_playlist_by_path,
            },
            PlaylistMenuItem {
                label: "SORT EXT",
                action: sort_playlist_by_extension,
            },
            PlaylistMenuItem {
                label: "SORT GENRE",
                action: sort_playlist_by_genre,
            },
            PlaylistMenuItem {
                label: "SORT TIME",
                action: sort_playlist_by_duration,
            },
            PlaylistMenuItem {
                label: "SORT TAG",
                action: sort_playlist_by_tag,
            },
            PlaylistMenuItem {
                label: "RANDOMIZE",
                action: randomize_playlist,
            },
        ],
        PlaylistFooterMenu::List => vec![
            PlaylistMenuItem {
                label: "NEW LIST",
                action: new_playlist,
            },
            PlaylistMenuItem {
                label: "IMPORT M3U",
                action: import_playlist,
            },
            PlaylistMenuItem {
                label: "EXPORT M3U",
                action: export_playlist,
            },
        ],
    }
}

fn playlist_footer_menu_width(menu: PlaylistFooterMenu) -> f32 {
    match menu {
        PlaylistFooterMenu::Misc => 78.0,
        PlaylistFooterMenu::Remove => 78.0,
        PlaylistFooterMenu::List => 76.0,
        PlaylistFooterMenu::Select | PlaylistFooterMenu::Add => 72.0,
    }
}

fn playlist_footer_menu_x(menu: PlaylistFooterMenu, window_width: f32, menu_width: f32) -> f32 {
    let button_x = match menu {
        PlaylistFooterMenu::Add => PLAYLIST_ADD_BUTTON_HIT_AREA.0,
        PlaylistFooterMenu::Remove => PLAYLIST_REM_BUTTON_HIT_AREA.0,
        PlaylistFooterMenu::Select => PLAYLIST_SEL_BUTTON_HIT_AREA.0,
        PlaylistFooterMenu::Misc => PLAYLIST_MISC_BUTTON_HIT_AREA.0,
        PlaylistFooterMenu::List => PLAYLIST_LIST_BUTTON_HIT_AREA.0,
    };
    button_x.clamp(4.0, (window_width - menu_width - 4.0).max(4.0))
}

#[composable]
fn FilledRect(x: f32, y: f32, width: f32, height: f32, scale: f32, color: Color) {
    let width = scaled(width, scale);
    let height = scaled(height, scale);
    Canvas(
        Modifier::empty()
            .size_points(width, height)
            .absolute_offset(scaled(x, scale), scaled(y, scale)),
        move |scope| {
            scope.draw_rect(Brush::solid(color));
        },
    );
}

#[composable]
fn SystemWinampText(
    text: String,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    scale: f32,
    color: [u8; 4],
) {
    StyledSystemWinampText(
        text,
        SystemTextBox {
            x,
            y,
            width,
            height,
            scale,
        },
        color,
        WINAMP_SYSTEM_TEXT_METRICS,
    );
}

#[composable]
fn PlaylistWinampText(
    text: String,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    scale: f32,
    color: [u8; 4],
) {
    StyledSystemWinampText(
        text,
        SystemTextBox {
            x,
            y,
            width,
            height,
            scale,
        },
        color,
        WINAMP_PLAYLIST_TEXT_METRICS,
    );
}

#[composable]
fn StyledSystemWinampText(
    text: String,
    layout: SystemTextBox,
    color: [u8; 4],
    metrics: SystemTextMetrics,
) {
    let text_y = ((layout.height - metrics.line_height).max(0.0) * 0.5) + metrics.y_adjust;
    Box(
        Modifier::empty()
            .size_points(
                scaled(layout.width, layout.scale),
                scaled(layout.height, layout.scale),
            )
            .absolute_offset(
                scaled(layout.x, layout.scale),
                scaled(layout.y, layout.scale),
            )
            .clip_to_bounds(),
        BoxSpec::default(),
        move || {
            BasicText(
                text.clone(),
                Modifier::empty()
                    .size_points(
                        scaled(layout.width, layout.scale),
                        scaled(metrics.line_height, layout.scale),
                    )
                    .absolute_offset(0.0, scaled(text_y, layout.scale)),
                TextStyle::new(
                    SpanStyle {
                        color: Some(Color::from_rgba_u8(color[0], color[1], color[2], color[3])),
                        font_size: TextUnit::Sp(metrics.font_size * layout.scale),
                        font_family: Some(FontFamily::Monospace),
                        letter_spacing: TextUnit::Sp(0.0),
                        ..SpanStyle::default()
                    },
                    ParagraphStyle {
                        line_height: TextUnit::Sp(metrics.line_height * layout.scale),
                        ..ParagraphStyle::default()
                    },
                ),
                TextOverflow::Clip,
                false,
                1,
                1,
            );
        },
    );
}

#[composable]
fn SystemWinampTextBlock(
    text: Rc<AnnotatedString>,
    layout: SystemTextBox,
    color: [u8; 4],
    max_lines: usize,
) {
    StyledSystemWinampTextBlock(text, layout, color, max_lines, WINAMP_SYSTEM_TEXT_METRICS);
}

#[composable]
fn PlaylistWinampTextBlock(
    text: Rc<AnnotatedString>,
    layout: SystemTextBox,
    color: [u8; 4],
    max_lines: usize,
) {
    StyledSystemWinampTextBlock(text, layout, color, max_lines, WINAMP_PLAYLIST_TEXT_METRICS);
}

#[composable]
fn StyledSystemWinampTextBlock(
    text: Rc<AnnotatedString>,
    layout: SystemTextBox,
    color: [u8; 4],
    max_lines: usize,
    metrics: SystemTextMetrics,
) {
    let text_height = metrics.line_height * max_lines.max(1) as f32;
    let text_y = ((layout.height - text_height).max(0.0) * 0.5) + metrics.y_adjust;
    Box(
        Modifier::empty()
            .size_points(
                scaled(layout.width, layout.scale),
                scaled(layout.height, layout.scale),
            )
            .absolute_offset(
                scaled(layout.x, layout.scale),
                scaled(layout.y, layout.scale),
            )
            .clip_to_bounds(),
        BoxSpec::default(),
        move || {
            BasicText(
                text.clone(),
                Modifier::empty()
                    .size_points(
                        scaled(layout.width, layout.scale),
                        scaled(text_height, layout.scale),
                    )
                    .absolute_offset(0.0, scaled(text_y, layout.scale)),
                TextStyle::new(
                    SpanStyle {
                        color: Some(Color::from_rgba_u8(color[0], color[1], color[2], color[3])),
                        font_size: TextUnit::Sp(metrics.font_size * layout.scale),
                        font_family: Some(FontFamily::Monospace),
                        letter_spacing: TextUnit::Sp(0.0),
                        ..SpanStyle::default()
                    },
                    ParagraphStyle {
                        line_height: TextUnit::Sp(metrics.line_height * layout.scale),
                        ..ParagraphStyle::default()
                    },
                ),
                TextOverflow::Clip,
                false,
                max_lines,
                1,
            );
        },
    );
}

#[derive(Clone, Copy, PartialEq)]
struct SystemTextBox {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    scale: f32,
}

fn marquee_system_text(text: String, width: f32, phase: f32) -> String {
    let max_chars = (width / WINAMP_SYSTEM_MARQUEE_CHAR_WIDTH).ceil().max(1.0) as usize;
    let char_count = text.chars().count();
    if char_count <= max_chars {
        return text;
    }

    let max_offset = char_count - max_chars;
    let offset = ping_pong_offset(phase, max_offset);
    text.chars().skip(offset).collect()
}

fn ping_pong_offset(position: f32, max_offset: usize) -> usize {
    if max_offset == 0 {
        return 0;
    }

    let span = max_offset as f32;
    let cycle = span * 2.0;
    let position = position.max(0.0) % cycle;
    if position <= span {
        position.floor() as usize
    } else {
        (cycle - position).floor() as usize
    }
}

fn main_display_title(state: &WinampState) -> String {
    state
        .current_index
        .and_then(|index| state.playlist.get(index))
        .map(|track| track.display_title().to_string())
        .unwrap_or_else(|| state.status.clone())
}

#[composable]
fn MainMetaReadouts(state: WinampState, scale: f32, display_text_color: [u8; 4]) {
    if state
        .current_index
        .and_then(|index| state.playlist.get(index))
        .is_some()
    {
        SystemWinampText(
            "320".to_string(),
            POS_MAIN_META_TEXT.0,
            POS_MAIN_META_TEXT.1,
            20.0,
            WINAMP_SYSTEM_LINE_HEIGHT,
            scale,
            display_text_color,
        );
        SystemWinampText(
            "44".to_string(),
            POS_MAIN_META_TEXT.0 + 45.0,
            POS_MAIN_META_TEXT.1,
            16.0,
            WINAMP_SYSTEM_LINE_HEIGHT,
            scale,
            display_text_color,
        );
        return;
    }

    SystemWinampText(
        main_display_meta(&state),
        POS_MAIN_META_TEXT.0,
        POS_MAIN_META_TEXT.1,
        MAIN_META_TEXT_WIDTH,
        WINAMP_SYSTEM_LINE_HEIGHT,
        scale,
        display_text_color,
    );
}

fn main_display_meta(state: &WinampState) -> String {
    let prefix = match state.playback {
        PlaybackState::Playing => "PLAY",
        PlaybackState::Paused => "PAUSE",
        PlaybackState::Stopped => "STOP",
    };
    let count = state.playlist.len();
    if count == 0 {
        return prefix.to_string();
    }

    let index = state.current_index.map(|index| index + 1).unwrap_or(1);
    format!("{prefix} {index:02}/{count:02}")
}

#[composable]
fn Visualizer(playing: bool, bands: audio::VisualizerBands, viscolor: VisColor, scale: f32) {
    let bitmap = visualizer_bitmap(playing, bands, viscolor);
    let width = scaled(VISUALIZER_WIDTH, scale);
    let height = scaled(VISUALIZER_HEIGHT, scale);
    Canvas(
        Modifier::empty()
            .size_points(width, height)
            .absolute_offset(
                scaled(POS_VISUALIZER.0, scale),
                scaled(POS_VISUALIZER.1, scale),
            ),
        move |scope| {
            scope.draw_image_at(
                Rect {
                    x: 0.0,
                    y: 0.0,
                    width,
                    height,
                },
                bitmap.clone(),
                1.0,
                None,
            );
        },
    );
}

#[composable]
fn EqCurve(values: [f32; 11], display_text_color: [u8; 4], scale: f32) {
    let bitmap = eq_curve_bitmap(values, display_text_color);
    let width = scaled(EQ_GRAPH_BG.2, scale);
    let height = scaled(EQ_GRAPH_BG.3, scale);

    Canvas(
        Modifier::empty()
            .size_points(width, height)
            .absolute_offset(
                scaled(POS_EQ_GRAPH_BG.0, scale),
                scaled(POS_EQ_GRAPH_BG.1, scale),
            ),
        move |scope| {
            scope.draw_image_at(
                Rect {
                    x: 0.0,
                    y: 0.0,
                    width,
                    height,
                },
                bitmap.clone(),
                1.0,
                None,
            );
        },
    );
}

fn eq_curve_bitmap(values: [f32; 11], display_text_color: [u8; 4]) -> ImageBitmap {
    let width = EQ_GRAPH_BG.2 as u32;
    let height = EQ_GRAPH_BG.3 as u32;
    let mut pixels = vec![0u8; width as usize * height as usize * 4];
    let band_values = &values[1..];
    let mut points = Vec::with_capacity(band_values.len());

    for (index, value) in band_values.iter().copied().enumerate() {
        let x = if band_values.len() <= 1 {
            0
        } else {
            ((index as f32 / (band_values.len() - 1) as f32) * (width - 1) as f32).round() as i32
        };
        let y = ((1.0 - clamp01(value)) * (height - 1) as f32).round() as i32;
        points.push((x, y));
    }

    for pair in points.windows(2) {
        let from = pair[0];
        let to = pair[1];
        draw_bitmap_line(&mut pixels, width, height, from, to, [38, 91, 132, 255]);
        draw_bitmap_line(
            &mut pixels,
            width,
            height,
            (from.0, from.1 - 1),
            (to.0, to.1 - 1),
            display_text_color,
        );
    }

    ImageBitmap::from_rgba8(width, height, pixels)
        .expect("rendered EQ curve bitmap should be valid")
}

fn draw_bitmap_line(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    from: (i32, i32),
    to: (i32, i32),
    color: [u8; 4],
) {
    let (mut x0, mut y0) = from;
    let (x1, y1) = to;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        set_bitmap_pixel(pixels, width, height, x0, y0, color);
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = err * 2;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn set_bitmap_pixel(pixels: &mut [u8], width: u32, height: u32, x: i32, y: i32, color: [u8; 4]) {
    if x < 0 || y < 0 {
        return;
    }
    let x = x as u32;
    let y = y as u32;
    if x >= width || y >= height {
        return;
    }

    let offset = ((y * width + x) * 4) as usize;
    pixels[offset..offset + 4].copy_from_slice(&color);
}

fn visualizer_bitmap(
    playing: bool,
    bands: audio::VisualizerBands,
    viscolor: VisColor,
) -> ImageBitmap {
    let width = VISUALIZER_WIDTH as u32;
    let height = VISUALIZER_HEIGHT as u32;
    let bg = viscolor.background();
    let mut pixels = vec![0u8; width as usize * height as usize * 4];
    for pixel in pixels.chunks_exact_mut(4) {
        pixel.copy_from_slice(&bg);
    }

    if !playing {
        return ImageBitmap::from_rgba8(width, height, pixels)
            .expect("rendered visualizer bitmap should be valid");
    }

    let max_segments = 5;
    let bar_width = 3;
    let bar_pitch = 4;
    let segment_height = 2;
    let segment_pitch = 3;
    for bar in 0..VISUALIZER_BARS {
        let value = visualizer_band_height(bands, bar);
        let x = (bar * bar_pitch) as u32;
        for segment in 0..max_segments {
            let threshold = ((segment + 1) as f32 / max_segments as f32) * VISUALIZER_HEIGHT;
            let color =
                visualizer_segment_rgba(segment, max_segments, value >= threshold, &viscolor);
            let y = height as i32 - ((segment + 1) * segment_pitch) as i32 + 1;
            fill_visualizer_rect(
                &mut pixels,
                width,
                height,
                (x, y.max(0) as u32, bar_width, segment_height),
                color,
            );
        }
    }

    ImageBitmap::from_rgba8(width, height, pixels)
        .expect("rendered visualizer bitmap should be valid")
}

fn fill_visualizer_rect(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    rect: (u32, u32, u32, u32),
    color: [u8; 4],
) {
    let (x, y, rect_width, rect_height) = rect;
    for yy in y..(y + rect_height).min(height) {
        for xx in x..(x + rect_width).min(width) {
            let offset = ((yy * width + xx) * 4) as usize;
            pixels[offset..offset + 4].copy_from_slice(&color);
        }
    }
}

fn visualizer_band_height(bands: audio::VisualizerBands, bar: usize) -> f32 {
    let level = bands.get(bar).copied().unwrap_or(0.0).clamp(0.0, 1.0);
    level * 16.0
}

fn visualizer_segment_rgba(
    segment: usize,
    max_segments: usize,
    lit: bool,
    viscolor: &VisColor,
) -> [u8; 4] {
    if !lit {
        return viscolor.background();
    }
    let gradient = viscolor.analyzer_gradient();
    let last = gradient.len() - 1;
    let from_top = max_segments - 1 - segment;
    let denom = (max_segments - 1).max(1);
    let index = ((from_top * last) + denom / 2) / denom;
    gradient[index.min(last)]
}

#[composable]
fn Sprite(image: ImageBitmap, source: SpriteRect, x: f32, y: f32, scale: f32) {
    let w = scaled(source.2, scale);
    let h = scaled(source.3, scale);
    Canvas(
        Modifier::empty()
            .size_points(w, h)
            .absolute_offset(scaled(x, scale), scaled(y, scale)),
        move |scope| {
            let dst = Rect {
                x: 0.0,
                y: 0.0,
                width: w,
                height: h,
            };
            scope.draw_image_src(image.clone(), to_rect(source), dst, 1.0, None);
        },
    );
}

#[composable]
fn StretchSprite(
    image: ImageBitmap,
    source: SpriteRect,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    scale: f32,
) {
    let w = scaled(width.max(1.0), scale);
    let h = scaled(height.max(1.0), scale);
    Canvas(
        Modifier::empty()
            .size_points(w, h)
            .absolute_offset(scaled(x, scale), scaled(y, scale)),
        move |scope| {
            let dst = Rect {
                x: 0.0,
                y: 0.0,
                width: w,
                height: h,
            };
            scope.draw_image_src(image.clone(), to_rect(source), dst, 1.0, None);
        },
    );
}

#[composable]
fn PressableSprite(
    image: ImageBitmap,
    normal: SpriteRect,
    pressed: SpriteRect,
    x: f32,
    y: f32,
    scale: f32,
    on_click: impl Fn() + 'static,
) {
    let is_pressed = cranpose_core::useState(|| false);
    let on_click = Rc::new(on_click);

    let current = if is_pressed.get() { pressed } else { normal };
    let w = scaled(normal.2, scale);
    let h = scaled(normal.3, scale);

    Canvas(
        Modifier::empty()
            .size_points(w, h)
            .absolute_offset(scaled(x, scale), scaled(y, scale))
            .pointer_input((), {
                move |scope: PointerInputScope| {
                    let on_click = on_click.clone();
                    async move {
                        scope
                            .await_pointer_event_scope(|await_scope| async move {
                                loop {
                                    let event = await_scope.await_pointer_event().await;
                                    match event.kind {
                                        PointerEventKind::Down => {
                                            is_pressed.set(true);
                                            event.consume();
                                        }
                                        PointerEventKind::Move => {
                                            if is_pressed.get()
                                                && !event.buttons.contains(PointerButton::Primary)
                                            {
                                                is_pressed.set(false);
                                            }
                                        }
                                        PointerEventKind::Up => {
                                            let was_pressed = is_pressed.get();
                                            is_pressed.set(false);
                                            let inside = event.position.x >= 0.0
                                                && event.position.x <= w
                                                && event.position.y >= 0.0
                                                && event.position.y <= h;
                                            if was_pressed && inside {
                                                on_click();
                                            }
                                            event.consume();
                                        }
                                        PointerEventKind::Cancel => {
                                            is_pressed.set(false);
                                        }
                                        PointerEventKind::Scroll
                                        | PointerEventKind::Enter
                                        | PointerEventKind::Exit => {}
                                    }
                                }
                            })
                            .await;
                    }
                }
            }),
        move |scope| {
            let dst = Rect {
                x: 0.0,
                y: 0.0,
                width: scaled(current.2, scale),
                height: scaled(current.3, scale),
            };
            scope.draw_image_src(image.clone(), to_rect(current), dst, 1.0, None);
        },
    );
}

#[composable]
fn ClickTarget(x: f32, y: f32, width: f32, height: f32, scale: f32, on_click: impl Fn() + 'static) {
    let is_pressed = cranpose_core::useState(|| false);
    let on_click = Rc::new(on_click);
    let w = scaled(width, scale);
    let h = scaled(height, scale);

    Box(
        Modifier::empty()
            .size_points(w, h)
            .absolute_offset(scaled(x, scale), scaled(y, scale))
            .pointer_input((), {
                move |scope: PointerInputScope| {
                    let on_click = on_click.clone();
                    async move {
                        scope
                            .await_pointer_event_scope(|await_scope| async move {
                                loop {
                                    let event = await_scope.await_pointer_event().await;
                                    match event.kind {
                                        PointerEventKind::Down => {
                                            is_pressed.set(true);
                                            event.consume();
                                        }
                                        PointerEventKind::Move => {
                                            if is_pressed.get()
                                                && !event.buttons.contains(PointerButton::Primary)
                                            {
                                                is_pressed.set(false);
                                            }
                                        }
                                        PointerEventKind::Up => {
                                            let was_pressed = is_pressed.get();
                                            is_pressed.set(false);
                                            let inside = event.position.x >= 0.0
                                                && event.position.x <= w
                                                && event.position.y >= 0.0
                                                && event.position.y <= h;
                                            if was_pressed && inside {
                                                on_click();
                                            }
                                            event.consume();
                                        }
                                        PointerEventKind::Cancel => {
                                            is_pressed.set(false);
                                        }
                                        PointerEventKind::Scroll
                                        | PointerEventKind::Enter
                                        | PointerEventKind::Exit => {}
                                    }
                                }
                            })
                            .await;
                    }
                }
            }),
        BoxSpec::default(),
        || {},
    );
}

#[composable]
fn DragSlider(
    area: ControlRect,
    on_change: impl Fn(f32) + 'static,
    on_drag_state: impl Fn(bool) + 'static,
) {
    let on_change = Rc::new(on_change);
    let on_drag_state = Rc::new(on_drag_state);

    Box(
        Modifier::empty()
            .size_points(area.scaled_width(), area.scaled_height())
            .absolute_offset(area.scaled_x(), area.scaled_y())
            .pointer_input((), {
                move |scope: PointerInputScope| {
                    let on_change = on_change.clone();
                    let on_drag_state = on_drag_state.clone();
                    async move {
                        scope
                            .await_pointer_event_scope(|await_scope| async move {
                                let mut dragging = false;
                                loop {
                                    let event = await_scope.await_pointer_event().await;
                                    match event.kind {
                                        PointerEventKind::Down => {
                                            dragging = true;
                                            on_drag_state(true);
                                            let value = (event.position.x / area.scaled_width())
                                                .clamp(0.0, 1.0);
                                            on_change(value);
                                            event.consume();
                                        }
                                        PointerEventKind::Move if dragging => {
                                            let value = (event.position.x / area.scaled_width())
                                                .clamp(0.0, 1.0);
                                            on_change(value);
                                            event.consume();
                                        }
                                        PointerEventKind::Up | PointerEventKind::Cancel
                                            if dragging =>
                                        {
                                            dragging = false;
                                            on_drag_state(false);
                                        }
                                        _ => {}
                                    }
                                }
                            })
                            .await;
                    }
                }
            }),
        BoxSpec::default(),
        || {},
    );
}

#[composable]
fn VerticalDragSlider(
    area: ControlRect,
    invert: bool,
    on_change: impl Fn(f32) + 'static,
    on_drag_state: impl Fn(bool) + 'static,
) {
    let on_change = Rc::new(on_change);
    let on_drag_state = Rc::new(on_drag_state);

    Box(
        Modifier::empty()
            .size_points(area.scaled_width(), area.scaled_height())
            .absolute_offset(area.scaled_x(), area.scaled_y())
            .pointer_input((), {
                move |scope: PointerInputScope| {
                    let on_change = on_change.clone();
                    let on_drag_state = on_drag_state.clone();
                    async move {
                        scope
                            .await_pointer_event_scope(|await_scope| async move {
                                let mut dragging = false;
                                loop {
                                    let event = await_scope.await_pointer_event().await;
                                    match event.kind {
                                        PointerEventKind::Down => {
                                            dragging = true;
                                            on_drag_state(true);
                                            let raw = (event.position.y / area.scaled_height())
                                                .clamp(0.0, 1.0);
                                            let value = if invert { 1.0 - raw } else { raw };
                                            on_change(value);
                                            event.consume();
                                        }
                                        PointerEventKind::Move if dragging => {
                                            let raw = (event.position.y / area.scaled_height())
                                                .clamp(0.0, 1.0);
                                            let value = if invert { 1.0 - raw } else { raw };
                                            on_change(value);
                                            event.consume();
                                        }
                                        PointerEventKind::Up | PointerEventKind::Cancel
                                            if dragging =>
                                        {
                                            dragging = false;
                                            on_drag_state(false);
                                        }
                                        _ => {}
                                    }
                                }
                            })
                            .await;
                    }
                }
            }),
        BoxSpec::default(),
        || {},
    );
}

#[composable]
fn WindowDragHandle(drag_target: WinampDragTarget, area: SpriteRect, scale: f32) {
    let modifier = Modifier::empty()
        .size_points(scaled(area.2, scale), scaled(area.3, scale))
        .absolute_offset(scaled(area.0, scale), scaled(area.1, scale));

    match drag_target {
        WinampDragTarget::NativeGroup => {
            Box(modifier.window_drag_area(), BoxSpec::default(), || {});
        }
        #[cfg(any(target_os = "android", all(feature = "web", target_arch = "wasm32")))]
        WinampDragTarget::Fixed(_) => {
            Box(modifier, BoxSpec::default(), || {});
        }
        WinampDragTarget::Inline(window_position) => {
            let drag_offset = cranpose_core::useState(|| None::<Point>);

            Box(
                modifier.pointer_input((), {
                    move |scope: PointerInputScope| async move {
                        scope
                            .await_pointer_event_scope(|await_scope| async move {
                                loop {
                                    let event = await_scope.await_pointer_event().await;
                                    match event.kind {
                                        PointerEventKind::Down => {
                                            let current = window_position.get();
                                            drag_offset.set(Some(Point::new(
                                                event.global_position.x - current.x,
                                                event.global_position.y - current.y,
                                            )));
                                            event.consume();
                                        }
                                        PointerEventKind::Move => {
                                            if !event.buttons.contains(PointerButton::Primary) {
                                                drag_offset.set(None);
                                                continue;
                                            }
                                            if let Some(offset) = drag_offset.get() {
                                                window_position.set(Point::new(
                                                    snap_to_pixel(
                                                        event.global_position.x - offset.x,
                                                    ),
                                                    snap_to_pixel(
                                                        event.global_position.y - offset.y,
                                                    ),
                                                ));
                                                event.consume();
                                            }
                                        }
                                        PointerEventKind::Up | PointerEventKind::Cancel => {
                                            drag_offset.set(None);
                                        }
                                        PointerEventKind::Scroll
                                        | PointerEventKind::Enter
                                        | PointerEventKind::Exit => {}
                                    }
                                }
                            })
                            .await;
                    }
                }),
                BoxSpec::default(),
                || {},
            );
        }
    }
}

#[composable]
fn WindowResizeHandle(
    drag_target: WinampDragTarget,
    direction: WindowResizeDirection,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    scale: f32,
) {
    if !matches!(drag_target, WinampDragTarget::NativeGroup) {
        return;
    }

    Box(
        Modifier::empty()
            .size_points(scaled(width, scale), scaled(height, scale))
            .absolute_offset(scaled(x, scale), scaled(y, scale))
            .window_resize_area(direction),
        BoxSpec::default(),
        || {},
    );
}

#[composable]
fn TransportButtons(cbuttons: ImageBitmap, state: MutableState<WinampState>, scale: f32) {
    {
        let state_click = state;
        PressableSprite(
            cbuttons.clone(),
            PREV_BUTTON,
            PREV_BUTTON_ACTIVE,
            POS_CBUTTONS.0,
            POS_CBUTTONS.1,
            scale,
            move || {
                previous_track(state_click);
            },
        );
    }

    {
        let state_click = state;
        PressableSprite(
            cbuttons.clone(),
            PLAY_BUTTON,
            PLAY_BUTTON_ACTIVE,
            POS_CBUTTONS.0 + 23.0,
            POS_CBUTTONS.1,
            scale,
            move || {
                play_or_resume(state_click);
            },
        );
    }

    {
        let state_click = state;
        PressableSprite(
            cbuttons.clone(),
            PAUSE_BUTTON,
            PAUSE_BUTTON_ACTIVE,
            POS_CBUTTONS.0 + 46.0,
            POS_CBUTTONS.1,
            scale,
            move || {
                pause_playback(state_click);
            },
        );
    }

    {
        let state_click = state;
        PressableSprite(
            cbuttons.clone(),
            STOP_BUTTON,
            STOP_BUTTON_ACTIVE,
            POS_CBUTTONS.0 + 69.0,
            POS_CBUTTONS.1,
            scale,
            move || {
                stop_playback(state_click);
            },
        );
    }

    {
        let state_click = state;
        PressableSprite(
            cbuttons.clone(),
            NEXT_BUTTON,
            NEXT_BUTTON_ACTIVE,
            POS_CBUTTONS.0 + 92.0,
            POS_CBUTTONS.1,
            scale,
            move || {
                next_track(state_click);
            },
        );
    }

    {
        let state_click = state;
        PressableSprite(
            cbuttons,
            EJECT_BUTTON,
            EJECT_BUTTON_ACTIVE,
            POS_EJECT.0,
            POS_EJECT.1,
            scale,
            move || {
                open_audio_files(state_click);
            },
        );
    }
}

#[cfg(target_arch = "wasm32")]
fn apply_loaded_skin(
    state: MutableState<WinampState>,
    skin_state: WinampSkinState,
    bytes: &[u8],
    label: &str,
    skin_path: Option<String>,
) {
    match load_skin(bytes) {
        Ok(skin) => {
            skin_state.set(Ok(skin));
            state.update(|s| {
                s.skin_path = skin_path.clone();
                s.status = if label.is_empty() {
                    "Loaded Skin".to_string()
                } else {
                    format!("Loaded Skin {label}")
                };
            });
        }
        Err(error) => {
            state.update(|s| s.status = format!("Skin Load Failed: {error:#}"));
        }
    }
}

#[cfg(target_os = "android")]
fn open_skin_file(state: MutableState<WinampState>, _skin_state: WinampSkinState) {
    state.update(|s| s.status = "Opening Android Skin Picker".to_string());
    match android_bridge::request_skin_import() {
        Ok(()) => {}
        Err(error) => state.update(|s| s.status = error),
    }
}

#[cfg(all(
    not(target_arch = "wasm32"),
    not(target_os = "android"),
    feature = "native-dialogs"
))]
fn open_skin_file(state: MutableState<WinampState>, skin_state: WinampSkinState) {
    state.update(|s| s.status = "Opening Skin".to_string());
    let Some(path) = rfd::FileDialog::new()
        .set_title("Open Winamp skin")
        .add_filter("Winamp skin", &["wsz", "zip"])
        .pick_file()
    else {
        state.update(|s| s.status = "Skin Open Cancelled".to_string());
        return;
    };

    let label = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("skin.wsz")
        .to_string();
    load_skin_file_background(
        state,
        skin_state,
        path,
        true,
        Some(format!("Loaded Skin {label}")),
        true,
    );
}

#[cfg(all(
    not(target_arch = "wasm32"),
    not(target_os = "android"),
    not(feature = "native-dialogs")
))]
fn open_skin_file(state: MutableState<WinampState>, _skin_state: WinampSkinState) {
    state.update(|s| s.status = "Native skin picker is unavailable".to_string());
}

#[cfg(all(feature = "web", target_arch = "wasm32"))]
fn open_skin_file(state: MutableState<WinampState>, skin_state: WinampSkinState) {
    state.update(|s| s.status = "Opening Skin".to_string());
    wasm_bindgen_futures::spawn_local(async move {
        let Some(handle) = rfd::AsyncFileDialog::new()
            .set_title("Open Winamp skin")
            .add_filter("Winamp skin", &["wsz", "zip"])
            .pick_file()
            .await
        else {
            state.update(|s| s.status = "Skin Open Cancelled".to_string());
            return;
        };
        let label = handle.file_name();
        let bytes = handle.read().await;
        apply_loaded_skin(state, skin_state, &bytes, &label, None);
    });
}

#[cfg(all(not(feature = "web"), target_arch = "wasm32"))]
fn open_skin_file(state: MutableState<WinampState>, _skin_state: WinampSkinState) {
    state.update(|s| s.status = "Web skin picker is not enabled".to_string());
}

#[cfg(target_os = "android")]
fn open_audio_files(state: MutableState<WinampState>) {
    state.update(|s| s.status = "Opening Android File Picker".to_string());
    match android_bridge::request_audio_files(AndroidLoadMode::Replace) {
        Ok(()) => {}
        Err(error) => state.update(|s| s.status = error),
    }
}

#[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
fn open_audio_files(state: MutableState<WinampState>) {
    state.update(|s| s.status = "Opening File".to_string());
    match audio::pick_audio_files() {
        Ok(Some(tracks)) => {
            replace_playlist_and_play(state, tracks);
        }
        Ok(None) => state.update(|s| s.status = "Open Cancelled".to_string()),
        Err(error) => state.update(|s| s.status = error),
    }
}

#[cfg(all(feature = "web", target_arch = "wasm32"))]
fn open_audio_files(state: MutableState<WinampState>) {
    state.update(|s| s.status = "Opening File".to_string());
    wasm_bindgen_futures::spawn_local(async move {
        match audio::pick_web_audio_file().await {
            Ok(Some(picked)) => {
                let track = picked.track.clone();
                let snapshot = state.get_non_reactive();
                match audio::play_web_bytes(picked.bytes, snapshot.volume, false) {
                    Ok(()) => {
                        state.update(|s| {
                            set_playlist_tracks(s, vec![track.clone()]);
                            s.current_index = Some(0);
                            set_playlist_selection(s, [0]);
                            scroll_playlist_to_track(s, 0);
                            s.position = 0.0;
                            s.elapsed_seconds = 0.0;
                            s.duration_seconds = None;
                            s.title_marquee_phase = 0.0;
                            s.playback = PlaybackState::Playing;
                            s.status = format!("Playing {}", track.display_title());
                            refresh_shuffle_order(s);
                        });
                    }
                    Err(error) => state.update(|s| s.status = error),
                }
            }
            Ok(None) => state.update(|s| s.status = "Open Cancelled".to_string()),
            Err(error) => state.update(|s| s.status = error),
        }
    });
}

#[cfg(all(not(feature = "web"), target_arch = "wasm32"))]
fn open_audio_files(state: MutableState<WinampState>) {
    state.update(|s| s.status = "Web picker is not enabled".to_string());
}

#[cfg(target_os = "android")]
fn add_audio_files(state: MutableState<WinampState>) {
    state.update(|s| s.status = "Opening Android File Picker".to_string());
    match android_bridge::request_audio_files(AndroidLoadMode::Append) {
        Ok(()) => {}
        Err(error) => state.update(|s| s.status = error),
    }
}

#[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
fn add_audio_files(state: MutableState<WinampState>) {
    state.update(|s| s.status = "Adding File".to_string());
    match audio::pick_audio_files() {
        Ok(Some(tracks)) => {
            append_playlist_and_play(state, tracks);
        }
        Ok(None) => state.update(|s| s.status = "Add Cancelled".to_string()),
        Err(error) => state.update(|s| s.status = error),
    }
}

#[cfg(target_arch = "wasm32")]
fn add_audio_files(state: MutableState<WinampState>) {
    state.update(|s| {
        s.status = "Playlist add unavailable in the web widget".to_string();
    });
}

#[cfg(target_os = "android")]
fn add_audio_folder(state: MutableState<WinampState>) {
    state.update(|s| s.status = "Opening Android Folder Picker".to_string());
    match android_bridge::request_audio_folder(AndroidLoadMode::Append) {
        Ok(()) => {}
        Err(error) => state.update(|s| s.status = error),
    }
}

#[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
fn add_audio_folder(state: MutableState<WinampState>) {
    state.update(|s| s.status = "Adding Folder".to_string());
    match audio::pick_audio_folder_path() {
        Ok(Some(folder)) => {
            run_native_io(
                move || audio::tracks_from_audio_folder(folder),
                move |tracks| {
                    append_playlist_and_play(state, tracks);
                },
            );
        }
        Ok(None) => state.update(|s| s.status = "Add Cancelled".to_string()),
        Err(error) => state.update(|s| s.status = error),
    }
}

#[cfg(target_arch = "wasm32")]
fn add_audio_folder(state: MutableState<WinampState>) {
    state.update(|s| {
        s.status = "Folder picker unavailable in the web widget".to_string();
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn replace_playlist_and_play(state: MutableState<WinampState>, tracks: Vec<Track>) {
    if tracks.is_empty() {
        state.update(|s| {
            s.status = "No Supported Audio".to_string();
        });
        return;
    }

    state.update(|s| {
        replace_playlist_tracks(s, tracks);
    });
    start_track(state, 0);
}

#[cfg(not(target_arch = "wasm32"))]
fn append_playlist_and_play(state: MutableState<WinampState>, tracks: Vec<Track>) {
    if tracks.is_empty() {
        state.update(|s| {
            s.status = "No Supported Audio".to_string();
        });
        return;
    }

    let should_start = state.get_non_reactive().playlist.is_empty();
    state.update(|s| {
        append_playlist_tracks(s, tracks);
    });
    if should_start {
        start_track(state, 0);
    }
}

#[cfg(any(not(target_arch = "wasm32"), test))]
fn replace_playlist_tracks(state: &mut WinampState, tracks: Vec<Track>) {
    set_playlist_tracks(state, tracks);
    state.current_index = Some(0);
    set_playlist_selection(state, [0]);
    state.playlist_scroll = 0.0;
    state.position = 0.0;
    state.elapsed_seconds = 0.0;
    state.duration_seconds = None;
    state.title_marquee_phase = 0.0;
    state.status = format!("Loaded {} Track(s)", state.playlist.len());
    refresh_shuffle_order(state);
}

#[cfg(any(not(target_arch = "wasm32"), test))]
fn append_playlist_tracks(state: &mut WinampState, tracks: Vec<Track>) -> bool {
    let was_empty = state.playlist.is_empty();
    let added_count = tracks.len();

    playlist_tracks_mut(state).extend(tracks);
    if was_empty {
        state.current_index = Some(0);
        set_playlist_selection(state, [0]);
        state.playlist_scroll = 0.0;
        state.position = 0.0;
        state.elapsed_seconds = 0.0;
        state.duration_seconds = None;
        state.title_marquee_phase = 0.0;
        state.status = format!("Loaded {} Track(s)", state.playlist.len());
    } else {
        normalize_playlist_selection(state);
        state.status = format!("Added {added_count} Track(s)");
    }
    refresh_shuffle_order(state);

    was_empty
}

fn set_playlist_selection<I>(state: &mut WinampState, indices: I)
where
    I: IntoIterator<Item = usize>,
{
    state.selected_indices = indices.into_iter().collect();
    normalize_playlist_selection(state);
    state.selection_anchor = state.selected_indices.last().copied();
}

fn normalize_playlist_selection(state: &mut WinampState) {
    state.selected_indices.sort_unstable();
    state.selected_indices.dedup();
    let len = state.playlist.len();
    state.selected_indices.retain(|index| *index < len);
    if state
        .selection_anchor
        .is_some_and(|anchor| anchor >= state.playlist.len())
    {
        state.selection_anchor = state.selected_indices.last().copied();
    }
}

fn selected_playlist_indices_or_current(state: &WinampState) -> Vec<usize> {
    let mut indices = state
        .selected_indices
        .iter()
        .copied()
        .filter(|index| *index < state.playlist.len())
        .collect::<Vec<_>>();
    if indices.is_empty() {
        if let Some(index) = state
            .current_index
            .filter(|index| *index < state.playlist.len())
        {
            indices.push(index);
        }
    }
    indices.sort_unstable();
    indices.dedup();
    indices
}

fn remove_all_tracks(state: MutableState<WinampState>) {
    let snapshot = state.get_non_reactive();
    if snapshot.playlist.is_empty() {
        state.update(|s| s.status = "Playlist Empty".to_string());
        return;
    }
    if snapshot.playback != PlaybackState::Stopped {
        let _ = audio::stop();
    }
    state.update(|s| {
        clear_playlist_state(s);
    });
}

fn clear_playlist_state(state: &mut WinampState) {
    set_playlist_tracks(state, Vec::new());
    state.current_index = None;
    state.selected_indices.clear();
    state.selection_anchor = None;
    state.shuffle_order.clear();
    state.playlist_scroll = 0.0;
    state.playback = PlaybackState::Stopped;
    state.position = 0.0;
    state.elapsed_seconds = 0.0;
    state.duration_seconds = None;
    state.title_marquee_phase = 0.0;
    state.status = "Playlist Empty".to_string();
}

fn remove_selected_tracks(state: MutableState<WinampState>) {
    let snapshot = state.get_non_reactive();
    let indices = selected_playlist_indices_or_current(&snapshot);
    remove_playlist_indices_action(state, indices);
}

fn remove_unselected_tracks(state: MutableState<WinampState>) {
    let snapshot = state.get_non_reactive();
    let selected = selected_playlist_indices_or_current(&snapshot)
        .into_iter()
        .collect::<HashSet<_>>();
    let indices = snapshot
        .playlist
        .iter()
        .enumerate()
        .filter_map(|(index, _)| (!selected.contains(&index)).then_some(index))
        .collect::<Vec<_>>();
    remove_playlist_indices_action(state, indices);
}

fn remove_duplicate_tracks(state: MutableState<WinampState>) {
    let snapshot = state.get_non_reactive();
    let indices = duplicate_playlist_indices(snapshot.playlist.as_slice());
    remove_playlist_indices_action(state, indices);
}

fn duplicate_playlist_indices(playlist: &[Track]) -> Vec<usize> {
    let mut seen = HashSet::new();
    let mut indices = Vec::new();
    for (index, track) in playlist.iter().enumerate() {
        if !seen.insert(duplicate_track_key(track)) {
            indices.push(index);
        }
    }
    indices
}

fn duplicate_track_key(track: &Track) -> String {
    track
        .path
        .as_deref()
        .filter(|path| !path.is_empty())
        .unwrap_or_else(|| track.display_title())
        .to_ascii_lowercase()
}

fn remove_playlist_indices_action(state: MutableState<WinampState>, indices: Vec<usize>) {
    let snapshot = state.get_non_reactive();
    if snapshot.playlist.is_empty() {
        state.update(|s| s.status = "Playlist Empty".to_string());
        return;
    }
    if indices.is_empty() {
        state.update(|s| s.status = "No Tracks Removed".to_string());
        return;
    }
    if snapshot
        .current_index
        .is_some_and(|current| indices.contains(&current))
        && snapshot.playback != PlaybackState::Stopped
    {
        let _ = audio::stop();
    }

    state.update(|s| {
        let removed = remove_playlist_indices(s, &indices);
        if removed > 0 && !s.playlist.is_empty() {
            s.status = format!("Removed {removed} Track(s)");
        }
    });
}

fn remove_playlist_indices(state: &mut WinampState, indices: &[usize]) -> usize {
    let len = state.playlist.len();
    if len == 0 {
        clear_playlist_state(state);
        return 0;
    }

    let mut remove = vec![false; len];
    for index in indices.iter().copied().filter(|index| *index < len) {
        remove[index] = true;
    }
    let removed_count = remove.iter().filter(|remove| **remove).count();
    if removed_count == 0 {
        return 0;
    }

    let old_current = state.current_index.filter(|index| *index < len);
    let removed_current = old_current.is_some_and(|index| remove[index]);
    let old_selected = state.selected_indices.clone();
    let old_playlist = Rc::clone(&state.playlist);
    let mut old_to_new = vec![None; len];
    let mut new_playlist = Vec::with_capacity(len - removed_count);

    for (old_index, track) in old_playlist.iter().cloned().enumerate() {
        if remove[old_index] {
            continue;
        }
        old_to_new[old_index] = Some(new_playlist.len());
        new_playlist.push(track);
    }

    set_playlist_tracks(state, new_playlist);
    if state.playlist.is_empty() {
        clear_playlist_state(state);
        return removed_count;
    }

    state.current_index = old_current
        .and_then(|index| old_to_new[index])
        .or_else(|| nearest_surviving_playlist_index(old_current.unwrap_or(0), &old_to_new))
        .or(Some(0));

    if removed_current {
        state.playback = PlaybackState::Stopped;
        state.position = 0.0;
        state.elapsed_seconds = 0.0;
        state.duration_seconds = None;
        state.title_marquee_phase = 0.0;
    }

    state.selected_indices = old_selected
        .into_iter()
        .filter_map(|index| old_to_new.get(index).copied().flatten())
        .collect();
    normalize_playlist_selection(state);
    if state.selected_indices.is_empty() {
        if let Some(current) = state.current_index {
            set_playlist_selection(state, [current]);
        }
    }
    if let Some(current) = state.current_index {
        scroll_playlist_to_track(state, current);
    }
    refresh_shuffle_order(state);
    removed_count
}

fn nearest_surviving_playlist_index(
    old_index: usize,
    old_to_new: &[Option<usize>],
) -> Option<usize> {
    old_to_new
        .iter()
        .skip(old_index)
        .find_map(|index| *index)
        .or_else(|| {
            old_to_new
                .iter()
                .take(old_index)
                .rev()
                .find_map(|index| *index)
        })
}

#[cfg(test)]
fn remove_playlist_track_at(state: &mut WinampState, index: usize) -> bool {
    if index >= state.playlist.len() {
        state.status = "Track Missing".to_string();
        return false;
    }

    let removed_number = index + 1;
    remove_playlist_indices(state, &[index]);
    if !state.playlist.is_empty() {
        state.status = format!("Removed Track {removed_number}");
    }
    true
}

fn select_no_tracks(state: MutableState<WinampState>) {
    state.update(|s| {
        s.selected_indices.clear();
        s.selection_anchor = None;
        s.status = "Selection Cleared".to_string();
    });
}

fn select_all_tracks(state: MutableState<WinampState>) {
    state.update(|s| {
        let len = s.playlist.len();
        set_playlist_selection(s, 0..len);
        s.status = format!("Selected {} Track(s)", s.selected_indices.len());
    });
}

fn invert_track_selection(state: MutableState<WinampState>) {
    state.update(|s| {
        let selected = s.selected_indices.iter().copied().collect::<HashSet<_>>();
        let inverted = (0..s.playlist.len())
            .filter(|index| !selected.contains(index))
            .collect::<Vec<_>>();
        set_playlist_selection(s, inverted);
        s.status = format!("Selected {} Track(s)", s.selected_indices.len());
    });
}

fn select_search_matches(state: MutableState<WinampState>) {
    state.update(|s| {
        if !s.playlist_search_visible {
            s.playlist_search_query = playlist_search_query(s).unwrap_or_default();
            s.playlist_search_revision = s.playlist_search_revision.wrapping_add(1);
        }
        s.playlist_search_visible = true;
        let query = s.playlist_search_query.clone();
        apply_playlist_search_filter_in_state(s, &query);
        s.status = format!("Selected {} Match(es)", s.selected_indices.len());
    });
}

fn apply_playlist_search_filter_in_state(state: &mut WinampState, query: &str) {
    let query = query.trim().to_ascii_lowercase();
    if query.is_empty() {
        state.selected_indices.clear();
        state.selection_anchor = None;
        return;
    }
    let matches = state
        .playlist
        .iter()
        .enumerate()
        .filter_map(|(index, track)| {
            let title = track.display_title().to_ascii_lowercase();
            let path = track.path.as_deref().unwrap_or("").to_ascii_lowercase();
            (title.contains(&query) || path.contains(&query)).then_some(index)
        })
        .collect::<Vec<_>>();
    set_playlist_selection(state, matches);
}

fn set_text_field_text(field: &TextFieldState, text: &str) {
    if field.text() == text {
        return;
    }
    field.edit(|buffer| {
        let len = buffer.text().len();
        buffer.replace(TextRange::new(0, len), text);
        buffer.place_cursor_at_end();
    });
}

fn playlist_search_query(state: &WinampState) -> Option<String> {
    let index = state
        .selection_anchor
        .or(state.current_index)
        .filter(|index| *index < state.playlist.len())?;
    let title = state.playlist[index].display_title();
    let artist = parsed_track_artist(title);
    let query = artist.filter(|artist| artist.len() >= 2).unwrap_or(title);
    let query = query.trim();
    (!query.is_empty()).then(|| query.to_string())
}

fn sort_playlist_by_title(state: MutableState<WinampState>) {
    state.update(
        |s| {
            if sort_playlist_tracks_by_field(s, PlaylistSortField::Title) {}
        },
    );
}

#[cfg(test)]
fn sort_playlist_tracks_by_title(state: &mut WinampState) -> bool {
    sort_playlist_tracks_by_field(state, PlaylistSortField::Title)
}

fn sort_playlist_by_artist(state: MutableState<WinampState>) {
    state.update(
        |s| {
            if sort_playlist_tracks_by_field(s, PlaylistSortField::Artist) {}
        },
    );
}

fn sort_playlist_by_file_name(state: MutableState<WinampState>) {
    state.update(
        |s| {
            if sort_playlist_tracks_by_field(s, PlaylistSortField::FileName) {}
        },
    );
}

fn sort_playlist_by_path(state: MutableState<WinampState>) {
    state.update(
        |s| {
            if sort_playlist_tracks_by_field(s, PlaylistSortField::Path) {}
        },
    );
}

fn sort_playlist_by_extension(state: MutableState<WinampState>) {
    state.update(
        |s| {
            if sort_playlist_tracks_by_field(s, PlaylistSortField::Extension) {}
        },
    );
}

fn sort_playlist_by_duration(state: MutableState<WinampState>) {
    state.update(
        |s| {
            if sort_playlist_tracks_by_field(s, PlaylistSortField::Duration) {}
        },
    );
}

fn sort_playlist_by_genre(state: MutableState<WinampState>) {
    state.update(
        |s| {
            if sort_playlist_tracks_by_field(s, PlaylistSortField::Genre) {}
        },
    );
}

fn sort_playlist_by_tag(state: MutableState<WinampState>) {
    state.update(
        |s| {
            if sort_playlist_tracks_by_field(s, PlaylistSortField::Tag) {}
        },
    );
}

#[derive(Clone, Copy)]
enum PlaylistSortField {
    Title,
    Artist,
    FileName,
    Path,
    Extension,
    Genre,
    Duration,
    Tag,
}

fn sort_playlist_tracks_by_field(state: &mut WinampState, field: PlaylistSortField) -> bool {
    if state.playlist.len() < 2 {
        state.status = "Playlist Sorted".to_string();
        return false;
    }

    let current_track = state
        .current_index
        .and_then(|index| state.playlist.get(index))
        .cloned();
    let selected_tracks = selected_tracks_snapshot(state);

    playlist_tracks_mut(state).sort_by(|left, right| {
        compare_tracks_by_field(left, right, field)
            .then_with(|| left.title.cmp(&right.title))
            .then_with(|| left.path.cmp(&right.path))
    });
    restore_current_and_selection_after_reorder(state, current_track, selected_tracks);
    state.status = "Playlist Sorted".to_string();
    refresh_shuffle_order(state);
    true
}

fn compare_tracks_by_field(
    left: &Track,
    right: &Track,
    field: PlaylistSortField,
) -> std::cmp::Ordering {
    match field {
        PlaylistSortField::Duration => left
            .duration_seconds
            .unwrap_or(f32::MAX)
            .total_cmp(&right.duration_seconds.unwrap_or(f32::MAX)),
        _ => playlist_sort_key(left, field).cmp(&playlist_sort_key(right, field)),
    }
}

fn playlist_sort_key(track: &Track, field: PlaylistSortField) -> String {
    match field {
        PlaylistSortField::Title => track.display_title().to_ascii_lowercase(),
        PlaylistSortField::Artist => parsed_track_artist(track.display_title())
            .unwrap_or(track.display_title())
            .to_ascii_lowercase(),
        PlaylistSortField::FileName => playlist_file_stem(track).to_ascii_lowercase(),
        PlaylistSortField::Path => track.path.as_deref().unwrap_or("").to_ascii_lowercase(),
        PlaylistSortField::Extension => playlist_path_extension(track).to_ascii_lowercase(),
        PlaylistSortField::Genre => playlist_parent_folder(track).to_ascii_lowercase(),
        PlaylistSortField::Duration => String::new(),
        PlaylistSortField::Tag => format!(
            "{} {} {}",
            parsed_track_artist(track.display_title()).unwrap_or(""),
            parsed_track_title(track.display_title()),
            playlist_file_stem(track)
        )
        .to_ascii_lowercase(),
    }
}

fn parsed_track_artist(title: &str) -> Option<&str> {
    title
        .split_once(" - ")
        .map(|(artist, _)| artist.trim())
        .filter(|artist| !artist.is_empty())
}

fn parsed_track_title(title: &str) -> &str {
    title
        .split_once(" - ")
        .map(|(_, title)| title.trim())
        .filter(|title| !title.is_empty())
        .unwrap_or(title)
}

fn playlist_file_stem(track: &Track) -> String {
    track
        .path
        .as_deref()
        .and_then(|path| {
            std::path::Path::new(path)
                .file_stem()
                .or_else(|| std::path::Path::new(path).file_name())
                .and_then(|name| name.to_str())
        })
        .unwrap_or_else(|| track.display_title())
        .to_string()
}

fn playlist_parent_folder(track: &Track) -> String {
    track
        .path
        .as_deref()
        .and_then(|path| std::path::Path::new(path).parent())
        .and_then(|parent| parent.file_name())
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_string()
}

fn playlist_path_extension(track: &Track) -> String {
    track
        .path
        .as_deref()
        .and_then(|path| std::path::Path::new(path).extension())
        .and_then(|extension| extension.to_str())
        .unwrap_or("")
        .to_string()
}

fn selected_tracks_snapshot(state: &WinampState) -> Vec<Track> {
    state
        .selected_indices
        .iter()
        .filter_map(|index| state.playlist.get(*index).cloned())
        .collect()
}

fn restore_current_and_selection_after_reorder(
    state: &mut WinampState,
    current_track: Option<Track>,
    selected_tracks: Vec<Track>,
) {
    state.current_index = current_track
        .as_ref()
        .and_then(|current| state.playlist.iter().position(|track| track == current))
        .or_else(|| (!state.playlist.is_empty()).then_some(0));
    state.selected_indices = indices_for_tracks(state.playlist.as_slice(), &selected_tracks);
    normalize_playlist_selection(state);
    if state.selected_indices.is_empty() {
        if let Some(current) = state.current_index {
            set_playlist_selection(state, [current]);
        }
    } else {
        state.selection_anchor = state.selected_indices.last().copied();
    }
    if let Some(index) = state.current_index {
        scroll_playlist_to_track(state, index);
    }
}

fn indices_for_tracks(playlist: &[Track], targets: &[Track]) -> Vec<usize> {
    let mut used = vec![false; targets.len()];
    let mut indices = Vec::new();
    for (playlist_index, track) in playlist.iter().enumerate() {
        if let Some(target_index) = targets
            .iter()
            .enumerate()
            .find_map(|(target_index, target)| {
                (!used[target_index] && target == track).then_some(target_index)
            })
        {
            used[target_index] = true;
            indices.push(playlist_index);
        }
    }
    indices
}

fn randomize_playlist(state: MutableState<WinampState>) {
    state.update(|s| if randomize_playlist_tracks(s) {});
}

fn randomize_playlist_tracks(state: &mut WinampState) -> bool {
    if state.playlist.len() < 2 {
        state.status = "Playlist Randomized".to_string();
        return false;
    }

    let current_track = state
        .current_index
        .and_then(|index| state.playlist.get(index))
        .cloned();
    let selected_tracks = selected_tracks_snapshot(state);
    let old_playlist = Rc::clone(&state.playlist);
    let mut order = (0..old_playlist.len()).collect::<Vec<_>>();
    shuffle_indices(&mut order, random_shuffle_seed());
    set_playlist_tracks(
        state,
        order
            .into_iter()
            .map(|index| old_playlist[index].clone())
            .collect(),
    );
    restore_current_and_selection_after_reorder(state, current_track, selected_tracks);
    state.status = "Playlist Randomized".to_string();
    refresh_shuffle_order(state);
    true
}

fn new_playlist(state: MutableState<WinampState>) {
    let snapshot = state.get_non_reactive();
    if snapshot.playback != PlaybackState::Stopped {
        let _ = audio::stop();
    }
    state.update(|s| {
        clear_playlist_state(s);
        s.status = "New Playlist".to_string();
    });
}

#[cfg(target_os = "android")]
fn import_playlist(state: MutableState<WinampState>) {
    state.update(|s| s.status = "Importing Android Playlist".to_string());
    match android_bridge::request_playlist_import() {
        Ok(()) => {}
        Err(error) => state.update(|s| s.status = error),
    }
}

#[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
fn import_playlist(state: MutableState<WinampState>) {
    state.update(|s| s.status = "Importing Playlist".to_string());
    let picked = pick_playlist_file().map(|path| {
        let path = path?;
        Some(path)
    });
    match picked {
        Ok(Some(path)) => run_native_io(
            move || {
                let text = std::fs::read_to_string(&path)
                    .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
                let tracks = parse_m3u_playlist(&text, path.parent());
                Ok::<_, String>(tracks)
            },
            move |result| match result {
                Ok(tracks) if tracks.is_empty() => {
                    state.update(|s| s.status = "No Playlist Tracks".to_string());
                }
                Ok(tracks) => {
                    let _ = audio::stop();
                    state.update(|s| {
                        replace_playlist_tracks(s, tracks);
                        s.playback = PlaybackState::Stopped;
                        s.status = format!("Imported {} Track(s)", s.playlist.len());
                    });
                }
                Err(error) => state.update(|s| s.status = error),
            },
        ),
        Ok(None) => state.update(|s| s.status = "Import Cancelled".to_string()),
        Err(error) => state.update(|s| s.status = error),
    }
}

#[cfg(target_arch = "wasm32")]
fn import_playlist(state: MutableState<WinampState>) {
    state.update(|s| s.status = "Playlist import unavailable in the web widget".to_string());
}

#[cfg(target_os = "android")]
fn export_playlist(state: MutableState<WinampState>) {
    let snapshot = state.get_non_reactive();
    if snapshot.playlist.is_empty() {
        state.update(|s| s.status = "Playlist Empty".to_string());
        return;
    }

    state.update(|s| s.status = "Exporting Android Playlist".to_string());
    let text = format_m3u_playlist(snapshot.playlist.as_slice());
    match android_bridge::request_playlist_export(&text) {
        Ok(()) => {}
        Err(error) => state.update(|s| s.status = error),
    }
}

#[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
fn export_playlist(state: MutableState<WinampState>) {
    let snapshot = state.get_non_reactive();
    if snapshot.playlist.is_empty() {
        state.update(|s| s.status = "Playlist Empty".to_string());
        return;
    }

    state.update(|s| s.status = "Exporting Playlist".to_string());
    match save_playlist_file().map(|path| {
        let path = path?;
        let text = format_m3u_playlist(snapshot.playlist.as_slice());
        Some((path, text))
    }) {
        Ok(Some((path, text))) => run_native_io(
            {
                let path_for_work = path.clone();
                move || {
                    std::fs::write(&path_for_work, text).map_err(|error| {
                        format!("failed to write {}: {error}", path_for_work.display())
                    })?;
                    Ok::<_, String>(path_for_work)
                }
            },
            move |result| match result {
                Ok(path) => {
                    state.update(|s| s.status = format!("Exported {}", path.display()));
                }
                Err(error) => state.update(|s| s.status = error),
            },
        ),
        Ok(None) => state.update(|s| s.status = "Export Cancelled".to_string()),
        Err(error) => state.update(|s| s.status = error),
    }
}

#[cfg(target_arch = "wasm32")]
fn export_playlist(state: MutableState<WinampState>) {
    state.update(|s| s.status = "Playlist export unavailable in the web widget".to_string());
}

#[cfg(all(
    not(target_arch = "wasm32"),
    not(target_os = "android"),
    not(target_os = "ios"),
    feature = "native-dialogs"
))]
fn pick_playlist_file() -> Result<Option<std::path::PathBuf>, String> {
    Ok(rfd::FileDialog::new()
        .set_title("Import playlist")
        .add_filter("Playlist", &["m3u", "m3u8", "M3U", "M3U8"])
        .pick_file())
}

#[cfg(all(
    not(target_arch = "wasm32"),
    not(target_os = "android"),
    not(all(not(target_os = "ios"), feature = "native-dialogs"))
))]
fn pick_playlist_file() -> Result<Option<std::path::PathBuf>, String> {
    Err("native playlist picker is not available on this target yet".to_string())
}

#[cfg(all(
    not(target_arch = "wasm32"),
    not(target_os = "android"),
    not(target_os = "ios"),
    feature = "native-dialogs"
))]
fn save_playlist_file() -> Result<Option<std::path::PathBuf>, String> {
    Ok(rfd::FileDialog::new()
        .set_title("Export playlist")
        .set_file_name("playlist.m3u")
        .add_filter("Playlist", &["m3u", "m3u8"])
        .save_file())
}

#[cfg(all(
    not(target_arch = "wasm32"),
    not(target_os = "android"),
    not(all(not(target_os = "ios"), feature = "native-dialogs"))
))]
fn save_playlist_file() -> Result<Option<std::path::PathBuf>, String> {
    Err("native playlist saver is not available on this target yet".to_string())
}

#[cfg(any(not(target_arch = "wasm32"), test))]
fn parse_m3u_playlist(input: &str, base_dir: Option<&std::path::Path>) -> Vec<Track> {
    let mut tracks = Vec::new();
    let mut pending_extinf = None::<(Option<f32>, String)>;

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(extinf) = line.strip_prefix("#EXTINF:") {
            pending_extinf = parse_extinf(extinf);
            continue;
        }
        if line.starts_with('#') {
            continue;
        }

        let resolved_path = resolve_playlist_path(line, base_dir);
        if !is_supported_playlist_path(&resolved_path) {
            pending_extinf = None;
            continue;
        }
        let (duration_seconds, title) = pending_extinf
            .take()
            .unwrap_or_else(|| (None, playlist_title_from_path(&resolved_path)));
        tracks.push(Track {
            title: if title.is_empty() {
                playlist_title_from_path(&resolved_path)
            } else {
                title
            },
            path: Some(resolved_path),
            duration_seconds,
        });
    }

    tracks
}

#[cfg(any(not(target_arch = "wasm32"), test))]
fn parse_extinf(input: &str) -> Option<(Option<f32>, String)> {
    let (duration, title) = input.split_once(',').unwrap_or((input, ""));
    let duration_seconds = duration
        .trim()
        .parse::<f32>()
        .ok()
        .filter(|duration| *duration > 0.0);
    Some((duration_seconds, title.trim().to_string()))
}

#[cfg(any(not(target_arch = "wasm32"), test))]
fn resolve_playlist_path(path: &str, base_dir: Option<&std::path::Path>) -> String {
    let path = path.trim();
    if path.starts_with("file://") {
        return path.trim_start_matches("file://").to_string();
    }
    let candidate = std::path::Path::new(path);
    if candidate.is_absolute() {
        return path.to_string();
    }
    base_dir
        .map(|base| base.join(candidate).to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

#[cfg(any(not(target_arch = "wasm32"), test))]
fn is_supported_playlist_path(path: &str) -> bool {
    std::path::Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            let extension = extension.to_ascii_lowercase();
            audio::supported_audio_extensions()
                .iter()
                .any(|candidate| *candidate == extension)
        })
        .unwrap_or(false)
}

#[cfg(any(not(target_arch = "wasm32"), test))]
fn playlist_title_from_path(path: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .or_else(|| std::path::Path::new(path).file_name())
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("Untitled")
        .to_string()
}

#[cfg(any(not(target_arch = "wasm32"), test))]
fn format_m3u_playlist(playlist: &[Track]) -> String {
    let mut lines = vec!["#EXTM3U".to_string()];
    for track in playlist {
        if let Some(duration) = track.duration_seconds {
            lines.push(format!(
                "#EXTINF:{},{}",
                duration.round().max(0.0) as u32,
                track.display_title()
            ));
        }
        lines.push(
            track
                .path
                .clone()
                .unwrap_or_else(|| track.display_title().to_string()),
        );
    }
    lines.join("\n") + "\n"
}

#[cfg(target_os = "android")]
fn handle_android_bridge_results(state: MutableState<WinampState>, skin_state: WinampSkinState) {
    for result in android_bridge::take_results() {
        match result {
            AndroidBridgeResult::AudioPaths { mode, paths } => {
                let text = paths
                    .iter()
                    .map(|path| path.to_string_lossy())
                    .collect::<Vec<_>>()
                    .join("\n");
                let tracks = parse_m3u_playlist(&text, None);
                if tracks.is_empty() {
                    state.update(|s| s.status = "No Android Audio Tracks".to_string());
                    continue;
                }
                match mode {
                    AndroidLoadMode::Replace => {
                        replace_playlist_and_play(state, tracks);
                    }
                    AndroidLoadMode::Append => {
                        append_playlist_and_play(state, tracks);
                    }
                }
            }
            AndroidBridgeResult::PlaylistImport { text } => {
                let tracks = parse_m3u_playlist(&text, None);
                if tracks.is_empty() {
                    state.update(|s| s.status = "No Playlist Tracks".to_string());
                } else {
                    let _ = audio::stop();
                    state.update(|s| {
                        replace_playlist_tracks(s, tracks);
                        s.playback = PlaybackState::Stopped;
                        s.status = format!("Imported {} Track(s)", s.playlist.len());
                    });
                }
            }
            AndroidBridgeResult::PlaylistExport { target } => {
                state.update(|s| {
                    s.status = if target.is_empty() {
                        "Exported Playlist".to_string()
                    } else {
                        format!("Exported {target}")
                    };
                });
            }
            AndroidBridgeResult::SkinImport { path } => {
                let label = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("skin.wsz")
                    .to_string();
                load_skin_file_background(
                    state,
                    skin_state,
                    path,
                    true,
                    Some(format!("Loaded Skin {label}")),
                    true,
                );
            }
            AndroidBridgeResult::Cancelled { operation } => {
                state.update(|s| s.status = format!("{operation} Cancelled"));
            }
            AndroidBridgeResult::Error(error) => state.update(|s| s.status = error),
        }
    }
}

fn scroll_playlist_by_rows(state: MutableState<WinampState>, rows: i32) {
    state.update(|s| scroll_playlist_by_rows_in_state(s, rows));
}

fn scroll_playlist_by_rows_in_state(state: &mut WinampState, rows: i32) {
    let max_start = state
        .playlist
        .len()
        .saturating_sub(state.playlist_visible_rows.max(1));
    if max_start == 0 {
        state.playlist_scroll = 0.0;
    }

    let start = (state.playlist_scroll.clamp(0.0, 1.0) * max_start as f32).round() as i32;
    let next = (start + rows).clamp(0, max_start as i32);
    state.playlist_scroll = next as f32 / max_start as f32;
}

fn scroll_playlist_to_track(state: &mut WinampState, index: usize) {
    state.playlist_scroll = playlist_scroll_for_track(
        index,
        state.playlist.len(),
        state.playlist_visible_rows,
        state.playlist_scroll,
    );
}

fn playlist_scroll_for_track(
    index: usize,
    len: usize,
    visible_rows: usize,
    current_scroll: f32,
) -> f32 {
    if len <= 1 {
        return 0.0;
    }

    let visible_rows = visible_rows.max(1).min(len);
    let max_start = len.saturating_sub(visible_rows);
    if max_start == 0 {
        return 0.0;
    }

    let first_visible =
        ((current_scroll.clamp(0.0, 1.0) * max_start as f32).round() as usize).min(max_start);
    let last_visible = first_visible + visible_rows - 1;
    let index = index.min(len - 1);
    if (first_visible..=last_visible).contains(&index) {
        return current_scroll.clamp(0.0, 1.0);
    }

    let centered_start = index.saturating_sub(visible_rows / 2).min(max_start);
    centered_start as f32 / max_start as f32
}

fn refresh_shuffle_order(state: &mut WinampState) {
    if !state.shuffle {
        state.shuffle_order.clear();
    }

    let len = state.playlist.len();
    if len == 0 {
        state.shuffle_order.clear();
        return;
    }

    let current = state.current_index.unwrap_or(0).min(len - 1);
    state.shuffle_order = shuffled_order(len, current);
}

fn sync_shuffle_order_to_current(state: &mut WinampState, index: usize) {
    if !state.shuffle {
        state.shuffle_order.clear();
        return;
    }

    if !valid_shuffle_order(&state.shuffle_order, state.playlist.len()) {
        state.shuffle_order = shuffled_order(state.playlist.len(), index);
    }
}

fn valid_shuffle_order(order: &[usize], len: usize) -> bool {
    if order.len() != len {
        return false;
    }

    let mut seen = vec![false; len];
    for &index in order {
        if index >= len || seen[index] {
            return false;
        }
        seen[index] = true;
    }
    true
}

fn shuffled_order(len: usize, first: usize) -> Vec<usize> {
    shuffled_order_with_seed(len, first, random_shuffle_seed())
}

fn shuffled_order_with_seed(len: usize, first: usize, seed: u64) -> Vec<usize> {
    if len == 0 {
        return Vec::new();
    }

    let first = first.min(len - 1);
    let mut rest = (0..len).filter(|index| *index != first).collect::<Vec<_>>();
    shuffle_indices(&mut rest, seed);

    let mut order = Vec::with_capacity(len);
    order.push(first);
    order.extend(rest);
    order
}

fn shuffle_indices(indices: &mut [usize], mut seed: u64) {
    for i in (1..indices.len()).rev() {
        seed = next_shuffle_seed(seed);
        let j = (seed as usize) % (i + 1);
        indices.swap(i, j);
    }
}

fn next_shuffle_seed(seed: u64) -> u64 {
    seed.wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407)
}

fn random_shuffle_seed() -> u64 {
    getrandom::u64().unwrap_or(0x9e37_79b9_7f4a_7c15)
}

fn play_or_resume(state: MutableState<WinampState>) {
    let snapshot = state.get_non_reactive();
    if snapshot.playback == PlaybackState::Paused {
        match audio::resume() {
            Ok(()) => {
                state.update(|s| {
                    s.playback = PlaybackState::Playing;
                    s.status = current_track_status(s, "Playing");
                });
            }
            Err(error) => state.update(|s| s.status = error),
        }
        return;
    }

    let Some(index) = snapshot
        .current_index
        .or_else(|| (!snapshot.playlist.is_empty()).then_some(0))
    else {
        state.update(|s| s.status = "Open File".to_string());
        return;
    };

    #[cfg(target_arch = "wasm32")]
    if snapshot
        .playlist
        .get(index)
        .map(|track| track.path.is_none())
        .unwrap_or(false)
    {
        match audio::resume() {
            Ok(()) => {
                state.update(|s| {
                    s.playback = PlaybackState::Playing;
                    s.status = current_track_status(s, "Playing");
                });
            }
            Err(error) => state.update(|s| s.status = error),
        }
        return;
    }

    start_track(state, index);
}

fn pause_playback(state: MutableState<WinampState>) {
    match audio::pause() {
        Ok(()) => {
            state.update(|s| {
                s.playback = PlaybackState::Paused;
                s.status = current_track_status(s, "Paused");
            });
        }
        Err(error) => state.update(|s| s.status = error),
    }
}

fn stop_playback(state: MutableState<WinampState>) {
    match audio::stop() {
        Ok(()) => {
            state.update(|s| {
                s.playback = PlaybackState::Stopped;
                s.position = 0.0;
                s.elapsed_seconds = 0.0;
                s.title_marquee_phase = 0.0;
                s.status = "Stopped".to_string();
            });
        }
        Err(error) => state.update(|s| s.status = error),
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TrackDirection {
    Next,
    Previous,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TrackAdvanceMode {
    Manual,
    Automatic,
}

fn next_track(state: MutableState<WinampState>) {
    advance_track(state, TrackDirection::Next, TrackAdvanceMode::Manual);
}

fn previous_track(state: MutableState<WinampState>) {
    advance_track(state, TrackDirection::Previous, TrackAdvanceMode::Manual);
}

fn advance_finished_track(state: MutableState<WinampState>) {
    advance_track(state, TrackDirection::Next, TrackAdvanceMode::Automatic);
}

fn advance_track(
    state: MutableState<WinampState>,
    direction: TrackDirection,
    mode: TrackAdvanceMode,
) {
    let snapshot = state.get_non_reactive();
    if snapshot.playlist.is_empty() {
        state.update(|s| {
            s.status = "Open File".to_string();
        });
        return;
    }

    let Some(plan) = playlist_advance_plan(&snapshot, direction, mode) else {
        finish_playlist(state);
        return;
    };

    if let Some(order) = plan.shuffle_order {
        state.update(|s| s.shuffle_order = order);
    }
    start_track(state, plan.index);
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TrackAdvancePlan {
    index: usize,
    shuffle_order: Option<Vec<usize>>,
}

fn playlist_advance_plan(
    state: &WinampState,
    direction: TrackDirection,
    mode: TrackAdvanceMode,
) -> Option<TrackAdvancePlan> {
    let len = state.playlist.len();
    if len == 0 {
        return None;
    }

    let current = state.current_index.unwrap_or(0).min(len - 1);
    let should_wrap = mode == TrackAdvanceMode::Manual || state.repeat;
    if state.shuffle {
        return shuffled_advance_plan(state, direction, should_wrap);
    }

    let index = sequential_advance_index(len, current, direction, should_wrap)?;
    Some(TrackAdvancePlan {
        index,
        shuffle_order: None,
    })
}

fn sequential_advance_index(
    len: usize,
    current: usize,
    direction: TrackDirection,
    should_wrap: bool,
) -> Option<usize> {
    match direction {
        TrackDirection::Next if current + 1 < len => Some(current + 1),
        TrackDirection::Next if should_wrap => Some(0),
        TrackDirection::Previous if current > 0 => Some(current - 1),
        TrackDirection::Previous if should_wrap => Some(len - 1),
        _ => None,
    }
}

fn shuffled_advance_plan(
    state: &WinampState,
    direction: TrackDirection,
    should_wrap: bool,
) -> Option<TrackAdvancePlan> {
    let len = state.playlist.len();
    let current = state.current_index.unwrap_or(0).min(len - 1);
    let mut replacement_order = None;
    let mut order = if valid_shuffle_order(&state.shuffle_order, len) {
        state.shuffle_order.clone()
    } else {
        let order = shuffled_order(len, current);
        replacement_order = Some(order.clone());
        order
    };

    let cursor = order
        .iter()
        .position(|index| *index == current)
        .unwrap_or(0);
    let index = match direction {
        TrackDirection::Next if cursor + 1 < order.len() => order[cursor + 1],
        TrackDirection::Next if should_wrap => {
            order = shuffled_order(len, current);
            let next_index = if order.len() > 1 { order[1] } else { order[0] };
            replacement_order = Some(order);
            next_index
        }
        TrackDirection::Previous if cursor > 0 => order[cursor - 1],
        TrackDirection::Previous if should_wrap => order[order.len() - 1],
        _ => return None,
    };

    Some(TrackAdvancePlan {
        index,
        shuffle_order: replacement_order,
    })
}

fn finish_playlist(state: MutableState<WinampState>) {
    let stop_result = audio::stop();
    state.update(|s| {
        s.playback = PlaybackState::Stopped;
        s.title_marquee_phase = 0.0;
        s.status = match stop_result {
            Ok(()) => "Stopped".to_string(),
            Err(error) => error,
        };
    });
}

fn start_track(state: MutableState<WinampState>, index: usize) {
    let snapshot = state.get_non_reactive();
    let Some(track) = snapshot.playlist.get(index).cloned() else {
        state.update(|s| s.status = "Track Missing".to_string());
        return;
    };

    #[cfg(target_arch = "wasm32")]
    if track.path.is_none() {
        match audio::seek_fraction(0.0).and_then(|()| audio::resume()) {
            Ok(()) => {
                state.update(|s| {
                    s.current_index = Some(index);
                    set_playlist_selection(s, [index]);
                    scroll_playlist_to_track(s, index);
                    sync_shuffle_order_to_current(s, index);
                    s.playback = PlaybackState::Playing;
                    s.position = 0.0;
                    s.elapsed_seconds = 0.0;
                    s.duration_seconds = None;
                    s.title_marquee_phase = 0.0;
                    s.status = format!("Playing {}", track.display_title());
                });
            }
            Err(error) => state.update(|s| {
                s.current_index = Some(index);
                set_playlist_selection(s, [index]);
                scroll_playlist_to_track(s, index);
                s.playback = PlaybackState::Stopped;
                s.title_marquee_phase = 0.0;
                s.status = error;
            }),
        }
        return;
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let volume = snapshot.volume;
        let track_for_play = track.clone();
        run_native_io(
            move || audio::play_track(&track_for_play, volume, false),
            move |result| match result {
                Ok(()) => {
                    state.update(|s| {
                        s.current_index = Some(index);
                        set_playlist_selection(s, [index]);
                        scroll_playlist_to_track(s, index);
                        sync_shuffle_order_to_current(s, index);
                        s.playback = PlaybackState::Playing;
                        s.position = 0.0;
                        s.elapsed_seconds = 0.0;
                        s.duration_seconds = None;
                        s.title_marquee_phase = 0.0;
                        s.status = format!("Playing {}", track.display_title());
                    });
                }
                Err(error) => state.update(|s| {
                    s.current_index = Some(index);
                    set_playlist_selection(s, [index]);
                    scroll_playlist_to_track(s, index);
                    s.playback = PlaybackState::Stopped;
                    s.title_marquee_phase = 0.0;
                    s.status = error;
                }),
            },
        );
    }

    #[cfg(target_arch = "wasm32")]
    match audio::play_track(&track, snapshot.volume, false) {
        Ok(()) => {
            state.update(|s| {
                s.current_index = Some(index);
                set_playlist_selection(s, [index]);
                scroll_playlist_to_track(s, index);
                sync_shuffle_order_to_current(s, index);
                s.playback = PlaybackState::Playing;
                s.position = 0.0;
                s.elapsed_seconds = 0.0;
                s.duration_seconds = None;
                s.title_marquee_phase = 0.0;
                s.status = format!("Playing {}", track.display_title());
            });
        }
        Err(error) => state.update(|s| {
            s.current_index = Some(index);
            set_playlist_selection(s, [index]);
            scroll_playlist_to_track(s, index);
            s.playback = PlaybackState::Stopped;
            s.title_marquee_phase = 0.0;
            s.status = error;
        }),
    }
}

fn handle_playlist_row_click(state: MutableState<WinampState>, index: usize) {
    let now_ms = current_time_ms();
    let modifiers = current_playlist_click_modifiers();

    let should_play =
        state.update(|s| handle_playlist_row_click_in_state(s, index, now_ms, modifiers));
    if should_play {
        start_track(state, index);
    }
}

fn handle_playlist_row_click_in_state(
    state: &mut WinampState,
    index: usize,
    now_ms: u64,
    modifiers: PlaylistClickModifiers,
) -> bool {
    if index >= state.playlist.len() {
        state.status = "Track Missing".to_string();
        return false;
    }

    let should_play = state.playlist_last_click_index == Some(index)
        && now_ms.saturating_sub(state.playlist_last_click_ms) <= PLAYLIST_DOUBLE_CLICK_MS
        && !modifiers.any();
    if should_play {
        state.playlist_last_click_index = None;
        state.playlist_last_click_ms = 0;
        return true;
    }

    select_playlist_row_in_state(state, index, modifiers);
    state.playlist_last_click_index = Some(index);
    state.playlist_last_click_ms = now_ms;
    false
}

fn select_playlist_row_in_state(
    state: &mut WinampState,
    index: usize,
    modifiers: PlaylistClickModifiers,
) {
    if index >= state.playlist.len() {
        state.status = "Track Missing".to_string();
        return;
    }

    if modifiers.shift {
        let anchor = state
            .selection_anchor
            .or_else(|| state.selected_indices.last().copied())
            .or(state.current_index)
            .unwrap_or(index)
            .min(state.playlist.len() - 1);
        let (start, end) = if anchor <= index {
            (anchor, index)
        } else {
            (index, anchor)
        };
        let range = start..=end;
        if modifiers.ctrl {
            state.selected_indices.extend(range);
            normalize_playlist_selection(state);
        } else {
            set_playlist_selection(state, range);
        }
        state.selection_anchor = Some(anchor);
    } else if modifiers.ctrl {
        if state.selected_indices.contains(&index) {
            state.selected_indices.retain(|selected| *selected != index);
            normalize_playlist_selection(state);
        } else {
            state.selected_indices.push(index);
            normalize_playlist_selection(state);
            state.selection_anchor = Some(index);
        }
    } else {
        set_playlist_selection(state, [index]);
    }

    scroll_playlist_to_track(state, index);
    state.status = if state.selected_indices.len() == 1 {
        state
            .playlist
            .get(index)
            .map(|track| format!("Selected {}", track.display_title()))
            .unwrap_or_else(|| "Selected".to_string())
    } else {
        format!("Selected {} Track(s)", state.selected_indices.len())
    };
}

#[cfg(all(feature = "web", target_arch = "wasm32"))]
fn current_time_ms() -> u64 {
    js_sys::Date::now().max(0.0).min(u64::MAX as f64) as u64
}

#[cfg(all(target_arch = "wasm32", not(feature = "web")))]
fn current_time_ms() -> u64 {
    0
}

#[cfg(not(target_arch = "wasm32"))]
fn current_time_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

fn current_playlist_click_modifiers() -> PlaylistClickModifiers {
    native_playlist_click_modifiers().unwrap_or_default()
}

#[cfg(all(
    not(target_arch = "wasm32"),
    not(target_os = "android"),
    not(target_os = "ios")
))]
fn native_playlist_click_modifiers() -> Option<PlaylistClickModifiers> {
    use x11rb::protocol::xproto::ConnectionExt as _;

    let (connection, _) = x11rb::connect(None).ok()?;
    let modifier_mapping = connection.get_modifier_mapping().ok()?.reply().ok()?;
    let keymap = connection.query_keymap().ok()?.reply().ok()?;
    let per_modifier = usize::from(modifier_mapping.keycodes_per_modifier());
    if per_modifier == 0 {
        return None;
    }

    let keycodes = modifier_mapping.keycodes;
    let key_active = |keycode: u8| -> bool {
        if keycode == 0 {
            return false;
        }
        let byte = usize::from(keycode / 8);
        let bit = keycode % 8;
        keymap
            .keys
            .get(byte)
            .is_some_and(|value| (value & (1 << bit)) != 0)
    };
    let group_active = |group: usize| -> bool {
        let start = group * per_modifier;
        let end = start + per_modifier;
        keycodes
            .get(start..end)
            .unwrap_or(&[])
            .iter()
            .copied()
            .any(key_active)
    };

    Some(PlaylistClickModifiers {
        shift: group_active(0),
        ctrl: group_active(2),
    })
}

#[cfg(not(all(
    not(target_arch = "wasm32"),
    not(target_os = "android"),
    not(target_os = "ios")
)))]
fn native_playlist_click_modifiers() -> Option<PlaylistClickModifiers> {
    None
}

fn current_track_status(state: &WinampState, prefix: &str) -> String {
    state
        .current_index
        .and_then(|index| state.playlist.get(index))
        .map(|track| format!("{prefix} {}", track.display_title()))
        .unwrap_or_else(|| prefix.to_string())
}

#[derive(Clone, Debug, PartialEq)]
struct SavedTrack {
    title: String,
    path: String,
}

#[derive(Clone, Debug, PartialEq)]
struct SavedPlayerState {
    shuffle: bool,
    repeat: bool,
    eq_visible: bool,
    playlist_visible: bool,
    eq_enabled: bool,
    eq_auto: bool,
    eq_values: [f32; 11],
    skin_path: Option<String>,
    playlist_scroll: f32,
    volume: f32,
    balance: f32,
    current_index: Option<usize>,
    tracks: Vec<SavedTrack>,
}

#[derive(Clone, Debug, PartialEq)]
struct SavedPlayerStateKey {
    shuffle: bool,
    repeat: bool,
    eq_visible: bool,
    playlist_visible: bool,
    eq_enabled: bool,
    eq_auto: bool,
    eq_values: [f32; 11],
    skin_path: Option<String>,
    playlist_scroll: f32,
    volume: f32,
    balance: f32,
    current_index: Option<usize>,
    playlist_identity: usize,
    playlist_len: usize,
}

impl SavedPlayerStateKey {
    fn from_state(state: &WinampState) -> Self {
        Self {
            shuffle: state.shuffle,
            repeat: state.repeat,
            eq_visible: state.eq_visible,
            playlist_visible: state.playlist_visible,
            eq_enabled: state.eq_enabled,
            eq_auto: state.eq_auto,
            eq_values: state.eq_values.map(clamp01),
            skin_path: state.skin_path.clone(),
            playlist_scroll: clamp01(state.playlist_scroll),
            volume: clamp01(state.volume),
            balance: clamp01(state.balance),
            current_index: state.current_index,
            playlist_identity: Rc::as_ptr(&state.playlist) as *const () as usize,
            playlist_len: state.playlist.len(),
        }
    }
}

impl Default for SavedPlayerState {
    fn default() -> Self {
        Self::from_state(&WinampState::default())
    }
}

impl SavedPlayerState {
    fn from_state(state: &WinampState) -> Self {
        let mut current_index = None;
        let mut tracks = Vec::new();
        for (index, track) in state.playlist.iter().enumerate() {
            let Some(path) = track.path.as_ref().filter(|path| !path.is_empty()) else {
                continue;
            };
            if state.current_index == Some(index) {
                current_index = Some(tracks.len());
            }
            tracks.push(SavedTrack {
                title: track.title.clone(),
                path: path.clone(),
            });
        }

        Self {
            shuffle: state.shuffle,
            repeat: state.repeat,
            eq_visible: state.eq_visible,
            playlist_visible: state.playlist_visible,
            eq_enabled: state.eq_enabled,
            eq_auto: state.eq_auto,
            eq_values: state.eq_values.map(clamp01),
            skin_path: state.skin_path.clone(),
            playlist_scroll: clamp01(state.playlist_scroll),
            volume: clamp01(state.volume),
            balance: clamp01(state.balance),
            current_index,
            tracks,
        }
    }
}

fn restore_saved_player_state(saved: SavedPlayerState) -> WinampState {
    let mut state = WinampState {
        shuffle: saved.shuffle,
        repeat: saved.repeat,
        eq_visible: saved.eq_visible,
        playlist_visible: saved.playlist_visible,
        eq_enabled: saved.eq_enabled,
        eq_auto: saved.eq_auto,
        eq_values: saved.eq_values.map(clamp01),
        skin_path: valid_saved_skin_path(saved.skin_path),
        playlist_scroll: clamp01(saved.playlist_scroll),
        volume: clamp01(saved.volume),
        balance: clamp01(saved.balance),
        ..WinampState::default()
    };
    let saved_current_index = saved.current_index;
    let mut restored_current_index = None;
    let mut tracks = Vec::new();
    for (saved_index, track) in saved.tracks.into_iter().enumerate() {
        let Some(track) = restore_saved_track(track) else {
            continue;
        };
        if saved_current_index == Some(saved_index) {
            restored_current_index = Some(tracks.len());
        }
        tracks.push(track);
    }
    set_playlist_tracks(&mut state, tracks);
    if !state.playlist.is_empty() {
        state.current_index = restored_current_index.or(Some(0));
        if let Some(index) = state.current_index {
            set_playlist_selection(&mut state, [index]);
        }
        state.status = format!("Restored {} Track(s)", state.playlist.len());
    }
    state
}

fn restore_saved_track(track: SavedTrack) -> Option<Track> {
    if track.path.is_empty() {
        return None;
    }

    #[cfg(not(target_arch = "wasm32"))]
    if !std::path::Path::new(&track.path).is_file() {
        return None;
    }

    let title = if track.title.is_empty() {
        "Untitled".to_string()
    } else {
        track.title
    };
    Some(audio::track_from_title_path(title, track.path))
}

fn valid_saved_skin_path(path: Option<String>) -> Option<String> {
    let path = path.filter(|path| !path.is_empty())?;

    #[cfg(not(target_arch = "wasm32"))]
    {
        std::path::Path::new(&path).is_file().then_some(path)
    }

    #[cfg(target_arch = "wasm32")]
    {
        Some(path)
    }
}

fn serialize_player_state(config: &SavedPlayerState) -> String {
    let mut lines = vec![
        "version=1".to_string(),
        format!("shuffle={}", bool_value(config.shuffle)),
        format!("repeat={}", bool_value(config.repeat)),
        format!("eq_visible={}", bool_value(config.eq_visible)),
        format!("playlist_visible={}", bool_value(config.playlist_visible)),
        format!("eq_enabled={}", bool_value(config.eq_enabled)),
        format!("eq_auto={}", bool_value(config.eq_auto)),
        format!("playlist_scroll={:.6}", clamp01(config.playlist_scroll)),
        format!("volume={:.6}", clamp01(config.volume)),
        format!("balance={:.6}", clamp01(config.balance)),
        format!(
            "skin_path={}",
            config
                .skin_path
                .as_deref()
                .map(hex_encode)
                .unwrap_or_default()
        ),
        format!(
            "current_index={}",
            config
                .current_index
                .map(|index| index.to_string())
                .unwrap_or_default()
        ),
        format!(
            "eq_values={}",
            config
                .eq_values
                .iter()
                .map(|value| format!("{:.6}", clamp01(*value)))
                .collect::<Vec<_>>()
                .join(",")
        ),
    ];

    for track in &config.tracks {
        lines.push(format!(
            "track={}\t{}",
            hex_encode(&track.title),
            hex_encode(&track.path)
        ));
    }

    lines.join("\n") + "\n"
}

fn parse_player_state(input: &str) -> SavedPlayerState {
    let mut config = SavedPlayerState::default();
    config.tracks.clear();
    config.current_index = None;

    for line in input.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        apply_player_state_value(&mut config, key.trim(), value.trim());
    }

    config
}

fn apply_player_state_value(config: &mut SavedPlayerState, key: &str, value: &str) {
    match key {
        "shuffle" => update_bool(&mut config.shuffle, value),
        "repeat" => update_bool(&mut config.repeat, value),
        "eq_visible" => update_bool(&mut config.eq_visible, value),
        "playlist_visible" => update_bool(&mut config.playlist_visible, value),
        "eq_enabled" => update_bool(&mut config.eq_enabled, value),
        "eq_auto" => update_bool(&mut config.eq_auto, value),
        "playlist_scroll" => update_f32(&mut config.playlist_scroll, value),
        "volume" => update_f32(&mut config.volume, value),
        "balance" => update_f32(&mut config.balance, value),
        "skin_path" => config.skin_path = parse_optional_hex_string(value),
        "current_index" => config.current_index = parse_optional_usize(value),
        "eq_values" => {
            if let Some(values) = parse_eq_values(value) {
                config.eq_values = values;
            }
        }
        "track" => {
            if let Some(track) = parse_saved_track(value) {
                config.tracks.push(track);
            }
        }
        _ => {}
    }
}

fn bool_value(value: bool) -> &'static str {
    if value {
        "1"
    } else {
        "0"
    }
}

fn update_bool(target: &mut bool, value: &str) {
    match value {
        "1" | "true" | "on" => *target = true,
        "0" | "false" | "off" => *target = false,
        _ => {}
    }
}

fn update_f32(target: &mut f32, value: &str) {
    if let Ok(parsed) = value.parse::<f32>() {
        *target = clamp01(parsed);
    }
}

fn parse_optional_usize(value: &str) -> Option<usize> {
    if value.is_empty() || value == "none" {
        None
    } else {
        value.parse::<usize>().ok()
    }
}

fn parse_optional_hex_string(value: &str) -> Option<String> {
    if value.is_empty() || value == "none" {
        None
    } else {
        hex_decode(value)
    }
}

fn parse_eq_values(value: &str) -> Option<[f32; 11]> {
    let values = value
        .split(',')
        .map(str::trim)
        .map(str::parse::<f32>)
        .collect::<Result<Vec<_>, _>>()
        .ok()?;
    values
        .try_into()
        .ok()
        .map(|values: [f32; 11]| values.map(clamp01))
}

fn parse_saved_track(value: &str) -> Option<SavedTrack> {
    let (title, path) = value.split_once('\t')?;
    Some(SavedTrack {
        title: hex_decode(title)?,
        path: hex_decode(path)?,
    })
}

fn hex_encode(input: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(input.len() * 2);
    for byte in input.as_bytes() {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn hex_decode(input: &str) -> Option<String> {
    let bytes = input.as_bytes();
    if !bytes.len().is_multiple_of(2) {
        return None;
    }

    let mut output = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let high = hex_value(pair[0])?;
        let low = hex_value(pair[1])?;
        output.push((high << 4) | low);
    }
    String::from_utf8(output).ok()
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn load_saved_player_state() -> Option<SavedPlayerState> {
    load_player_state_text().map(|text| parse_player_state(&text))
}

#[cfg(not(target_arch = "wasm32"))]
fn load_player_state_text() -> Option<String> {
    std::fs::read_to_string(player_config_path()).ok()
}

#[cfg(target_arch = "wasm32")]
fn load_player_state_text() -> Option<String> {
    browser_storage()?
        .get_item(PLAYER_STORAGE_KEY)
        .ok()
        .flatten()
}

fn save_player_state(config: &SavedPlayerState) -> Result<(), String> {
    save_player_state_text(&serialize_player_state(config))
}

#[cfg(not(target_arch = "wasm32"))]
fn save_player_state_background(config: SavedPlayerState) {
    std::thread::spawn(move || {
        let _ = save_player_state(&config);
    });
}

#[cfg(target_arch = "wasm32")]
fn save_player_state_background(config: SavedPlayerState) {
    let _ = save_player_state(&config);
}

#[cfg(not(target_arch = "wasm32"))]
fn save_player_state_text(text: &str) -> Result<(), String> {
    let path = player_config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    std::fs::write(&path, text)
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

#[cfg(target_arch = "wasm32")]
fn save_player_state_text(text: &str) -> Result<(), String> {
    browser_storage()
        .ok_or_else(|| "browser localStorage is unavailable".to_string())?
        .set_item(PLAYER_STORAGE_KEY, text)
        .map_err(js_storage_error)
}

#[cfg(not(target_arch = "wasm32"))]
fn player_config_path() -> std::path::PathBuf {
    config_home_dir().join("cranamp").join("player.conf")
}

#[cfg(target_arch = "wasm32")]
const PLAYER_STORAGE_KEY: &str = "cranamp.player";

#[cfg(target_arch = "wasm32")]
fn browser_storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}

#[cfg(target_arch = "wasm32")]
fn js_storage_error(value: wasm_bindgen::JsValue) -> String {
    value
        .as_string()
        .unwrap_or_else(|| "browser localStorage operation failed".to_string())
}

const WINAMP_NATIVE_HOST_OFFSET_X: f32 = 640.0;
const WINAMP_NATIVE_HOST_OFFSET_Y: f32 = 118.0;
const WINAMP_ATTACH_EPSILON: f32 = 3.0;
const WINAMP_SNAP_DISTANCE: f32 = 8.0;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct SavedWindowConfig {
    position: Option<Point>,
    size: Option<Size>,
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct SavedWinampWindowConfig {
    main: SavedWindowConfig,
    equalizer: SavedWindowConfig,
    playlist: SavedWindowConfig,
}

#[cfg(not(target_arch = "wasm32"))]
impl SavedWinampWindowConfig {
    fn from_states(peer_windows: WinampPeerWindowStates) -> Self {
        Self {
            main: SavedWindowConfig {
                position: peer_windows.main.position_non_reactive(),
                size: Some(peer_windows.main.size_non_reactive()),
            },
            equalizer: SavedWindowConfig {
                position: peer_windows.equalizer.position_non_reactive(),
                size: Some(peer_windows.equalizer.size_non_reactive()),
            },
            playlist: SavedWindowConfig {
                position: peer_windows.playlist.position_non_reactive(),
                size: Some(peer_windows.playlist.size_non_reactive()),
            },
        }
    }
}

#[composable]
fn remember_saved_window_config(peer_windows: WinampPeerWindowStates) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        cranpose_core::remember(move || {
            if let Some(config) = load_saved_window_config() {
                apply_saved_window_config(peer_windows, config);
            }
        });
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = peer_windows;
    }
}

#[composable]
fn NativeWindowPersistence(peer_windows: WinampPeerWindowStates) {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let last_saved = cranpose_core::remember(|| None::<SavedWinampWindowConfig>);
        cranpose_core::SideEffect(move || {
            let config = SavedWinampWindowConfig::from_states(peer_windows);
            last_saved.update(|last| {
                if last.as_ref() != Some(&config) {
                    save_window_config_background(config);
                    *last = Some(config);
                }
            });
        });
    }

    #[cfg(target_arch = "wasm32")]
    {
        let _ = peer_windows;
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn apply_saved_window_config(
    peer_windows: WinampPeerWindowStates,
    config: SavedWinampWindowConfig,
) {
    if let Some(position) = config.main.position {
        peer_windows.main.set_position(Some(position));
    }
    if let Some(position) = config.equalizer.position {
        peer_windows.equalizer.set_position(Some(position));
    }
    if let Some(position) = config.playlist.position {
        peer_windows.playlist.set_position(Some(position));
    }
    if let Some(size) = config.playlist.size {
        peer_windows
            .playlist
            .set_size(clamp_playlist_window_size(size));
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn clamp_playlist_window_size(size: Size) -> Size {
    Size::new(
        size.width.max(PLAYLIST_WIDTH),
        size.height.max(PLAYLIST_HEIGHT),
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn serialize_window_config(config: SavedWinampWindowConfig) -> String {
    let mut lines = Vec::new();
    push_window_config_lines(&mut lines, "main", config.main);
    push_window_config_lines(&mut lines, "equalizer", config.equalizer);
    push_window_config_lines(&mut lines, "playlist", config.playlist);
    lines.join("\n") + "\n"
}

#[cfg(not(target_arch = "wasm32"))]
fn push_window_config_lines(lines: &mut Vec<String>, name: &str, config: SavedWindowConfig) {
    if let Some(position) = config.position {
        lines.push(format!("{name}.x={:.3}", position.x));
        lines.push(format!("{name}.y={:.3}", position.y));
    }
    if let Some(size) = config.size {
        lines.push(format!("{name}.width={:.3}", size.width));
        lines.push(format!("{name}.height={:.3}", size.height));
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn parse_window_config(input: &str) -> SavedWinampWindowConfig {
    let mut config = SavedWinampWindowConfig::default();
    for line in input.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let Ok(value) = value.trim().parse::<f32>() else {
            continue;
        };
        apply_window_config_value(&mut config, key.trim(), value);
    }
    config
}

#[cfg(not(target_arch = "wasm32"))]
fn apply_window_config_value(config: &mut SavedWinampWindowConfig, key: &str, value: f32) {
    let Some((window, field)) = key.split_once('.') else {
        return;
    };
    let Some(target) = saved_window_mut(config, window) else {
        return;
    };

    match field {
        "x" => target.position.get_or_insert(Point::ZERO).x = value,
        "y" => target.position.get_or_insert(Point::ZERO).y = value,
        "width" => target.size.get_or_insert(Size::ZERO).width = value,
        "height" => target.size.get_or_insert(Size::ZERO).height = value,
        _ => {}
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn saved_window_mut<'a>(
    config: &'a mut SavedWinampWindowConfig,
    window: &str,
) -> Option<&'a mut SavedWindowConfig> {
    match window {
        "main" => Some(&mut config.main),
        "equalizer" => Some(&mut config.equalizer),
        "playlist" => Some(&mut config.playlist),
        _ => None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn load_saved_window_config() -> Option<SavedWinampWindowConfig> {
    std::fs::read_to_string(window_config_path())
        .ok()
        .map(|text| parse_window_config(&text))
}

#[cfg(not(target_arch = "wasm32"))]
fn save_window_config(config: SavedWinampWindowConfig) -> Result<(), String> {
    let path = window_config_path();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("failed to create {}: {error}", parent.display()))?;
    }
    std::fs::write(&path, serialize_window_config(config))
        .map_err(|error| format!("failed to write {}: {error}", path.display()))
}

#[cfg(not(target_arch = "wasm32"))]
fn save_window_config_background(config: SavedWinampWindowConfig) {
    std::thread::spawn(move || {
        let _ = save_window_config(config);
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn window_config_path() -> std::path::PathBuf {
    config_home_dir().join("cranamp").join("windows.conf")
}

#[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
fn config_home_dir() -> std::path::PathBuf {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| {
            std::env::var_os("HOME")
                .map(std::path::PathBuf::from)
                .map(|home| home.join(".config"))
        })
        .unwrap_or_else(|| std::path::PathBuf::from("."))
}

#[cfg(target_os = "android")]
fn config_home_dir() -> std::path::PathBuf {
    android_bridge::config_dir().unwrap_or_else(|| std::path::PathBuf::from("."))
}

fn default_main_position() -> Point {
    WINAMP_DEFAULT_SCREEN_POSITION
}

fn default_equalizer_position() -> Point {
    Point::new(
        WINAMP_DEFAULT_SCREEN_POSITION.x,
        WINAMP_DEFAULT_SCREEN_POSITION.y + MAIN_HEIGHT,
    )
}

fn default_playlist_position() -> Point {
    Point::new(
        WINAMP_DEFAULT_SCREEN_POSITION.x,
        WINAMP_DEFAULT_SCREEN_POSITION.y + MAIN_HEIGHT + EQ_HEIGHT,
    )
}

fn native_winamp_windows_available() -> bool {
    #[cfg(all(
        not(target_arch = "wasm32"),
        not(target_os = "android"),
        not(target_os = "ios")
    ))]
    {
        std::env::var_os("CRANPOSE_WINAMP_INLINE").is_none()
    }

    #[cfg(any(target_arch = "wasm32", target_os = "android", target_os = "ios"))]
    {
        false
    }
}

fn base_winamp_window_config(placement: WinampWindowPlacement) -> WindowConfig {
    let state_size = placement.state.size();
    let config = WindowConfig::borderless(placement.title, state_size.width, state_size.height);
    let config = match placement.initial_position {
        WinampInitialWindowPosition::Host(position) => config.with_host_window_position(
            snap_to_pixel(position.x + WINAMP_NATIVE_HOST_OFFSET_X),
            snap_to_pixel(position.y + WINAMP_NATIVE_HOST_OFFSET_Y),
        ),
        WinampInitialWindowPosition::Screen(position) => {
            config.with_position(snap_to_pixel(position.x), snap_to_pixel(position.y))
        }
    };
    config
        .with_transparent(false)
        .with_resizable(false)
        .with_visible(true)
}

fn winamp_window_config(placement: WinampWindowPlacement) -> WindowConfig {
    let state = placement.state;
    base_winamp_window_config(placement).with_state(state)
}

fn winamp_attach_policy() -> WindowAttachPolicy {
    WindowAttachPolicy::new(
        WINAMP_SNAP_DISTANCE,
        WINAMP_ATTACH_EPSILON,
        WindowMoveMode::DragLeaderOnly(vec![winamp_main_window_id()]),
    )
}

fn winamp_main_window_id() -> WindowId {
    WindowId::from_static("cranamp-winamp-main")
}

fn winamp_equalizer_window_id() -> WindowId {
    WindowId::from_static("cranamp-winamp-equalizer")
}

fn winamp_playlist_window_id() -> WindowId {
    WindowId::from_static("cranamp-winamp-playlist")
}

fn winamp_window_modifier(
    width: f32,
    height: f32,
    scale: f32,
    drag_target: WinampDragTarget,
) -> Modifier {
    let modifier = Modifier::empty().size_points(scaled(width, scale), scaled(height, scale));
    match drag_target {
        WinampDragTarget::Inline(position) => {
            let position = position.get();
            modifier.offset(snap_to_pixel(position.x), snap_to_pixel(position.y))
        }
        #[cfg(any(target_os = "android", all(feature = "web", target_arch = "wasm32")))]
        WinampDragTarget::Fixed(position) => {
            modifier.offset(scaled(position.x, scale), scaled(position.y, scale))
        }
        WinampDragTarget::NativeGroup => modifier,
    }
}

fn ui_scale() -> f32 {
    // Skin pixel coordinates map directly to dp.  On high-density screens the
    // renderer upscales automatically, keeping the skin at the same visual
    // size as on a 1× desktop display.
    1.0
}

fn snap_to_pixel(value: f32) -> f32 {
    let density = current_density();
    if density > 0.0 {
        (value * density).round() / density
    } else {
        value.round()
    }
}

fn scaled(value: f32, scale: f32) -> f32 {
    snap_to_pixel(value * scale)
}

fn clamp01(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn slider_thumb_x(value: f32, bar_width: f32, knob_width: f32) -> f32 {
    clamp01(value) * (bar_width - knob_width)
}

fn slider_frame(value: f32, frames: u32) -> u32 {
    if frames <= 1 {
        return 0;
    }
    let max_index = frames - 1;
    (clamp01(value) * max_index as f32).round() as u32
}

fn eq_slider_bg_rect(value: f32) -> SpriteRect {
    let frame = slider_frame(value, EQ_SLIDER_BG_FRAMES);
    let row = (frame / EQ_SLIDER_BG_FRAMES_PER_ROW) as usize;
    let column = frame % EQ_SLIDER_BG_FRAMES_PER_ROW;
    (
        EQ_SLIDER_BG.0 + EQ_SLIDER_BG_STRIDE * column as f32,
        EQ_SLIDER_BG_ROW_Y[row.min(EQ_SLIDER_BG_ROW_Y.len() - 1)],
        EQ_SLIDER_BG.2,
        EQ_SLIDER_BG.3,
    )
}

fn vertical_slider_thumb_y(value: f32, track_height: f32, knob_height: f32) -> f32 {
    (1.0 - clamp01(value)) * (track_height - knob_height)
}

fn vertical_slider_thumb_y_down(value: f32, track_height: f32, knob_height: f32) -> f32 {
    clamp01(value) * (track_height - knob_height)
}

fn time_digits(elapsed_seconds: f32) -> [u8; 4] {
    let seconds = elapsed_seconds.max(0.0).round() as u32;
    let minutes = (seconds / 60) % 100;
    let remainder = seconds % 60;
    [
        ((minutes / 10) % 10) as u8,
        (minutes % 10) as u8,
        (remainder / 10) as u8,
        (remainder % 10) as u8,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_playlist(tracks: Vec<Track>) -> Rc<Vec<Track>> {
        Rc::new(tracks)
    }

    #[test]
    fn time_digits_are_mapped_correctly() {
        assert_eq!(time_digits(0.0), [0, 0, 0, 0]);
        assert_eq!(time_digits(65.0), [0, 1, 0, 5]);
        assert_eq!(time_digits(-1.0), [0, 0, 0, 0]);
    }

    #[test]
    fn slider_helpers_clamp_values() {
        assert_eq!(slider_frame(-1.0, 28), 0);
        assert_eq!(slider_frame(2.0, 28), 27);
        assert_eq!(slider_thumb_x(-1.0, 248.0, 29.0), 0.0);
        assert_eq!(slider_thumb_x(2.0, 248.0, 29.0), 219.0);
    }

    #[test]
    fn eq_slider_background_uses_skin_frame_for_value() {
        assert_eq!(eq_slider_bg_rect(0.0), (13.0, 164.0, 14.0, 63.0));
        assert_eq!(eq_slider_bg_rect(0.5), (13.0, 229.0, 14.0, 63.0));
        assert_eq!(eq_slider_bg_rect(1.0), (208.0, 229.0, 14.0, 63.0));
    }

    #[test]
    fn playlist_row_line_text_fits_title_and_duration() {
        let ten_chars = WINAMP_PLAYLIST_ROW_CHAR_WIDTH * 10.0;
        assert_eq!(
            playlist_row_line_text("TrackTitle".to_string(), Some("1:23"), ten_chars),
            "Track 1:23"
        );
        assert_eq!(
            playlist_row_line_text("LongTrackName".to_string(), None, ten_chars * 0.5),
            "LongT"
        );
    }

    #[test]
    fn playlist_visible_text_blanks_current_overlay_row() {
        let tracks = vec![test_track("One"), test_track("Two"), test_track("Three")];
        let text = build_playlist_visible_text(
            &tracks,
            0,
            3,
            Some(1),
            WINAMP_PLAYLIST_ROW_CHAR_WIDTH * 12.0,
        );

        assert_eq!(text.lines().collect::<Vec<_>>(), vec!["One", "", "Three"]);
    }

    #[test]
    fn playlist_visible_row_capacity_uses_full_list_area() {
        let default_list_height =
            PLAYLIST_HEIGHT - PLAYLIST_BOTTOM_LEFT_CORNER.3 - PLAYLIST_LIST_BG.1;

        assert_eq!(playlist_visible_row_capacity(default_list_height), 18);
    }

    #[test]
    fn main_display_meta_does_not_include_bitrate_units() {
        let state = WinampState {
            playlist: test_playlist(vec![test_track("One")]),
            current_index: Some(0),
            ..WinampState::default()
        };
        let meta = main_display_meta(&state);

        assert!(!meta.contains("kbps"));
        assert!(!meta.contains("khz"));
    }

    #[test]
    fn vertical_slider_helpers_clamp_values() {
        assert_eq!(vertical_slider_thumb_y(-1.0, 63.0, 11.0), 52.0);
        assert_eq!(vertical_slider_thumb_y(2.0, 63.0, 11.0), 0.0);
        assert_eq!(vertical_slider_thumb_y_down(-1.0, 145.0, 18.0), 0.0);
        assert_eq!(vertical_slider_thumb_y_down(2.0, 145.0, 18.0), 127.0);
    }

    #[test]
    fn progress_fraction_uses_duration_when_known() {
        assert_eq!(progress_fraction(30.0, Some(120.0)), 0.25);
        assert_eq!(progress_fraction(130.0, Some(120.0)), 1.0);
        assert_eq!(progress_fraction(30.0, None), 0.0);
    }

    #[test]
    fn playlist_scroll_tracks_current_index() {
        assert_eq!(playlist_scroll_for_track(3, 20, 5, 0.0), 0.0);
        assert_eq!(playlist_scroll_for_track(10, 20, 5, 0.5), 0.5);
        assert!((playlist_scroll_for_track(10, 20, 5, 0.0) - (8.0 / 15.0)).abs() < f32::EPSILON);
        assert_eq!(playlist_scroll_for_track(99, 20, 5, 0.0), 1.0);
        assert_eq!(playlist_scroll_for_track(0, 1, 5, 0.5), 0.0);
    }

    #[test]
    fn marquee_text_ping_pongs_long_titles() {
        let title = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".to_string();
        let width = WINAMP_SYSTEM_MARQUEE_CHAR_WIDTH * 14.0;

        assert!(marquee_system_text(title.clone(), width, 0.0).starts_with("ABCDEFGHIJKLMN"));
        assert!(marquee_system_text(title.clone(), width, 2.0).starts_with("CDEFGHIJKLMNOP"));
        assert!(marquee_system_text(title, width, 18.0).starts_with("GHIJKLMNOPQRST"));
        assert_eq!(
            marquee_system_text("SHORT".to_string(), width, 60.0),
            "SHORT"
        );
    }

    #[test]
    fn visualizer_bitmap_contains_bright_bars() {
        let bitmap = visualizer_bitmap(
            true,
            [0.8; audio::VISUALIZER_BAND_COUNT],
            VisColor::default(),
        );

        assert_eq!(bitmap.width(), VISUALIZER_WIDTH as u32);
        assert_eq!(bitmap.height(), VISUALIZER_HEIGHT as u32);
        assert!(bitmap
            .pixels()
            .chunks_exact(4)
            .any(|pixel| pixel[1] > 180 && pixel[3] == 255));
    }

    #[test]
    fn visualizer_bitmap_is_blank_when_stopped() {
        let bitmap = visualizer_bitmap(
            false,
            [0.8; audio::VISUALIZER_BAND_COUNT],
            VisColor::default(),
        );

        assert!(bitmap
            .pixels()
            .chunks_exact(4)
            .all(|pixel| pixel == [0, 0, 0, 255]));
    }

    #[test]
    fn visualizer_uses_viscolor_palette_for_lit_bars() {
        let palette = VisColor([[5; 4]; 24]);
        let bitmap = visualizer_bitmap(true, [1.0; audio::VISUALIZER_BAND_COUNT], palette);
        assert!(bitmap
            .pixels()
            .chunks_exact(4)
            .all(|pixel| pixel == [5, 5, 5, 5]));
    }

    #[test]
    fn player_state_config_round_trips_settings_and_playlist() {
        let mut eq_values = [0.5; 11];
        eq_values[0] = 0.1;
        eq_values[10] = 0.9;
        let state = WinampState {
            shuffle: true,
            repeat: true,
            eq_visible: false,
            playlist_visible: false,
            eq_enabled: false,
            eq_auto: true,
            eq_values,
            skin_path: Some("/tmp/Cranamp Skin.wsz".to_string()),
            playlist_scroll: 0.42,
            volume: 0.33,
            balance: 0.66,
            playlist: test_playlist(vec![test_track("One=Track"), test_track("Two\tTrack")]),
            current_index: Some(1),
            ..WinampState::default()
        };

        let saved = SavedPlayerState::from_state(&state);
        let parsed = parse_player_state(&serialize_player_state(&saved));

        assert_eq!(parsed, saved);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn restored_player_state_filters_missing_tracks_and_remaps_current_index() {
        let fixture_path =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tone-video.mp4");
        let saved = SavedPlayerState {
            volume: 0.25,
            current_index: Some(1),
            skin_path: Some("/tmp/cranamp-definitely-missing.wsz".to_string()),
            tracks: vec![
                SavedTrack {
                    title: "Missing".to_string(),
                    path: "/tmp/cranamp-definitely-missing.mp3".to_string(),
                },
                SavedTrack {
                    title: "Present".to_string(),
                    path: fixture_path.to_string_lossy().to_string(),
                },
            ],
            ..SavedPlayerState::default()
        };

        let state = restore_saved_player_state(saved);

        assert_eq!(state.playlist.len(), 1);
        assert_eq!(state.playlist[0].title, "Present");
        assert_eq!(state.current_index, Some(0));
        assert_eq!(state.volume, 0.25);
        assert_eq!(state.skin_path, None);
    }

    #[test]
    fn elapsed_time_is_clamped_to_track_duration() {
        assert_eq!(normalized_elapsed_seconds(130.0, Some(120.0)), 120.0);
        assert_eq!(normalized_elapsed_seconds(30.0, Some(120.0)), 30.0);
    }

    #[test]
    fn saved_window_config_round_trips() {
        let config = SavedWinampWindowConfig {
            main: SavedWindowConfig {
                position: Some(Point::new(10.0, 20.0)),
                size: Some(Size::new(MAIN_WIDTH, MAIN_HEIGHT)),
            },
            equalizer: SavedWindowConfig {
                position: Some(Point::new(10.0, 136.0)),
                size: Some(Size::new(EQ_WIDTH, EQ_HEIGHT)),
            },
            playlist: SavedWindowConfig {
                position: Some(Point::new(10.0, 252.0)),
                size: Some(Size::new(320.0, 240.0)),
            },
        };

        assert_eq!(
            parse_window_config(&serialize_window_config(config)),
            config
        );
    }

    #[test]
    fn default_native_windows_stack_vertically() {
        assert_eq!(default_main_position(), Point::new(140.0, 120.0));
        assert_eq!(default_equalizer_position(), Point::new(140.0, 236.0));
        assert_eq!(default_playlist_position(), Point::new(140.0, 352.0));
    }

    #[test]
    fn append_playlist_tracks_starts_empty_playlist_at_first_added_track() {
        let mut state = WinampState::default();

        let should_start = append_playlist_tracks(&mut state, vec![test_track("First")]);

        assert!(should_start);
        assert_eq!(state.playlist.as_slice(), &[test_track("First")]);
        assert_eq!(state.current_index, Some(0));
        assert_eq!(state.playlist_scroll, 0.0);
        assert_eq!(state.position, 0.0);
        assert_eq!(state.elapsed_seconds, 0.0);
        assert_eq!(state.status, "Loaded 1 Track(s)");
    }

    #[test]
    fn append_playlist_tracks_preserves_current_track_when_playlist_exists() {
        let mut state = WinampState {
            playback: PlaybackState::Playing,
            playlist: test_playlist(vec![test_track("First")]),
            current_index: Some(0),
            playlist_scroll: 0.25,
            status: "Playing First".to_string(),
            ..WinampState::default()
        };

        let should_start = append_playlist_tracks(&mut state, vec![test_track("Second")]);

        assert!(!should_start);
        assert_eq!(
            state.playlist.as_slice(),
            &[test_track("First"), test_track("Second")]
        );
        assert_eq!(state.current_index, Some(0));
        assert_eq!(state.playback, PlaybackState::Playing);
        assert_eq!(state.playlist_scroll, 0.25);
        assert_eq!(state.status, "Added 1 Track(s)");
    }

    #[test]
    fn playlist_single_click_selects_without_playing() {
        let mut state = WinampState {
            playlist: test_playlist(vec![test_track("First"), test_track("Second")]),
            current_index: Some(0),
            selected_indices: vec![0],
            selection_anchor: Some(0),
            ..WinampState::default()
        };

        let should_play = handle_playlist_row_click_in_state(
            &mut state,
            1,
            1000,
            PlaylistClickModifiers::default(),
        );

        assert!(!should_play);
        assert_eq!(state.current_index, Some(0));
        assert_eq!(state.playback, PlaybackState::Stopped);
        assert_eq!(state.selected_indices, vec![1]);
        assert_eq!(state.selection_anchor, Some(1));
        assert_eq!(state.playlist_last_click_index, Some(1));
    }

    #[test]
    fn playlist_second_plain_click_requests_play() {
        let mut state = WinampState {
            playlist: test_playlist(vec![test_track("First"), test_track("Second")]),
            playlist_last_click_index: Some(1),
            playlist_last_click_ms: 1000,
            ..WinampState::default()
        };

        let should_play = handle_playlist_row_click_in_state(
            &mut state,
            1,
            1200,
            PlaylistClickModifiers::default(),
        );

        assert!(should_play);
        assert_eq!(state.playlist_last_click_index, None);
        assert_eq!(state.playlist_last_click_ms, 0);
    }

    #[test]
    fn playlist_shift_and_ctrl_click_match_winamp_selection_rules() {
        let mut state = WinampState {
            playlist: test_playlist(vec![
                test_track("First"),
                test_track("Second"),
                test_track("Third"),
                test_track("Fourth"),
            ]),
            selected_indices: vec![1],
            selection_anchor: Some(1),
            ..WinampState::default()
        };

        select_playlist_row_in_state(
            &mut state,
            3,
            PlaylistClickModifiers {
                shift: true,
                ctrl: false,
            },
        );
        assert_eq!(state.selected_indices, vec![1, 2, 3]);
        assert_eq!(state.selection_anchor, Some(1));

        select_playlist_row_in_state(
            &mut state,
            2,
            PlaylistClickModifiers {
                shift: false,
                ctrl: true,
            },
        );
        assert_eq!(state.selected_indices, vec![1, 3]);
        assert_eq!(state.selection_anchor, Some(1));

        select_playlist_row_in_state(
            &mut state,
            0,
            PlaylistClickModifiers {
                shift: false,
                ctrl: true,
            },
        );
        assert_eq!(state.selected_indices, vec![0, 1, 3]);
        assert_eq!(state.selection_anchor, Some(0));
    }

    #[test]
    fn remove_playlist_track_at_stops_removed_current_and_keeps_next_selected() {
        let mut state = WinampState {
            playback: PlaybackState::Playing,
            playlist: test_playlist(vec![
                test_track("First"),
                test_track("Second"),
                test_track("Third"),
            ]),
            current_index: Some(1),
            position: 0.5,
            elapsed_seconds: 12.0,
            duration_seconds: Some(120.0),
            ..WinampState::default()
        };

        assert!(remove_playlist_track_at(&mut state, 1));

        assert_eq!(
            state.playlist.as_slice(),
            &[test_track("First"), test_track("Third")]
        );
        assert_eq!(state.current_index, Some(1));
        assert_eq!(state.playback, PlaybackState::Stopped);
        assert_eq!(state.position, 0.0);
        assert_eq!(state.elapsed_seconds, 0.0);
        assert_eq!(state.duration_seconds, None);
        assert_eq!(state.status, "Removed Track 2");
    }

    #[test]
    fn sort_playlist_tracks_by_title_preserves_current_track() {
        let mut state = WinampState {
            playlist: test_playlist(vec![
                test_track("Bravo"),
                test_track("Alpha"),
                test_track("Charlie"),
            ]),
            current_index: Some(0),
            selected_indices: vec![0, 2],
            selection_anchor: Some(2),
            ..WinampState::default()
        };

        assert!(sort_playlist_tracks_by_title(&mut state));

        assert_eq!(
            state.playlist.as_slice(),
            &[
                test_track("Alpha"),
                test_track("Bravo"),
                test_track("Charlie")
            ]
        );
        assert_eq!(state.current_index, Some(1));
        assert_eq!(state.selected_indices, vec![1, 2]);
        assert_eq!(state.selection_anchor, Some(2));
        assert_eq!(state.status, "Playlist Sorted");
    }

    #[test]
    fn remove_playlist_indices_remaps_current_and_selection() {
        let mut state = WinampState {
            playback: PlaybackState::Playing,
            playlist: test_playlist(vec![
                test_track("First"),
                test_track("Second"),
                test_track("Third"),
                test_track("Fourth"),
            ]),
            current_index: Some(2),
            selected_indices: vec![1, 3],
            selection_anchor: Some(3),
            position: 0.5,
            elapsed_seconds: 12.0,
            duration_seconds: Some(120.0),
            ..WinampState::default()
        };

        assert_eq!(remove_playlist_indices(&mut state, &[1, 3]), 2);

        assert_eq!(
            state.playlist.as_slice(),
            &[test_track("First"), test_track("Third")]
        );
        assert_eq!(state.current_index, Some(1));
        assert_eq!(state.selected_indices, vec![1]);
        assert_eq!(state.playback, PlaybackState::Playing);
    }

    #[test]
    fn remove_playlist_indices_stops_when_current_is_removed() {
        let mut state = WinampState {
            playback: PlaybackState::Playing,
            playlist: test_playlist(vec![test_track("First"), test_track("Second")]),
            current_index: Some(0),
            selected_indices: vec![0],
            position: 0.5,
            elapsed_seconds: 12.0,
            duration_seconds: Some(120.0),
            ..WinampState::default()
        };

        assert_eq!(remove_playlist_indices(&mut state, &[0]), 1);

        assert_eq!(state.playlist.as_slice(), &[test_track("Second")]);
        assert_eq!(state.current_index, Some(0));
        assert_eq!(state.selected_indices, vec![0]);
        assert_eq!(state.playback, PlaybackState::Stopped);
        assert_eq!(state.position, 0.0);
        assert_eq!(state.elapsed_seconds, 0.0);
        assert_eq!(state.duration_seconds, None);
    }

    #[test]
    fn duplicate_playlist_indices_uses_path_when_available() {
        let playlist = vec![
            test_track_with_path("First", "/tmp/one.mp3"),
            test_track_with_path("Copy", "/tmp/one.mp3"),
            test_track_with_path("Other", "/tmp/two.mp3"),
        ];

        assert_eq!(duplicate_playlist_indices(&playlist), vec![1]);
    }

    #[test]
    fn select_search_query_uses_current_artist_prefix() {
        let state = WinampState {
            playlist: test_playlist(vec![
                test_track("Celldweller - Eon"),
                test_track("Celldweller - One Good Reason"),
            ]),
            current_index: Some(1),
            ..WinampState::default()
        };

        assert_eq!(
            playlist_search_query(&state),
            Some("Celldweller".to_string())
        );
    }

    #[test]
    fn playlist_search_filter_selects_matching_title_or_path() {
        let mut state = WinampState {
            playlist: test_playlist(vec![
                test_track_with_path("Blue October - Somebody", "/music/blue.mp3"),
                test_track_with_path("Celldweller - Eon", "/music/eon.flac"),
                test_track_with_path("Other", "/music/celldweller-live.ogg"),
            ]),
            ..WinampState::default()
        };

        apply_playlist_search_filter_in_state(&mut state, "celldweller");

        assert_eq!(state.selected_indices, vec![1, 2]);
        assert_eq!(state.selection_anchor, Some(2));
    }

    #[test]
    fn parse_m3u_playlist_accepts_plain_paths_and_extinf() {
        let input = "#EXTM3U\n#EXTINF:195,Broods - Heartlines\nrelative/song.mp3\n/home/s/Music/Other.flac\n";
        let tracks = parse_m3u_playlist(input, Some(std::path::Path::new("/tmp/list")));

        assert_eq!(tracks.len(), 2);
        assert_eq!(tracks[0].title, "Broods - Heartlines");
        assert_eq!(tracks[0].duration_seconds, Some(195.0));
        assert_eq!(
            tracks[0].path.as_deref(),
            Some("/tmp/list/relative/song.mp3")
        );
        assert_eq!(tracks[1].title, "Other");
        assert_eq!(tracks[1].path.as_deref(), Some("/home/s/Music/Other.flac"));
    }

    #[test]
    fn format_m3u_playlist_writes_extm3u_and_durations() {
        let playlist = vec![Track {
            title: "Broods - Heartlines".to_string(),
            path: Some("/home/s/Music/Broods - Heartlines.mp3".to_string()),
            duration_seconds: Some(199.0),
        }];

        let text = format_m3u_playlist(&playlist);

        assert!(text.starts_with("#EXTM3U\n"));
        assert!(text.contains("#EXTINF:199,Broods - Heartlines\n"));
        assert!(text.contains("/home/s/Music/Broods - Heartlines.mp3\n"));
    }

    #[test]
    fn scroll_playlist_by_rows_clamps_to_available_rows() {
        let mut state = WinampState {
            playlist: test_playlist(
                (0..20)
                    .map(|index| test_track(&format!("Track {index:02}")))
                    .collect(),
            ),
            playlist_visible_rows: 5,
            ..WinampState::default()
        };

        scroll_playlist_by_rows_in_state(&mut state, 3);
        assert!((state.playlist_scroll - (3.0 / 15.0)).abs() < f32::EPSILON);

        scroll_playlist_by_rows_in_state(&mut state, 99);
        assert_eq!(state.playlist_scroll, 1.0);

        scroll_playlist_by_rows_in_state(&mut state, -99);
        assert_eq!(state.playlist_scroll, 0.0);
    }

    #[test]
    fn automatic_advance_stops_at_playlist_end_without_repeat() {
        let state = WinampState {
            playlist: test_playlist(vec![test_track("First"), test_track("Second")]),
            current_index: Some(1),
            ..WinampState::default()
        };

        let plan = playlist_advance_plan(&state, TrackDirection::Next, TrackAdvanceMode::Automatic);

        assert_eq!(plan, None);
    }

    #[test]
    fn automatic_advance_wraps_at_playlist_end_with_repeat() {
        let state = WinampState {
            repeat: true,
            playlist: test_playlist(vec![test_track("First"), test_track("Second")]),
            current_index: Some(1),
            ..WinampState::default()
        };

        let plan = playlist_advance_plan(&state, TrackDirection::Next, TrackAdvanceMode::Automatic);

        assert_eq!(
            plan,
            Some(TrackAdvancePlan {
                index: 0,
                shuffle_order: None,
            })
        );
    }

    #[test]
    fn manual_advance_wraps_without_repeat() {
        let state = WinampState {
            playlist: test_playlist(vec![test_track("First"), test_track("Second")]),
            current_index: Some(1),
            ..WinampState::default()
        };

        let next = playlist_advance_plan(&state, TrackDirection::Next, TrackAdvanceMode::Manual);
        let previous =
            playlist_advance_plan(&state, TrackDirection::Previous, TrackAdvanceMode::Manual);

        assert_eq!(
            next,
            Some(TrackAdvancePlan {
                index: 0,
                shuffle_order: None,
            })
        );
        assert_eq!(
            previous,
            Some(TrackAdvancePlan {
                index: 0,
                shuffle_order: None,
            })
        );
    }

    #[test]
    fn shuffle_advance_follows_the_existing_order() {
        let state = WinampState {
            shuffle: true,
            playlist: test_playlist(vec![
                test_track("First"),
                test_track("Second"),
                test_track("Third"),
            ]),
            current_index: Some(0),
            shuffle_order: vec![0, 2, 1],
            ..WinampState::default()
        };

        let plan = playlist_advance_plan(&state, TrackDirection::Next, TrackAdvanceMode::Manual);

        assert_eq!(
            plan,
            Some(TrackAdvancePlan {
                index: 2,
                shuffle_order: None,
            })
        );
    }

    #[test]
    fn shuffle_repeat_rebuilds_order_after_exhaustion() {
        let state = WinampState {
            shuffle: true,
            repeat: true,
            playlist: test_playlist(vec![
                test_track("First"),
                test_track("Second"),
                test_track("Third"),
            ]),
            current_index: Some(2),
            shuffle_order: vec![0, 1, 2],
            ..WinampState::default()
        };

        let plan = playlist_advance_plan(&state, TrackDirection::Next, TrackAdvanceMode::Automatic)
            .expect("repeat should continue the playlist");
        let replacement = plan
            .shuffle_order
            .as_ref()
            .expect("shuffle should rebuild order on repeat wrap");

        assert_ne!(plan.index, 2);
        assert_eq!(replacement.first(), Some(&2));
        assert_eq!(replacement.get(1), Some(&plan.index));
        assert!(valid_shuffle_order(replacement, state.playlist.len()));
    }

    #[test]
    fn seeded_shuffle_order_keeps_current_track_first() {
        let order = shuffled_order_with_seed(4, 2, 42);

        assert_eq!(order.first(), Some(&2));
        assert!(valid_shuffle_order(&order, 4));
    }

    fn test_track(title: &str) -> Track {
        Track {
            title: title.to_string(),
            path: Some(format!("/tmp/{title}.mp3")),
            duration_seconds: None,
        }
    }

    fn test_track_with_path(title: &str, path: &str) -> Track {
        Track {
            title: title.to_string(),
            path: Some(path.to_string()),
            duration_seconds: None,
        }
    }
}
