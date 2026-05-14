# Roadmap

## Phase 1: Standalone MVP

- [x] Create Rust/Cranpose project structure.
- [x] Extract classic Winamp `.wsz` skin loading and sprite rendering from the Cranpose demo.
- [x] Run desktop as standalone native Winamp windows.
- [x] Run Android/iOS as fullscreen Cranpose surfaces.
- [x] Run WebAssembly as an embeddable canvas widget.
- [x] Add native desktop file and folder pickers.
- [x] Add Rodio playback for desktop audio files.
- [x] Add browser file picking and HTML audio playback for the wasm widget.
- [x] Forbid unsafe application code.
- [x] Add CI, tag release, and GitHub Pages workflows.

## Phase 2: Mobile Packaging

- [x] Add Android Gradle project and debug-signed APK release packaging.
- [x] Attach iOS device/simulator static library packages to tag releases.
- [x] Implement Android Storage Access Framework file/folder picker.
- [ ] Add signed Android APK/AAB release packaging.
- [ ] Add iOS Xcode project and archive/export workflow.
- [ ] Implement iOS document picker integration.
- [ ] Persist playlist and skin selection across launches.
- [ ] Implement Android system overlay mini-player after Cranpose can host a
  service-owned overlay surface.

## Phase 3: Winamp Compatibility

- [ ] Load user-selected `.wsz` skins at runtime.
- [ ] Parse classic playlist formats (`.m3u`, `.pls`).
- [ ] Add accurate track duration, seek position, and end-of-track advance.
- [ ] Implement shuffle order and repeat-one/repeat-all semantics.
- [ ] Add equalizer DSP or disable EQ controls until backed by audio processing.

## Phase 4: Release Quality

- [ ] Add screenshot/robot checks for desktop and wasm.
- [ ] Add audio backend tests around playlist transitions.
- [ ] Add packaging metadata and icons per platform.
- [ ] Add crash/error reporting hooks appropriate for each target.
- [x] Add an experimental browser Document Picture-in-Picture Cranamp window for
  Chromium-based browsers.
