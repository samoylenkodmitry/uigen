# Floating Surfaces

Cranamp has three different "floating" targets. They should stay separate in
the code and product language because they have different platform semantics.

## Android System Overlay

This is the Android path for a real Winamp-style mini-player that remains
visible after the user leaves the app.

Expected shape:

- The full `CranampActivity` remains the normal app experience.
- A user action enables the floating mini-player.
- Cranamp checks `Settings.canDrawOverlays(context)`.
- If needed, Cranamp launches `Settings.ACTION_MANAGE_OVERLAY_PERMISSION`.
- An explicit overlay lifecycle starts after permission is granted.
- The overlay is added through `WindowManager.addView()`.
- The overlay uses `WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY`.
- The overlay defaults to `FLAG_NOT_FOCUSABLE` so other apps keep focus.
- The overlay surface uses `PixelFormat.TRANSLUCENT`.
- Non-interactive skin areas drag the overlay.
- A close affordance dismisses the overlay.
- Permission denial or revocation falls back cleanly to the normal Activity.

Current blocker: Cranpose's Android runtime renders into the launcher
`NativeActivity` `ANativeWindow`. It does not expose a supported way to render a
Cranpose root into a service-owned `Surface`, `SurfaceView`, `TextureView`, or
other `ANativeWindow` created by `WindowManager`. That upstream requirement is
tracked in `samoylenkodmitry/Cranpose#232`.

Until that exists, Cranamp should not pretend Android freeform mode is the real
overlay implementation. Any Cranamp-side Android overlay work should be limited
to permission/service scaffolding or a temporary native Android placeholder, not
the skinned Cranpose player.

## Android Freeform Activity

Freeform is still useful, but it is not an always-on-top overlay.

Freeform Activity mode:

- depends on device, launcher, OEM, desktop mode, developer options, or
  large-screen environment,
- behaves like a normal Activity task window,
- does not guarantee visibility above other apps,
- can be moved/resized by the system only where the system allows it, and
- remains useful as an optional desktop/tablet/debug fallback.

Cranamp's current Android implementation belongs in this bucket: one stacked
Cranpose surface in a resizeable Activity, with SAF file/folder/playlist
loading and export.

## Browser Document Picture-in-Picture

For the WebAssembly widget, Chromium's Document Picture-in-Picture API is an
experimental browser-native floating surface. It can host arbitrary HTML after a
user gesture, but the browser owns the outer window and behavior.

Expected shape:

- The main tab runs the full Cranamp web/WASM app and owns playback/state.
- A user gesture opens the floating Cranamp window with
  `documentPictureInPicture.requestWindow({ width, height })`.
- The existing Cranamp canvas is moved into the PiP document, preserving the
  live Cranpose renderer, player state, and pointer handlers.
- Closing the PiP window restores the same canvas to the embedded page.
- Unsupported browsers use the normal embedded widget.

Limitations:

- A user gesture is required.
- Support is Chromium-focused.
- It is not borderless, transparent, or native.
- Size, position, and chrome are browser/OS controlled.
- It is not equivalent to Tauri, Electron, or native always-on-top windows.

Cranamp currently implements this in `index.html` by reparenting the live
`cranamp-canvas` into the PiP document. This should not be modeled as a Cranpose
native window.

## Native Desktop

Desktop already uses Cranpose native peer windows for the Winamp-style shape.
The future desktop floating target is a native always-on-top/skinned window
policy through the desktop window APIs, not Android freeform and not browser
Document Picture-in-Picture.
