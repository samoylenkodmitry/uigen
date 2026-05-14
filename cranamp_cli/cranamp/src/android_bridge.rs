#![allow(unsafe_code)]

use std::path::PathBuf;
use std::sync::OnceLock;

use android_activity::AndroidApp;
use jni::errors::Result as JniResult;
use jni::objects::{JObject, JString};
use jni::refs::Global;
use jni::signature::RuntimeMethodSignature;
use jni::strings::JNIString;
use jni::sys::jobject;
use jni::vm::JavaVM;
use jni::{jni_sig, jni_str, JValue};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AndroidLoadMode {
    Replace,
    Append,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AndroidBridgeResult {
    AudioPaths {
        mode: AndroidLoadMode,
        paths: Vec<PathBuf>,
    },
    PlaylistImport {
        text: String,
    },
    PlaylistExport {
        target: String,
    },
    SkinImport {
        path: PathBuf,
    },
    Cancelled {
        operation: &'static str,
    },
    Error(String),
}

struct AndroidBridge {
    vm: JavaVM,
    activity: Global<JObject<'static>>,
    bridge_dir: PathBuf,
}

static BRIDGE: OnceLock<AndroidBridge> = OnceLock::new();

pub fn init(app: &AndroidApp) -> Result<(), String> {
    if BRIDGE.get().is_some() {
        return Ok(());
    }

    let vm = unsafe { JavaVM::from_raw(app.vm_as_ptr().cast()) };
    let (activity, bridge_dir) = vm
        .attach_current_thread(|env| -> JniResult<(Global<JObject<'static>>, String)> {
            let activity_obj = unsafe { JObject::from_raw(env, app.activity_as_ptr() as jobject) };
            let activity = env.new_global_ref(&activity_obj)?;
            let bridge_dir_obj = env
                .call_method(
                    activity.as_obj(),
                    jni_str!("cranampBridgeDirectory"),
                    jni_sig!("()Ljava/lang/String;"),
                    &[],
                )
                .and_then(|value| value.l())?;
            let bridge_dir = JString::cast_local(env, bridge_dir_obj)?.try_to_string(env)?;
            Ok((activity, bridge_dir))
        })
        .map_err(|error| format!("failed to initialize Android bridge: {error}"))?;

    let bridge = AndroidBridge {
        vm,
        activity,
        bridge_dir: PathBuf::from(bridge_dir),
    };
    let _ = BRIDGE.set(bridge);
    Ok(())
}

pub fn request_audio_files(mode: AndroidLoadMode) -> Result<(), String> {
    call_android_picker(
        "cranampPickAudioFiles",
        "(I)V",
        &[JValue::Int(mode_value(mode))],
    )
}

pub fn request_audio_folder(mode: AndroidLoadMode) -> Result<(), String> {
    call_android_picker(
        "cranampPickAudioFolder",
        "(I)V",
        &[JValue::Int(mode_value(mode))],
    )
}

pub fn request_playlist_import() -> Result<(), String> {
    call_android_picker("cranampImportPlaylist", "()V", &[])
}

pub fn request_skin_import() -> Result<(), String> {
    call_android_picker("cranampPickSkinFile", "()V", &[])
}

pub fn request_playlist_export(text: &str) -> Result<(), String> {
    let Some(bridge) = BRIDGE.get() else {
        return Err("Android activity bridge is not initialized".to_string());
    };
    bridge
        .vm
        .attach_current_thread(|env| -> JniResult<()> {
            let text = env.new_string(text)?;
            let text_obj: &JObject<'_> = text.as_ref();
            env.call_method(
                bridge.activity.as_obj(),
                jni_str!("cranampExportPlaylist"),
                jni_sig!("(Ljava/lang/String;)V"),
                &[JValue::Object(text_obj)],
            )?;
            Ok(())
        })
        .map_err(|error| format!("failed to launch Android playlist export: {error}"))?;
    Ok(())
}

pub fn take_results() -> Vec<AndroidBridgeResult> {
    let Some(bridge) = BRIDGE.get() else {
        return Vec::new();
    };

    let mut results = Vec::new();
    collect_audio_result(
        &bridge.bridge_dir,
        "audio_replace",
        AndroidLoadMode::Replace,
        &mut results,
    );
    collect_audio_result(
        &bridge.bridge_dir,
        "audio_append",
        AndroidLoadMode::Append,
        &mut results,
    );
    collect_playlist_import_result(&bridge.bridge_dir, &mut results);
    collect_playlist_export_result(&bridge.bridge_dir, &mut results);
    collect_skin_import_result(&bridge.bridge_dir, &mut results);
    results
}

pub fn config_dir() -> Option<PathBuf> {
    BRIDGE.get().map(|bridge| bridge.bridge_dir.join("config"))
}

pub fn start_window_move(local_x_dp: f32, local_y_dp: f32) -> bool {
    let Some(bridge) = BRIDGE.get() else {
        return false;
    };
    bridge
        .vm
        .attach_current_thread(|env| -> JniResult<bool> {
            env.call_method(
                bridge.activity.as_obj(),
                jni_str!("cranampStartWindowMove"),
                jni_sig!("(FF)Z"),
                &[JValue::Float(local_x_dp), JValue::Float(local_y_dp)],
            )
            .and_then(|value| value.z())
        })
        .unwrap_or(false)
}

pub fn content_top_inset_dp() -> f32 {
    activity_float_method("cranampContentTopInsetDp")
}

pub fn content_bottom_inset_dp() -> f32 {
    activity_float_method("cranampContentBottomInsetDp")
}

fn activity_float_method(method: &str) -> f32 {
    let Some(bridge) = BRIDGE.get() else {
        return 0.0;
    };
    bridge
        .vm
        .attach_current_thread(|env| -> JniResult<f32> {
            match env
                .call_method(
                    bridge.activity.as_obj(),
                    JNIString::new(method).as_ref(),
                    jni_sig!("()F"),
                    &[],
                )
                .and_then(|value| value.f())
            {
                Ok(value) if value.is_finite() && value > 0.0 => Ok(value),
                _ => {
                    if env.exception_check() {
                        let _ = env.exception_clear();
                    }
                    Ok(0.0)
                }
            }
        })
        .unwrap_or(0.0)
}

fn mode_value(mode: AndroidLoadMode) -> i32 {
    match mode {
        AndroidLoadMode::Replace => 0,
        AndroidLoadMode::Append => 1,
    }
}

fn call_android_picker(method: &str, signature: &str, args: &[JValue<'_>]) -> Result<(), String> {
    let Some(bridge) = BRIDGE.get() else {
        return Err("Android activity bridge is not initialized".to_string());
    };
    let method = JNIString::new(method);
    let signature = RuntimeMethodSignature::from_str(signature)
        .map_err(|error| format!("invalid Android picker signature: {error}"))?;
    bridge
        .vm
        .attach_current_thread(|env| -> JniResult<()> {
            env.call_method(
                bridge.activity.as_obj(),
                method.as_ref(),
                signature.method_signature(),
                args,
            )?;
            Ok(())
        })
        .map_err(|error| format!("failed to launch Android picker: {error}"))?;
    Ok(())
}

fn collect_audio_result(
    bridge_dir: &std::path::Path,
    name: &'static str,
    mode: AndroidLoadMode,
    results: &mut Vec<AndroidBridgeResult>,
) {
    let paths_file = bridge_dir.join(format!("{name}.paths"));
    if let Some(text) = take_file_to_string(&paths_file) {
        let paths = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(PathBuf::from)
            .collect::<Vec<_>>();
        if !paths.is_empty() {
            results.push(AndroidBridgeResult::AudioPaths { mode, paths });
        }
    }
    collect_cancel_error(bridge_dir, name, "Audio Picker", results);
}

fn collect_playlist_import_result(
    bridge_dir: &std::path::Path,
    results: &mut Vec<AndroidBridgeResult>,
) {
    let import_file = bridge_dir.join("playlist_import.m3u");
    if let Some(text) = take_file_to_string(&import_file) {
        results.push(AndroidBridgeResult::PlaylistImport { text });
    }
    collect_cancel_error(bridge_dir, "playlist_import", "Playlist Import", results);
}

fn collect_playlist_export_result(
    bridge_dir: &std::path::Path,
    results: &mut Vec<AndroidBridgeResult>,
) {
    let ok_file = bridge_dir.join("playlist_export.ok");
    if let Some(target) = take_file_to_string(&ok_file) {
        results.push(AndroidBridgeResult::PlaylistExport {
            target: target.trim().to_string(),
        });
    }
    collect_cancel_error(bridge_dir, "playlist_export", "Playlist Export", results);
}

fn collect_skin_import_result(
    bridge_dir: &std::path::Path,
    results: &mut Vec<AndroidBridgeResult>,
) {
    let path_file = bridge_dir.join("skin_import.path");
    if let Some(path) = take_file_to_string(&path_file) {
        let path = path.trim();
        if !path.is_empty() {
            results.push(AndroidBridgeResult::SkinImport {
                path: PathBuf::from(path),
            });
        }
    }
    collect_cancel_error(bridge_dir, "skin_import", "Skin Import", results);
}

fn collect_cancel_error(
    bridge_dir: &std::path::Path,
    name: &'static str,
    operation: &'static str,
    results: &mut Vec<AndroidBridgeResult>,
) {
    let cancel_file = bridge_dir.join(format!("{name}.cancel"));
    if cancel_file.is_file() {
        let _ = std::fs::remove_file(cancel_file);
        results.push(AndroidBridgeResult::Cancelled { operation });
    }

    let error_file = bridge_dir.join(format!("{name}.error"));
    if let Some(error) = take_file_to_string(&error_file) {
        let error = error.trim();
        results.push(AndroidBridgeResult::Error(if error.is_empty() {
            format!("{operation} failed")
        } else {
            error.to_string()
        }));
    }
}

fn take_file_to_string(path: &std::path::Path) -> Option<String> {
    if !path.is_file() {
        return None;
    }
    let text = std::fs::read_to_string(path).ok()?;
    let _ = std::fs::remove_file(path);
    Some(text)
}
