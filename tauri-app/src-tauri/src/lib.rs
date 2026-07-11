//! Sylph Desktop — Tauri v2 Application Core
//!
//! Phase 1.3: System tray + push-to-talk hotkey
//! Phase 2.2: Python sidecar spawn + supervision
//! Phase 7.1: Screen capture command
//! Phase 7.2: Activity tracking

mod screen_capture;
mod screen_triggers;

use serde::Serialize;
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use tauri::{
    menu::{Menu, MenuItem},
    tray::TrayIconBuilder,
    AppHandle, Emitter, Manager, State,
};

// Windows API: SetWindowDisplayAffinity
// WDA_EXCLUDEFROMCAPTURE (0x11) tells the DWM compositor to permanently
// exclude a window from all screen capture methods (BitBlt, Desktop
// Duplication API, PrintWindow). Available since Windows 10 2004+.
#[cfg(target_os = "windows")]
extern "system" {
    fn SetWindowDisplayAffinity(hwnd: isize, dwaffinity: u32) -> i32;
}
#[cfg(target_os = "windows")]
const WDA_EXCLUDEFROMCAPTURE: u32 = 0x00000011;

/// Global flag for push-to-talk recording state
static IS_RECORDING: AtomicBool = AtomicBool::new(false);
static IS_QUIET: AtomicBool = AtomicBool::new(false);

#[cfg(target_os = "windows")]
fn get_windows_working_area() -> (i32, i32, i32, i32) {
    #[repr(C)]
    struct RECT {
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    }
    extern "system" {
        fn SystemParametersInfoW(
            ui_action: u32,
            ui_param: u32,
            pv_param: *mut RECT,
            f_win_ini: u32,
        ) -> i32;
    }
    unsafe {
        let mut rect = RECT { left: 0, top: 0, right: 0, bottom: 0 };
        const SPI_GETWORKAREA: u32 = 48;
        if SystemParametersInfoW(SPI_GETWORKAREA, 0, &mut rect, 0) != 0 {
            (rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top)
        } else {
            (0, 0, 1920, 1080)
        }
    }
}

// ---------------------------------------------------------------------------
// Tauri Commands (callable from React frontend)
// ---------------------------------------------------------------------------

/// Returns the current recording state.
#[tauri::command]
fn get_recording_state() -> bool {
    IS_RECORDING.load(Ordering::SeqCst)
}

/// Sync the frontend recording state to Rust.
#[tauri::command]
fn set_recording_state(recording: bool) {
    IS_RECORDING.store(recording, Ordering::SeqCst);
    log::info!("Recording state synchronized to Rust: {}", recording);
}

/// Sync the frontend quiet mode state to Rust.
#[tauri::command]
fn set_quiet_mode(quiet: bool) {
    IS_QUIET.store(quiet, Ordering::SeqCst);
    log::info!("Quiet mode synchronized to Rust: {}", quiet);
}

/// Returns the active foreground window title.
#[tauri::command]
fn get_active_window_title() -> String {
    screen_triggers::get_active_window_title()
}

/// Returns sidecar health status.
#[tauri::command]
async fn check_sidecar_health() -> Result<String, String> {
    Ok("sidecar_health_check_placeholder".to_string())
}

/// Phase 7.1: Capture the primary screen and return base64 JPEG.
///
/// The Sylph window has WDA_EXCLUDEFROMCAPTURE set at startup, so it is
/// automatically excluded from all screen captures by the DWM compositor.
/// No hide/show or position changes are needed — zero added latency.
#[tauri::command]
async fn capture_screen(_app: tauri::AppHandle) -> Result<screen_capture::CaptureResult, String> {
    // Privacy zones: currently empty (opt-in only).
    let privacy_zones: Vec<screen_capture::PrivacyZone> = vec![];
    if privacy_zones.is_empty() {
        log::debug!("[capture_screen] 0 privacy zones active, frame unmasked");
    } else {
        log::info!("[capture_screen] {} privacy zones applied", privacy_zones.len());
    }

    screen_capture::capture_screen(&privacy_zones)
}

/// Phase 7.2: Get the current activity state and idle time.
#[tauri::command]
fn get_activity_state(
    tracker: State<'_, Mutex<screen_triggers::ActivityTracker>>,
) -> Result<serde_json::Value, String> {
    let tracker = tracker.lock().map_err(|e| e.to_string())?;
    Ok(serde_json::json!({
        "state": tracker.current_state(),
        "idle_seconds": tracker.idle_seconds(),
    }))
}

/// Notify the frontend about recording state changes.
#[derive(Clone, Serialize)]
struct RecordingEvent {
    is_recording: bool,
}

// ---------------------------------------------------------------------------
// System Tray Setup (Phase 1.3)
// ---------------------------------------------------------------------------

fn setup_tray(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    let show_hide = MenuItem::with_id(app, "show_hide", "Show/Hide Sylph", true, None::<&str>)?;
    let quiet_mode = MenuItem::with_id(app, "quiet_mode", "Quiet Mode", true, None::<&str>)?;
    let settings = MenuItem::with_id(app, "settings", "Settings", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Sylph", true, None::<&str>)?;

    let menu = Menu::with_items(app, &[&show_hide, &quiet_mode, &settings, &quit])?;

    let _tray = TrayIconBuilder::new()
        .menu(&menu)
        .tooltip("Sylph — Desktop Companion")
        .on_menu_event(move |app, event| match event.id.as_ref() {
            "show_hide" => {
                if let Some(window) = app.get_webview_window("main") {
                    if window.is_visible().unwrap_or(false) {
                        let _ = window.hide();
                    } else {
                        let _ = window.show();
                        let _ = window.set_focus();
                    }
                }
            }
            "quiet_mode" => {
                let _ = app.emit("quiet_mode_toggle", ());
                log::info!("Quiet mode toggled");
            }
            "settings" => {
                let _ = app.emit("open_settings", ());
                log::info!("Settings requested");
            }
            "quit" => {
                log::info!("Quit requested from tray");
                app.exit(0);
            }
            _ => {}
        })
        .build(app)?;

    log::info!("System tray initialized");
    Ok(())
}

// ---------------------------------------------------------------------------
// Push-to-Talk Hotkey Setup (Phase 1.3)
// ---------------------------------------------------------------------------

fn setup_hotkey(app: &AppHandle) -> Result<(), Box<dyn std::error::Error>> {
    use tauri_plugin_global_shortcut::{GlobalShortcutExt, Shortcut, ShortcutState};

    let shortcut: Shortcut = "ctrl+shift+space".parse()?;

    app.global_shortcut().on_shortcut(shortcut, move |app, _shortcut, event| {
        if event.state == ShortcutState::Pressed {
            let was_recording = IS_RECORDING.fetch_xor(true, Ordering::SeqCst);
            let now_recording = !was_recording;

            log::info!(
                "Push-to-talk: {}",
                if now_recording { "RECORDING" } else { "STOPPED" }
            );

            let _ = app.emit(
                "recording_state",
                RecordingEvent {
                    is_recording: now_recording,
                },
            );
        }
    })?;

    log::info!("Global hotkey registered: Ctrl+Shift+Space (push-to-talk toggle)");
    Ok(())
}

// ---------------------------------------------------------------------------
// Application Entry Point
// ---------------------------------------------------------------------------
// Sidecar Supervisor (Phase 2.2)
// ---------------------------------------------------------------------------

pub struct SidecarManager {
    child: Arc<Mutex<Option<Child>>>,
}

impl SidecarManager {
    pub fn new() -> Self {
        Self {
            child: Arc::new(Mutex::new(None)),
        }
    }

    pub fn start(&self) {
        let child_clone = self.child.clone();
        thread::spawn(move || {
            let mut retries = 0;
            let max_retries = 3;

            // Determine paths relative to current directory
            let current_dir = std::env::current_dir().unwrap_or_default();
            log::info!("[SidecarManager] Current working directory: {:?}", current_dir);

            // Dynamically resolve python path and main.py script by traversing up parent directories
            let mut check_dir = current_dir.clone();
            let mut resolved = None;
            for _ in 0..4 {
                let python_bin = check_dir.join("venv/Scripts/python.exe");
                let script_file = check_dir.join("sidecar/main.py");
                if python_bin.exists() && script_file.exists() {
                    resolved = Some((python_bin, script_file, check_dir.clone()));
                    break;
                }
                
                // Also check Unix/MacOS virtualenv layout just in case
                let python_bin_unix = check_dir.join("venv/bin/python");
                if python_bin_unix.exists() && script_file.exists() {
                    resolved = Some((python_bin_unix, script_file, check_dir.clone()));
                    break;
                }

                if !check_dir.pop() {
                    break;
                }
            }

            // Trigger rebuild to reload updated Python sidecar, system prompt, and VAD thresholds
            let (python_path, script_path, work_dir) = if let Some(paths) = resolved {
                paths
            } else {
                log::warn!("[SidecarManager] Could not find venv/Scripts/python.exe or sidecar/main.py in current or parent directories. Falling back to system python.");
                (
                    std::path::PathBuf::from("python"),
                    std::path::PathBuf::from("sidecar/main.py"),
                    current_dir.clone(),
                )
            };

            // Sanity check: strip any stray quote characters from the path string.
            // On some Windows configurations, paths can get wrapped in literal quotes
            // which Command::new() does not expect (it handles quoting internally).
            let python_path_str = python_path.to_string_lossy().replace('"', "");
            let python_path = std::path::PathBuf::from(&python_path_str);

            // Sanity check: verify the resolved Python binary actually exists on disk.
            // A venv python.exe is a shim that points to the base Python installation.
            // If that base Python was uninstalled, the shim still "exists" as a file but
            // will fail at runtime with a cryptic exit code 103 and "No Python at ..." error.
            if python_path.to_str() != Some("python") && !python_path.exists() {
                log::error!(
                    "[SidecarManager] FATAL: Python binary does not exist at resolved path: {:?}. \
                     The virtualenv may be broken or the base Python was uninstalled. \
                     Please recreate the venv with: python -m venv venv",
                    python_path
                );
                return;
            }

            // Try to canonicalize (resolve symlinks/junctions) — if this fails, the path
            // may be a broken shim. Log a warning but continue since Command::new will
            // give us a proper error.
            match python_path.canonicalize() {
                Ok(canonical) => {
                    log::info!(
                        "[SidecarManager] Python path validated: {:?} → canonical {:?}",
                        python_path, canonical
                    );
                }
                Err(e) => {
                    log::warn!(
                        "[SidecarManager] Could not canonicalize Python path {:?}: {}. \
                         The venv shim may point to a nonexistent base Python installation.",
                        python_path, e
                    );
                }
            }

            log::info!(
                "[SidecarManager] Spawning sidecar: python={:?}, script={:?}, work_dir={:?}",
                python_path,
                script_path,
                work_dir
            );

            while retries < max_retries {
                log::info!("[SidecarManager] Starting Python sidecar (attempt {})...", retries + 1);

                let mut cmd = Command::new(&python_path);
                cmd.arg(&script_path);
                cmd.current_dir(&work_dir);

                // Hide console window on Windows
                #[cfg(windows)]
                {
                    use std::os::windows::process::CommandExt;
                    const CREATE_NO_WINDOW: u32 = 0x08000000;
                    cmd.creation_flags(CREATE_NO_WINDOW);
                }

                match cmd.spawn() {
                    Ok(child) => {
                        log::info!("[SidecarManager] Sidecar process started successfully with PID: {}", child.id());
                        
                        // Save the child handle
                        {
                            let mut lock = child_clone.lock().unwrap();
                            *lock = Some(child);
                        }

                        // Wait for child to exit
                        let mut running_ms = 0;
                        loop {
                            let mut lock = child_clone.lock().unwrap();
                            if let Some(ref mut c) = *lock {
                                match c.try_wait() {
                                    Ok(Some(status)) => {
                                        log::warn!("[SidecarManager] Sidecar process exited with status: {:?}", status);
                                        break; // Process exited, exit loop to restart
                                    }
                                    Ok(None) => {
                                        // Still running, sleep and check again
                                    }
                                    Err(e) => {
                                        log::error!("[SidecarManager] Error waiting for sidecar: {:?}", e);
                                        break;
                                    }
                                }
                            } else {
                                break;
                            }
                            drop(lock); // Release lock before sleeping
                            thread::sleep(Duration::from_millis(500));
                            
                            running_ms += 500;
                            if running_ms == 10000 {
                                log::info!("[SidecarManager] Sidecar has been running stably for 10s. Resetting retry count.");
                                retries = 0;
                            }
                        }

                        retries += 1;
                        log::info!("[SidecarManager] Sleeping 5s before restarting...");
                        thread::sleep(Duration::from_secs(5));
                    }
                    Err(e) => {
                        log::error!("[SidecarManager] Failed to spawn sidecar process: {:?}", e);
                        retries += 1;
                        thread::sleep(Duration::from_secs(5));
                    }
                }
            }
            log::error!("[SidecarManager] Sidecar failed to start after {} retries. Stopping supervision.", max_retries);
        });
    }

    pub fn stop(&self) {
        let mut lock = self.child.lock().unwrap();
        if let Some(mut child) = lock.take() {
            log::info!("[SidecarManager] Terminating sidecar process PID: {}", child.id());
            let _ = child.kill();
        }
    }
}

// ---------------------------------------------------------------------------

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    tauri::Builder::default()
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_shell::init())
        // Phase 7.2: Activity tracker as managed state
        .manage(Mutex::new(screen_triggers::ActivityTracker::new()))
        .invoke_handler(tauri::generate_handler![
            get_recording_state,
            set_recording_state,
            check_sidecar_health,
            capture_screen,
            get_activity_state,
            set_quiet_mode,
            get_active_window_title,
        ])
        .setup(|app| {
            log::info!("=== Sylph Desktop starting ===");

            // Spawn sidecar automatically (Phase 2.2)
            let sidecar = SidecarManager::new();
            sidecar.start();
            app.manage(sidecar);

            // Spawn background activity tracker loop
            let app_handle = app.handle().clone();
            thread::spawn(move || {
                log::info!("[ActivityThread] Starting background activity tracker loop");
                loop {
                    thread::sleep(Duration::from_secs(1));

                    if let Some(tracker_mutex) = app_handle.try_state::<Mutex<screen_triggers::ActivityTracker>>() {
                        if let Ok(tracker) = tracker_mutex.lock() {
                            tracker.poll_activity();
                        }
                    }
                }
            });

            // Phase 1.3: Register global shortcut plugin
            app.handle().plugin(
                tauri_plugin_global_shortcut::Builder::new().build(),
            )?;

            // Phase 1.3: System tray
            setup_tray(app.handle())?;

            // Phase 1.3: Push-to-talk hotkey
            setup_hotkey(app.handle())?;

            // Position window at bottom-right of screen, touching the taskbar
            if let Some(window) = app.get_webview_window("main") {
                if let Some(monitor) = window.current_monitor().ok().flatten() {
                    let scale = monitor.scale_factor();
                    
                    let (avail_x, avail_y, avail_w, avail_h);
                    
                    #[cfg(target_os = "windows")]
                    {
                        let (wx, wy, ww, wh) = get_windows_working_area();
                        // Win32 API returns coordinates in physical pixels
                        avail_x = (wx as f64 / scale) as i32;
                        avail_y = (wy as f64 / scale) as i32;
                        avail_w = (ww as f64 / scale) as i32;
                        avail_h = (wh as f64 / scale) as i32;
                    }
                    
                    #[cfg(not(target_os = "windows"))]
                    {
                        let m_pos = monitor.position();
                        let m_size = monitor.size();
                        avail_x = (m_pos.x as f64 / scale) as i32;
                        avail_y = (m_pos.y as f64 / scale) as i32;
                        avail_w = (m_size.width as f64 / scale) as i32;
                        avail_h = (m_size.height as f64 / scale) as i32;
                    }

                    let win_w = 350;
                    let win_h = 500;
                    let margin_right = 20;

                    let x = avail_x + avail_w - win_w - margin_right;
                    let y = avail_y + avail_h - win_h; // Glued to the top of the taskbar

                    let _ = window.set_position(tauri::LogicalPosition::new(x, y));
                    log::info!("Window positioned touching taskbar at: ({}, {})", x, y);
                }

                // Exclude the Sylph window from all screen captures.
                // This tells the Windows DWM compositor to render a black
                // hole where the Sylph window is in capture output, effectively
                // making it invisible to xcap's Monitor::capture_image().
                // The desktop content behind Sylph shows through instead.
                #[cfg(target_os = "windows")]
                {
                    match window.hwnd() {
                        Ok(hwnd) => {
                            let result = unsafe {
                                SetWindowDisplayAffinity(hwnd.0 as isize, WDA_EXCLUDEFROMCAPTURE)
                            };
                            if result != 0 {
                                log::info!("[Setup] SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE) succeeded — Sylph excluded from screen captures");
                            } else {
                                log::warn!("[Setup] SetWindowDisplayAffinity failed (err={}). Screen captures may include the Sylph overlay. Requires Windows 10 2004+.", 
                                    std::io::Error::last_os_error());
                            }
                        }
                        Err(e) => {
                            log::warn!("[Setup] Could not get HWND for display affinity: {:?}", e);
                        }
                    }
                }
            }

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Sylph")
        .run(|app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                if let Some(sidecar) = app_handle.try_state::<SidecarManager>() {
                    sidecar.stop();
                }
            }
        });
}
