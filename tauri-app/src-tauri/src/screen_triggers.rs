//! Sylph Screen Triggers — Phase 7.2
//!
//! Heuristic event-based capture triggers:
//! - Window focus changes (EVENT_SYSTEM_FOREGROUND on Windows)
//! - Keyboard/mouse idle time tracking
//! - Activity state machine: active / passive / idle / deep_work / sleep
//!
//! Fires capture events based on activity cadence.

#![allow(dead_code)]

use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// Activity states for the capture cadence machine
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub enum ActivityState {
    /// User is actively interacting (switches windows, types) — capture every 30s
    Active,
    /// User is present but passive (reading, watching) — capture every 60s
    Passive,
    /// No input for 2+ minutes — capture every 5 minutes
    Idle,
    /// Continuous typing for 10+ minutes — reduce capture to every 10 min
    DeepWork,
    /// No input for 15+ minutes — stop capturing
    Sleep,
}

impl ActivityState {
    /// Capture interval for this state
    pub fn capture_interval(&self) -> Duration {
        match self {
            ActivityState::Active => Duration::from_secs(30),
            ActivityState::Passive => Duration::from_secs(60),
            ActivityState::Idle => Duration::from_secs(300),
            ActivityState::DeepWork => Duration::from_secs(600),
            ActivityState::Sleep => Duration::from_secs(u64::MAX), // never
        }
    }
}

/// Tracks user activity and determines capture cadence
pub struct ActivityTracker {
    state: Mutex<ActivityState>,
    last_input_time: Mutex<Instant>,
    last_window_change: Mutex<Instant>,
    last_capture_time: Mutex<Instant>,
    continuous_typing_start: Mutex<Option<Instant>>,
    window_change_count: AtomicU64,
    last_window_title: Mutex<String>,
}

impl ActivityTracker {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            state: Mutex::new(ActivityState::Active),
            last_input_time: Mutex::new(now),
            last_window_change: Mutex::new(now),
            last_capture_time: Mutex::new(now),
            continuous_typing_start: Mutex::new(None),
            window_change_count: AtomicU64::new(0),
            last_window_title: Mutex::new(String::new()),
        }
    }

    /// Notify that user input was detected (keyboard/mouse)
    pub fn on_user_input(&self) {
        let now = Instant::now();
        *self.last_input_time.lock().unwrap() = now;
        self.update_state();
    }

    /// Notify that the foreground window changed
    pub fn on_window_change(&self) {
        let now = Instant::now();
        *self.last_window_change.lock().unwrap() = now;
        self.window_change_count.fetch_add(1, Ordering::Relaxed);
        *self.last_input_time.lock().unwrap() = now;

        // Reset deep work on window switch
        *self.continuous_typing_start.lock().unwrap() = None;

        self.update_state();
    }

    /// Notify that continuous typing was detected
    pub fn on_typing_detected(&self) {
        let now = Instant::now();
        let mut start = self.continuous_typing_start.lock().unwrap();
        if start.is_none() {
            *start = Some(now);
        }
        *self.last_input_time.lock().unwrap() = now;
        self.update_state();
    }

    /// Check whether a capture should fire now
    pub fn should_capture(&self) -> bool {
        let state = *self.state.lock().unwrap();
        if state == ActivityState::Sleep {
            return false;
        }

        let interval = state.capture_interval();
        let last_capture = *self.last_capture_time.lock().unwrap();

        if last_capture.elapsed() >= interval {
            *self.last_capture_time.lock().unwrap() = Instant::now();
            true
        } else {
            false
        }
    }

    /// Force a capture (e.g., on window focus change)
    pub fn force_capture(&self) {
        *self.last_capture_time.lock().unwrap() = Instant::now();
    }

    /// Get the current activity state
    pub fn current_state(&self) -> ActivityState {
        *self.state.lock().unwrap()
    }

    /// Get seconds since last user input
    pub fn idle_seconds(&self) -> u64 {
        self.last_input_time.lock().unwrap().elapsed().as_secs()
    }

    fn update_state(&self) {
        let idle = self.last_input_time.lock().unwrap().elapsed();
        let typing_duration = self
            .continuous_typing_start
            .lock()
            .unwrap()
            .map(|start| start.elapsed())
            .unwrap_or(Duration::ZERO);

        let new_state = if idle > Duration::from_secs(900) {
            // 15+ min idle → sleep
            ActivityState::Sleep
        } else if idle > Duration::from_secs(120) {
            // 2+ min idle → idle
            ActivityState::Idle
        } else if typing_duration > Duration::from_secs(600) {
            // 10+ min continuous typing → deep work
            ActivityState::DeepWork
        } else if idle < Duration::from_secs(10) {
            // Recent activity → active
            ActivityState::Active
        } else {
            // 10s-2min since last input → passive
            ActivityState::Passive
        };

        *self.state.lock().unwrap() = new_state;
    }

    /// Poll current activity state from the OS
    pub fn poll_activity(&self) {
        let idle_ms = get_os_idle_time_ms();
        let now = Instant::now();

        // Update last input time based on OS idle time
        {
            if let Ok(mut last_input) = self.last_input_time.lock() {
                // Subtract idle time from current instant to get when the last input occurred
                *last_input = now.checked_sub(Duration::from_millis(idle_ms)).unwrap_or(now);
            }
        }

        // Update continuous typing start
        {
            if let Ok(mut typing_start) = self.continuous_typing_start.lock() {
                if idle_ms < 1000 {
                    // If the user is active, track start of active interaction
                    if typing_start.is_none() {
                        *typing_start = Some(now);
                    }
                } else if idle_ms >= 5000 {
                    // If idle for 5+ seconds, reset typing start
                    *typing_start = None;
                }
            }
        }

        // Detect window changes
        let current_window = get_active_window_title();
        {
            if let Ok(mut last_title) = self.last_window_title.lock() {
                if current_window != *last_title {
                    *last_title = current_window;
                    if let Ok(mut window_change) = self.last_window_change.lock() {
                        *window_change = now;
                    }
                    self.window_change_count.fetch_add(1, Ordering::Relaxed);
                    if let Ok(mut typing_start) = self.continuous_typing_start.lock() {
                        *typing_start = None;
                    }
                }
            }
        }

        self.update_state();
    }
}

/// Get the active foreground window title
#[cfg(target_os = "windows")]
pub fn get_active_window_title() -> String {
    extern "system" {
        fn GetForegroundWindow() -> isize;
        fn GetWindowTextW(hwnd: isize, lpString: *mut u16, nMaxCount: i32) -> i32;
    }
    unsafe {
        let hwnd = GetForegroundWindow();
        if hwnd == 0 {
            return "Unknown".to_string();
        }
        let mut buf = vec![0u16; 512];
        let len = GetWindowTextW(hwnd, buf.as_mut_ptr(), buf.len() as i32);
        if len > 0 {
            String::from_utf16_lossy(&buf[..len as usize])
        } else {
            "Unknown".to_string()
        }
    }
}

#[cfg(not(target_os = "windows"))]
pub fn get_active_window_title() -> String {
    "Unknown".to_string()
}

/// Get the idle time from the OS (Windows)
#[cfg(target_os = "windows")]
pub fn get_os_idle_time_ms() -> u64 {
    use std::mem;
    #[repr(C)]
    struct LASTINPUTINFO {
        cb_size: u32,
        dw_time: u32,
    }
    extern "system" {
        fn GetLastInputInfo(plii: *mut LASTINPUTINFO) -> i32;
        fn GetTickCount() -> u32;
    }
    unsafe {
        let mut lii = LASTINPUTINFO {
            cb_size: mem::size_of::<LASTINPUTINFO>() as u32,
            dw_time: 0,
        };
        if GetLastInputInfo(&mut lii) != 0 {
            let tick = GetTickCount();
            (tick.wrapping_sub(lii.dw_time)) as u64
        } else {
            0
        }
    }
}

#[cfg(not(target_os = "windows"))]
pub fn get_os_idle_time_ms() -> u64 {
    0 // Placeholder for non-Windows
}
