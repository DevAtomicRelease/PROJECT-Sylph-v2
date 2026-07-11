//! Sylph Screen Capture — Phase 7.1
//!
//! Captures the primary monitor, applies privacy zone masking,
//! resizes to max 1280x720, compresses to JPEG quality 60,
//! and returns base64-encoded JPEG.

use base64::{engine::general_purpose, Engine as _};
use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, GenericImageView, RgbaImage};
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use xcap::Monitor;

/// Privacy zone rectangle (coordinates in screen pixels)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyZone {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Result of a screen capture
#[derive(Debug, Clone, Serialize)]
pub struct CaptureResult {
    pub base64_jpeg: String,
    pub width: u32,
    pub height: u32,
    pub monitor_name: String,
}

/// Capture the primary monitor and return a base64-encoded JPEG.
pub fn capture_screen(privacy_zones: &[PrivacyZone]) -> Result<CaptureResult, String> {
    // Get all monitors
    let monitors = Monitor::all().map_err(|e| format!("Failed to list monitors: {}", e))?;

    // Find the primary monitor
    let monitor = monitors
        .into_iter()
        .find(|m| m.is_primary())
        .ok_or_else(|| "No primary monitor found".to_string())?;

    let monitor_name = monitor.name().to_string();

    // Capture the screen
    let raw_image = monitor
        .capture_image()
        .map_err(|e| format!("Screen capture failed: {}", e))?;

    // Convert to DynamicImage for processing
    let (w, h) = (raw_image.width(), raw_image.height());
    let mut img = DynamicImage::ImageRgba8(
        RgbaImage::from_raw(w, h, raw_image.into_raw())
            .ok_or_else(|| "Failed to create image from raw pixels".to_string())?,
    );

    // Apply privacy zone masking (fill with black)
    if !privacy_zones.is_empty() {
        let rgba = img.as_mut_rgba8().ok_or("Failed to get mutable RGBA")?;
        for zone in privacy_zones {
            let x_end = (zone.x + zone.width).min(w);
            let y_end = (zone.y + zone.height).min(h);
            for y in zone.y..y_end {
                for x in zone.x..x_end {
                    rgba.put_pixel(x, y, image::Rgba([0, 0, 0, 255]));
                }
            }
        }
    }

    // Resize to max 1280x720 maintaining aspect ratio
    let max_w = 1280u32;
    let max_h = 720u32;
    let (orig_w, orig_h) = img.dimensions();
    let img = if orig_w > max_w || orig_h > max_h {
        let scale = f64::min(max_w as f64 / orig_w as f64, max_h as f64 / orig_h as f64);
        let new_w = (orig_w as f64 * scale) as u32;
        let new_h = (orig_h as f64 * scale) as u32;
        img.resize_exact(new_w, new_h, image::imageops::FilterType::Triangle)
    } else {
        img
    };

    let (final_w, final_h) = img.dimensions();

    // Encode as JPEG quality 60
    let rgb_img = img.to_rgb8();
    let mut jpeg_buf = Cursor::new(Vec::new());
    let mut encoder = JpegEncoder::new_with_quality(&mut jpeg_buf, 60);
    encoder
        .encode(&rgb_img, final_w, final_h, image::ExtendedColorType::Rgb8)
        .map_err(|e| format!("JPEG encoding failed: {}", e))?;

    let base64_jpeg = general_purpose::STANDARD.encode(jpeg_buf.into_inner());

    Ok(CaptureResult {
        base64_jpeg,
        width: final_w,
        height: final_h,
        monitor_name,
    })
}
