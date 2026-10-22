//! Prepare a form ▸ field detection by a trained model: FFDNet, the form-field detector of the
//! CommonForms project (Barrow, *CommonForms: A Large, Diverse Dataset for Form Field Detection*,
//! 2025), run through rten from its ONNX export.
//!
//! The model looks at a picture of the page, so it finds fields however the page was made: drawn
//! boxes and rules, check boxes that are font glyphs, labelled table cells, scans and pages whose
//! text was printed as outlines. It finds where fields go and what kind they are (text, check box,
//! signature); naming them from their labels is the caller's ([`pdfcraft_forms::detect::place`]).
//!
//! The model file is **not** part of PdfCraft and is never fetched by it: the user places it in a
//! models folder ([`Models::search_dirs`]) themselves. Without it [`FieldModel::find`] returns
//! [`OcrError::NoModels`] and callers use the built-in detector.
//!
//! The network is a YOLO-style detector: a square RGB input (the page letterboxed onto grey),
//! and one output row per anchor of `[cx, cy, w, h, score per class…]` in input pixels.

use std::path::PathBuf;

use rten::{Dimension, Model};
use rten_tensor::NdTensor;
use rten_tensor::prelude::*;

use crate::{Models, OcrError};

/// The model files looked for, best first.
pub const FIELD_MODELS: &[&str] = &["FFDNet-L.onnx", "FFDNet-S.onnx"];

/// The input side when the model does not fix one (the size FFDNet's ONNX export is made for).
const DEFAULT_SIDE: usize = 1216;
/// The largest input side accepted from a model file.
const MAX_SIDE: usize = 4096;
/// Letterbox padding grey, as the model was trained with.
const PAD: f32 = 114.0 / 255.0;
/// The largest picture accepted, in pixels.
const MAX_PIXELS: usize = 40_000_000;
/// The most raw detections kept for suppression (by score), bounding its quadratic work.
const MAX_RAW: usize = 3000;

/// What the model found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldKind {
    Text,
    CheckBox,
    Signature,
}

impl FieldKind {
    fn from_class(c: usize) -> Option<FieldKind> {
        match c {
            0 => Some(FieldKind::Text),
            1 => Some(FieldKind::CheckBox),
            2 => Some(FieldKind::Signature),
            _ => None,
        }
    }
}

/// A field the model found: its kind, confidence and box in picture pixels
/// `[left, top, right, bottom]` (y down).
#[derive(Clone, Debug, PartialEq)]
pub struct FoundField {
    pub kind: FieldKind,
    pub score: f32,
    pub rect: [f32; 4],
}

/// How detection decides.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldSettings {
    /// Lowest score kept.
    pub confidence: f32,
    /// Overlap (intersection over union) above which the weaker of two same-kind boxes goes.
    pub iou: f32,
    /// Share of the smaller box covered by a box of another kind above which the weaker goes.
    pub cross_kind: f32,
}

impl Default for FieldSettings {
    fn default() -> Self {
        // 0.2 found the Disabled column and the employer field on the Delta Dental form that 0.3
        // missed, without adding false fields.
        FieldSettings { confidence: 0.2, iou: 0.5, cross_kind: 0.6 }
    }
}

/// A loaded field-detection model.
pub struct FieldModel {
    model: Model,
    side: usize,
    path: PathBuf,
}

impl std::fmt::Debug for FieldModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FieldModel").field("path", &self.path).field("side", &self.side).finish()
    }
}

impl FieldModel {
    /// The first model file in the models folders ([`Models::search_dirs`]), best model first.
    pub fn path() -> Option<PathBuf> {
        let dirs = Models::search_dirs();
        FIELD_MODELS.iter().find_map(|name| dirs.iter().map(|d| d.join(name)).find(|p| p.is_file()))
    }

    /// Load the model at `path`.
    pub fn load(path: PathBuf) -> Result<FieldModel, OcrError> {
        let model = Model::load_file(&path).map_err(|e| OcrError::Load(path.display().to_string(), e.to_string()))?;
        let side = match model.input_shape(0).as_deref() {
            Some([_, _, Dimension::Fixed(h), Dimension::Fixed(w)]) if h == w && (32..=MAX_SIDE).contains(h) => *h,
            _ => DEFAULT_SIDE,
        };
        Ok(FieldModel { model, side, path })
    }

    /// Load the model [`FieldModel::path`] finds.
    pub fn find() -> Result<FieldModel, OcrError> {
        Self::load(Self::path().ok_or(OcrError::NoModels)?)
    }

    /// Where the model was loaded from.
    pub fn file(&self) -> &std::path::Path {
        &self.path
    }

    /// The fields in `rgba` (premultiplied RGBA8, `width` × `height`, read as composited over
    /// white), with boxes in its pixels.
    pub fn detect(&self, rgba: &[u8], width: u32, height: u32, settings: FieldSettings) -> Result<Vec<FoundField>, OcrError> {
        let (w, h) = (width as usize, height as usize);
        let pixels = w.checked_mul(h).filter(|n| *n > 0 && *n <= MAX_PIXELS).ok_or(OcrError::EmptyImage)?;
        if rgba.len() < pixels.saturating_mul(4) {
            return Err(OcrError::EmptyImage);
        }
        let (input, fit) = letterbox(rgba, w, h, self.side).ok_or(OcrError::EmptyImage)?;
        let out = self.model.run_one(input.view().into(), None).map_err(|e| OcrError::Recognize(e.to_string()))?;
        let out: NdTensor<f32, 3> = out.try_into().map_err(|e: rten::TryFromValueError| OcrError::Recognize(e.to_string()))?;
        Ok(suppress(decode(&out, fit, settings.confidence), settings))
    }
}

/// How the picture sits in the model's input: its scale and the padding before it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Fit {
    scale: f32,
    pad_x: f32,
    pad_y: f32,
    width: f32,
    height: f32,
}

/// The picture scaled to fit a `side` × `side` input (bilinear, keeping its aspect), centred on
/// grey, as planar RGB in 0..1; `None` for an empty picture.
fn letterbox(rgba: &[u8], w: usize, h: usize, side: usize) -> Option<(NdTensor<f32, 4>, Fit)> {
    if w == 0 || h == 0 || side == 0 || w.checked_mul(h)? > MAX_PIXELS {
        return None;
    }
    let scale = (side as f32 / w as f32).min(side as f32 / h as f32);
    let nw = ((w as f32 * scale).round() as usize).clamp(1, side);
    let nh = ((h as f32 * scale).round() as usize).clamp(1, side);
    let (px, py) = ((side - nw) / 2, (side - nh) / 2);
    let plane = side.checked_mul(side)?;
    let mut data = vec![PAD; plane.checked_mul(3)?];
    // A premultiplied pixel over white: colour + (255 − alpha).
    let at = |x: usize, y: usize, c: usize| -> f32 {
        let i = (y * w + x) * 4;
        let v = rgba.get(i + c).copied().unwrap_or(255);
        let a = rgba.get(i + 3).copied().unwrap_or(255);
        f32::from(v.saturating_add(255 - a)) / 255.0
    };
    let (sx, sy) = (w as f32 / nw as f32, h as f32 / nh as f32);
    for oy in 0..nh {
        let fy = ((oy as f32 + 0.5) * sy - 0.5).clamp(0.0, (h - 1) as f32);
        let (y0, ty) = (fy.floor() as usize, fy.fract());
        let y1 = (y0 + 1).min(h - 1);
        for ox in 0..nw {
            let fx = ((ox as f32 + 0.5) * sx - 0.5).clamp(0.0, (w - 1) as f32);
            let (x0, tx) = (fx.floor() as usize, fx.fract());
            let x1 = (x0 + 1).min(w - 1);
            let o = (py + oy) * side + px + ox;
            for c in 0..3 {
                let top = at(x0, y0, c) * (1.0 - tx) + at(x1, y0, c) * tx;
                let bottom = at(x0, y1, c) * (1.0 - tx) + at(x1, y1, c) * tx;
                if let Some(d) = data.get_mut(c * plane + o) {
                    *d = top * (1.0 - ty) + bottom * ty;
                }
            }
        }
    }
    let fit = Fit { scale, pad_x: px as f32, pad_y: py as f32, width: w as f32, height: h as f32 };
    // `from_data` panics only on a length mismatch; `data` holds exactly 3 × side × side values.
    Some((NdTensor::from_data([1, 3, side, side], data), fit))
}

/// The detections in the model's output (`[1, 4 + classes, anchors]`) scoring at least
/// `confidence`, mapped back to picture pixels and clipped to it; the best [`MAX_RAW`] at most.
fn decode(out: &NdTensor<f32, 3>, fit: Fit, confidence: f32) -> Vec<FoundField> {
    let [batch, rows, anchors] = out.shape();
    if batch == 0 || rows < 5 || fit.scale <= 0.0 || !fit.scale.is_finite() {
        return Vec::new();
    }
    let get = |r: usize, a: usize| out.get([0, r, a]).copied().unwrap_or(0.0);
    let mut found = Vec::new();
    for a in 0..anchors {
        let Some((class, score)) = (4..rows).map(|r| (r - 4, get(r, a))).filter(|(_, s)| s.is_finite()).max_by(|x, y| x.1.total_cmp(&y.1)) else {
            continue;
        };
        let Some(kind) = FieldKind::from_class(class) else { continue };
        if score < confidence {
            continue;
        }
        let (cx, cy, bw, bh) = (get(0, a), get(1, a), get(2, a), get(3, a));
        let map_x = |v: f32| ((v - fit.pad_x) / fit.scale).clamp(0.0, fit.width);
        let map_y = |v: f32| ((v - fit.pad_y) / fit.scale).clamp(0.0, fit.height);
        let rect = [map_x(cx - bw / 2.0), map_y(cy - bh / 2.0), map_x(cx + bw / 2.0), map_y(cy + bh / 2.0)];
        if rect.iter().all(|v| v.is_finite()) && rect[2] - rect[0] >= 1.0 && rect[3] - rect[1] >= 1.0 {
            found.push(FoundField { kind, score, rect });
        }
    }
    found.sort_by(|a, b| b.score.total_cmp(&a.score));
    found.truncate(MAX_RAW);
    found
}

fn area(r: [f32; 4]) -> f32 {
    (r[2] - r[0]).max(0.0) * (r[3] - r[1]).max(0.0)
}

fn intersection(a: [f32; 4], b: [f32; 4]) -> f32 {
    area([a[0].max(b[0]), a[1].max(b[1]), a[2].min(b[2]), a[3].min(b[3])])
}

/// Non-maximum suppression over `found` (sorted by score, best first): within a kind, a box
/// overlapping a stronger one by more than `iou` goes; across kinds, one mostly covered by a
/// stronger box goes.
fn suppress(found: Vec<FoundField>, settings: FieldSettings) -> Vec<FoundField> {
    let mut kept: Vec<FoundField> = Vec::new();
    for f in found {
        let clash = kept.iter().any(|k| {
            let i = intersection(k.rect, f.rect);
            if k.kind == f.kind {
                let union = area(k.rect) + area(f.rect) - i;
                union > 0.0 && i / union > settings.iou
            } else {
                let smaller = area(k.rect).min(area(f.rect));
                smaller > 0.0 && i / smaller > settings.cross_kind
            }
        });
        if !clash {
            kept.push(f);
        }
    }
    kept
}

/// `rgba` (`width` × `height`) turned clockwise by `quarters` quarter turns, with its new size;
/// `None` when the size is invalid or the buffer short.
pub fn turn(rgba: &[u8], width: u32, height: u32, quarters: u32) -> Option<(Vec<u8>, u32, u32)> {
    let (w, h) = (width as usize, height as usize);
    let n = w.checked_mul(h).filter(|n| *n > 0 && *n <= MAX_PIXELS)?;
    let src = rgba.get(..n.checked_mul(4)?)?;
    let q = quarters % 4;
    let (nw, nh) = if q.is_multiple_of(2) { (w, h) } else { (h, w) };
    let mut out = vec![0u8; n * 4];
    for y in 0..h {
        for x in 0..w {
            // Clockwise: (x, y) → (h − 1 − y, x); twice: (w − 1 − x, h − 1 − y); three times:
            // (y, w − 1 − x).
            let (tx, ty) = match q {
                0 => (x, y),
                1 => (h - 1 - y, x),
                2 => (w - 1 - x, h - 1 - y),
                _ => (y, w - 1 - x),
            };
            let (from, to) = ((y * w + x) * 4, (ty * nw + tx) * 4);
            if let (Some(s), Some(d)) = (src.get(from..from + 4), out.get_mut(to..to + 4)) {
                d.copy_from_slice(s);
            }
        }
    }
    Some((out, nw as u32, nh as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn white(w: usize, h: usize) -> Vec<u8> {
        vec![255; w * h * 4]
    }

    #[test]
    fn letterbox_centres_the_picture_on_grey() {
        let (t, fit) = letterbox(&white(200, 100), 200, 100, 64).unwrap();
        assert_eq!(t.shape(), [1, 3, 64, 64]);
        assert_eq!(fit.scale, 0.32);
        assert_eq!((fit.pad_x, fit.pad_y), (0.0, 16.0));
        assert_eq!(t[[0, 0, 0, 0]], PAD, "the band above the picture is padding");
        assert_eq!(t[[0, 1, 32, 32]], 1.0, "the picture itself is white");
    }

    #[test]
    fn letterbox_reads_transparent_pixels_as_white() {
        let (t, _) = letterbox(&[0, 0, 0, 0], 1, 1, 4).unwrap();
        assert_eq!(t[[0, 2, 0, 0]], 1.0);
    }

    #[test]
    fn letterbox_refuses_an_empty_picture() {
        assert!(letterbox(&[], 0, 10, 64).is_none());
    }

    fn output(rows: &[[f32; 7]]) -> NdTensor<f32, 3> {
        let mut data = vec![0.0; 7 * rows.len()];
        for (a, r) in rows.iter().enumerate() {
            for (c, v) in r.iter().enumerate() {
                data[c * rows.len() + a] = *v;
            }
        }
        NdTensor::from_data([1, 7, rows.len()], data)
    }

    #[test]
    fn decoding_maps_boxes_back_through_the_letterbox() {
        let fit = Fit { scale: 0.5, pad_x: 0.0, pad_y: 10.0, width: 100.0, height: 60.0 };
        let out = output(&[[20.0, 30.0, 10.0, 4.0, 0.1, 0.9, 0.0], [5.0, 5.0, 2.0, 2.0, 0.1, 0.05, 0.0]]);
        let found = decode(&out, fit, 0.2);
        assert_eq!(found, vec![FoundField { kind: FieldKind::CheckBox, score: 0.9, rect: [30.0, 36.0, 50.0, 44.0] }]);
    }

    #[test]
    fn decoding_survives_malformed_output() {
        let fit = Fit { scale: 1.0, pad_x: 0.0, pad_y: 0.0, width: 10.0, height: 10.0 };
        let nan = output(&[[f32::NAN, 1.0, 1.0, 1.0, f32::NAN, 0.9, 0.0]]);
        assert!(decode(&nan, fit, 0.2).is_empty());
        assert!(decode(&NdTensor::from_data([1, 2, 1], vec![0.0, 0.0]), fit, 0.2).is_empty());
        assert!(decode(&output(&[[5.0; 7]]), Fit { scale: 0.0, ..fit }, 0.2).is_empty());
    }

    #[test]
    fn suppression_keeps_the_stronger_of_overlapping_boxes() {
        let f = |kind, score, rect| FoundField { kind, score, rect };
        let found = vec![
            f(FieldKind::Text, 0.9, [0.0, 0.0, 100.0, 20.0]),
            f(FieldKind::Text, 0.8, [2.0, 0.0, 100.0, 20.0]),
            f(FieldKind::CheckBox, 0.7, [10.0, 2.0, 20.0, 12.0]),
            f(FieldKind::CheckBox, 0.6, [200.0, 0.0, 210.0, 10.0]),
        ];
        let kept = suppress(found, FieldSettings::default());
        assert_eq!(kept.iter().map(|k| k.score).collect::<Vec<_>>(), vec![0.9, 0.6]);
    }

    #[test]
    fn turning_moves_pixels_clockwise() {
        // A 3 × 2 picture with one red pixel at its top right, (2, 0).
        let mut px = white(3, 2);
        px[8..12].copy_from_slice(&[255, 0, 0, 255]);
        // Clockwise, the top right corner goes to the bottom right, the bottom left, the top left.
        for (q, size, at) in [(0, (3, 2), (2, 0)), (1, (2, 3), (1, 2)), (2, (3, 2), (0, 1)), (3, (2, 3), (0, 0))] {
            let (t, w, h) = turn(&px, 3, 2, q).unwrap();
            assert_eq!((w, h), size, "quarter turns {q}");
            let red = (0..(w * h) as usize).find(|i| t[i * 4 + 1] == 0).unwrap();
            assert_eq!((red % w as usize, red / w as usize), at, "quarter turns {q}");
        }
    }

    #[test]
    fn turning_refuses_a_short_buffer() {
        assert!(turn(&[0; 8], 2, 2, 1).is_none());
        assert!(turn(&[], 0, 0, 1).is_none());
    }

    /// Runs only where the model is installed (PDFCRAFT_MODELS): a blank page has no fields.
    #[test]
    fn a_blank_page_has_no_fields_when_the_model_is_installed() {
        let Some(path) = FieldModel::path() else { return };
        let model = FieldModel::load(path).unwrap();
        let found = model.detect(&white(612, 792), 612, 792, FieldSettings::default()).unwrap();
        assert!(found.is_empty(), "{found:?}");
    }
}
