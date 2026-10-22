//! Prepare a form ▸ field detection with a trained model ([`pdfcraft_ocr::fields`], FFDNet from
//! CommonForms) when the user has installed one. It replaces the built-in detector for every
//! page: the model reads a picture of the page, so it also finds the fields of pages the built-in
//! detector cannot read (text printed as outlines, check boxes that are font glyphs, labelled
//! table cells, forms drawn sideways). Without the model, or when it fails on a page, that page
//! falls back to the built-in detector.
//!
//! The page is rendered as displayed. The model only reads upright forms, so when it finds little
//! on the page as displayed it also reads the page turned a quarter each way and keeps the
//! orientation that finds the most. Fields are named from their labels in that upright frame
//! ([`pdfcraft_forms::detect::place`]), mapped back to user space and turned with the form
//! (`/MK /R`), so typed text reads upright.

use std::sync::{Arc, Mutex};

use pdfcraft_forms::detect::{Candidate, Kind, Word, place};
use pdfcraft_ocr::fields::{FieldKind, FieldModel, FieldSettings, FoundField, turn};
use pdfcraft_render::{PageRenderer, PageText, RenderRequest};

use crate::scan_fields::{Frame, MAX_SIDE, SCALE, corners};
use crate::{DocId, Session};

/// The page as displayed finding at least this many fields reads upright, and the turned
/// readings are skipped (each costs a model run).
const UPRIGHT_ENOUGH: usize = 4;
/// The quarter turns (clockwise) tried after the page as displayed: right, then left.
const TURNS: [u32; 2] = [1, 3];
/// Fields smaller than this (points, either side) are specks.
const MIN_SIDE: f64 = 2.0;

/// The installed field-detection model, loaded once; `None` when none is installed, it fails to
/// load, or `PDFCRAFT_FIELD_MODEL=0` turns it off (which keeps the built-in detector's tests
/// meaningful on a machine with the model installed). A failure is not remembered, so a model
/// installed later is picked up.
pub(crate) fn model() -> Option<Arc<FieldModel>> {
    if std::env::var_os("PDFCRAFT_FIELD_MODEL").is_some_and(|v| v == "0") {
        return None;
    }
    static MODEL: Mutex<Option<Arc<FieldModel>>> = Mutex::new(None);
    let mut slot = MODEL.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(m) = slot.as_ref() {
        return Some(m.clone());
    }
    let m = Arc::new(FieldModel::find().ok()?);
    *slot = Some(m.clone());
    Some(m)
}

/// A rectangle (y up) in a frame `w` × `h` points, turned clockwise `quarters` times with the
/// frame; the turned frame is `h` × `w` for an odd count.
fn turn_rect(r: [f64; 4], w: f64, h: f64, quarters: u32) -> [f64; 4] {
    let (mut r, mut w, mut h) = (r, w, h);
    for _ in 0..quarters % 4 {
        // A clockwise quarter turn, y up: (x, y) → (y, w − x), and the frame becomes h × w.
        r = [r[1], w - r[2], r[3], w - r[0]];
        (w, h) = (h, w);
    }
    r
}

fn kind(k: FieldKind) -> Kind {
    match k {
        FieldKind::CheckBox => Kind::CheckBox,
        // A signature line is a text field until signature fields are proposed (#3502's list).
        FieldKind::Text | FieldKind::Signature => Kind::Text,
    }
}

impl Session {
    /// The fields the installed `model` finds on `page` of `id`, in user space, and how far each
    /// turns to read upright; `None` when the page cannot be rendered or the model fails, so the
    /// caller falls back to the built-in detector. `text` is the page's text layer, `existing`
    /// its widget rectangles, `taken` the names in use.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn detect_with_model(
        &self,
        id: DocId,
        r: &mut PageRenderer,
        model: &FieldModel,
        page: usize,
        text: Option<&PageText>,
        existing: &[[f64; 4]],
        taken: &[String],
    ) -> Option<(Vec<Candidate>, i64)> {
        let info = self.get(id).and_then(|d| d.info.pages.get(page))?;
        let frame = Frame(info);
        let (w, h) = (f64::from(info.width), f64::from(info.height));
        if !(w.is_finite() && h.is_finite() && w > 0.0 && h > 0.0) {
            return None;
        }
        let scale = SCALE.min(MAX_SIDE / info.width.max(info.height).max(1.0));
        let shot = r.render(RenderRequest { page, scale, ..Default::default() });
        if shot.error.is_some() || shot.width == 0 || shot.height == 0 {
            return None;
        }
        let px_per_pt = f64::from(shot.width) / w;
        if !px_per_pt.is_finite() || px_per_pt <= 0.0 {
            return None;
        }
        let settings = FieldSettings::default();
        // The page as displayed, then turned each way while it finds little.
        let mut best: (u32, Vec<FoundField>) = (0, model.detect(&shot.rgba, shot.width, shot.height, settings).ok()?);
        if best.1.len() < UPRIGHT_ENOUGH {
            for q in TURNS {
                let Some((px, tw, th)) = turn(&shot.rgba, shot.width, shot.height, q) else { continue };
                let Ok(found) = model.detect(&px, tw, th, settings) else { continue };
                if found.len() > best.1.len() {
                    best = (q, found);
                }
            }
        }
        let (q, found) = best;
        // Everything is named in the upright frame (points, y up), where "to the left" and
        // "above" mean what they mean to the reader.
        let (uw, uh) = if q.is_multiple_of(2) { (w, h) } else { (h, w) };
        let fields: Vec<(Kind, [f64; 4])> = found
            .iter()
            .map(|f| {
                // Turned-picture pixels (y down) → upright points (y up).
                let [x0, y0, x1, y1] = f.rect.map(f64::from);
                (kind(f.kind), [x0 / px_per_pt, uh - y1 / px_per_pt, x1 / px_per_pt, uh - y0 / px_per_pt])
            })
            .filter(|(_, r)| r[2] - r[0] >= MIN_SIDE && r[3] - r[1] >= MIN_SIDE)
            .collect();
        let mut words: Vec<Word> = text
            .map(|t| crate::js::words_with(t, |x, y| [x, y]))
            .unwrap_or_default()
            .into_iter()
            .filter_map(|(text, v)| Some(Word { text, rect: frame.view_box([[v[0], v[1]], [v[2], v[3]]])? }))
            .collect();
        if words.is_empty() && !fields.is_empty() {
            words = self.recognized_words(id, page, &frame);
        }
        let words: Vec<Word> = words.into_iter().map(|wd| Word { rect: turn_rect(wd.rect, w, h, q), ..wd }).collect();
        let existing: Vec<[f64; 4]> = existing.iter().filter_map(|r| frame.user_box(corners(*r))).map(|r| turn_rect(r, w, h, q)).collect();
        let placed = place(&fields, &words, &existing, taken);
        // Back: upright frame → the page as displayed (the remaining quarter turns) → user space.
        let back = (4 - q % 4) % 4;
        let out = placed.into_iter().map(|c| Candidate { rect: frame.user_rect(turn_rect(c.rect, uw, uh, back)), ..c }).collect();
        let turn = (i64::from(info.rotation % 360) + i64::from(q) * 90) % 360;
        Some((out, turn))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quarter_turn_moves_the_top_left_corner_to_the_top_right() {
        // A 10 × 10 box at the top left of a 100 × 50 frame (y up).
        let r = [0.0, 40.0, 10.0, 50.0];
        assert_eq!(turn_rect(r, 100.0, 50.0, 1), [40.0, 90.0, 50.0, 100.0]);
    }

    #[test]
    fn four_quarter_turns_come_back() {
        let r = [12.0, 3.0, 40.0, 9.5];
        for q in 0..4u32 {
            let (uw, uh) = if q.is_multiple_of(2) { (100.0, 50.0) } else { (50.0, 100.0) };
            let there = turn_rect(r, 100.0, 50.0, q);
            assert_eq!(turn_rect(there, uw, uh, (4 - q) % 4), r, "quarters {q}");
        }
    }

    #[test]
    fn signature_lines_become_text_fields() {
        assert_eq!(kind(FieldKind::Signature), Kind::Text);
        assert_eq!(kind(FieldKind::CheckBox), Kind::CheckBox);
    }
}
