// SPDX-FileCopyrightText: 2026 Wyss Center for Bio and Neuroengineering
// SPDX-License-Identifier: Apache-2.0

use serde_json::{json, Value};

/// The isotropic registration grid.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GridChoice {
    #[default]
    OneMm,
    TwoMm,
}

impl GridChoice {
    pub fn mm(self) -> f64 {
        match self {
            GridChoice::OneMm => 1.0,
            GridChoice::TwoMm => 2.0,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            GridChoice::OneMm => "1mm",
            GridChoice::TwoMm => "2mm",
        }
    }
}

/// What the page can choose.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Options {
    pub grid: GridChoice,
}

impl Options {
    pub fn to_json(self) -> Value {
        json!({ "grid": self.grid.name(), "iso_mm": self.grid.mm() })
    }
}

/// Reads `{"grid": "1mm" | "2mm"}`; an empty text or a missing key is the default (1 mm).
pub fn parse_options(text: &str) -> Result<Options, String> {
    if text.trim().is_empty() {
        return Ok(Options::default());
    }
    let value: Value =
        serde_json::from_str(text).map_err(|e| format!("options are not valid JSON: {e}"))?;
    let Value::Object(map) = value else {
        return Err("options must be a JSON object".into());
    };
    let mut options = Options::default();
    for (key, v) in &map {
        match key.as_str() {
            "grid" => {
                options.grid = match v.as_str() {
                    Some("1mm") => GridChoice::OneMm,
                    Some("2mm") => GridChoice::TwoMm,
                    _ => return Err(format!("grid must be \"1mm\" or \"2mm\", not {v}")),
                }
            }
            other => return Err(format!("unknown option {other:?}")),
        }
    }
    Ok(options)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_values() {
        assert_eq!(parse_options("").unwrap().grid, GridChoice::OneMm);
        assert_eq!(parse_options("{}").unwrap().grid, GridChoice::OneMm);
        assert_eq!(
            parse_options(r#"{"grid":"2mm"}"#).unwrap().grid,
            GridChoice::TwoMm
        );
        assert_eq!(GridChoice::TwoMm.mm(), 2.0);
        assert!(parse_options(r#"{"grid":"3mm"}"#).is_err());
        assert!(parse_options(r#"{"grid":2}"#).is_err());
        assert!(parse_options(r#"{"other":1}"#).is_err());
        assert!(parse_options("[]").is_err());
        assert!(parse_options("{").is_err());
    }
}
