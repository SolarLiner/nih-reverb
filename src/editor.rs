use std::sync::Arc;

// Copyright (c) 2022 solarliner
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT
use nih_plug::prelude::*;
use nih_plug_vizia::{
    assets, create_vizia_editor,
    vizia::prelude::*,
    widgets::{GenericUi, ResizeHandle},
    ViziaState,
};

use crate::DelayParams;

/// VIZIA uses points instead of pixels for text
const POINT_SCALE: f32 = 0.75;

#[derive(Lens)]
pub(crate) struct DelayEditor {
    params: Arc<DelayParams>,
}

impl Model for DelayEditor {}

impl DelayEditor {
    pub fn default_state() -> Arc<ViziaState> {
        ViziaState::from_size(380, 300)
    }

    pub fn create(
        params: Arc<DelayParams>,
        editor_state: Arc<ViziaState>,
    ) -> Option<Box<dyn Editor>> {
        create_vizia_editor(editor_state, move |cx, _| {
            DelayEditor {
                params: params.clone(),
            }
            .build(cx);
            ResizeHandle::new(cx);
            VStack::new(cx, |cx| {
                Label::new(cx, "NIH Reverb")
                    .font(assets::NOTO_SANS_THIN)
                    .font_size(40.0 * POINT_SCALE)
                    .height(Pixels(50.0))
                    .child_top(Stretch(1.0))
                    .child_bottom(Pixels(10.0))
                    .right(Percentage(12.0));
                ScrollView::new(cx, 0.0, 0.0, false, true, |cx| {
                    GenericUi::new(cx, DelayEditor::params)
                        .width(Percentage(100.0))
                        .height(Auto)
                        .child_top(Pixels(5.0))
                        .child_right(Pixels(10.0));
                })
                .width(Percentage(100.0));
            })
            .width(Percentage(100.0))
            .row_between(Pixels(0.0))
            .child_left(Stretch(1.0))
            .child_right(Stretch(1.0));
        })
    }
}
