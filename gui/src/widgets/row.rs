use eframe::egui;
use eframe::egui::{Align, InnerResponse, Layout, Ui, UiBuilder, vec2};

/// A horizontal row of widgets, each centered on the tallest, as a flex
/// row with `items-center` does.
///
/// `ui.horizontal` centers each widget on the row as tall as it is when the
/// widget comes, so a label before a taller button sits too high. This row
/// remembers its height from the last frame instead, and lays itself out
/// again (in a frame nobody sees) when that height changes.
///
/// Inside, `ui.with_layout(Layout::right_to_left(Align::Center), …)` puts
/// the rest at the right end, centered as well.
pub fn centered_row<R>(ui: &mut Ui, add_contents: impl FnOnce(&mut Ui) -> R) -> InnerResponse<R> {
    let id = ui.next_auto_id().with("centered_row");
    let known = ui.data(|data| data.get_temp::<f32>(id));
    let height = known.unwrap_or(0.0);
    let rect = egui::Rect::from_min_size(ui.cursor().min, vec2(ui.available_width(), height));
    let mut child = ui.new_child(
        UiBuilder::new()
            .max_rect(rect)
            .layout(Layout::left_to_right(Align::Center)),
    );
    let inner = add_contents(&mut child);

    let content = child.min_rect();
    if known != Some(content.height()) {
        ui.data_mut(|data| data.insert_temp(id, content.height()));
        ui.ctx().request_discard("a row's height changed");
    }
    let rect = egui::Rect::from_min_max(
        rect.min,
        egui::pos2(content.max.x.max(rect.min.x), rect.top() + content.height()),
    );
    let response = ui.allocate_rect(rect, egui::Sense::hover());
    InnerResponse::new(inner, response)
}
