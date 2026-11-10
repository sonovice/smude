//! Platform metadata for independently hosted native navigation-bar actions.
use nsh_core::JsonValue;
use nsh_platform_ios::{Components, NativeComponent};
use objc2::{MainThreadMarker, rc::Retained};
use objc2_ui_kit::UIView;
use std::rc::Rc;
struct Symbol {
    view: Retained<UIView>,
    symbol: String,
}
impl NativeComponent for Symbol {
    fn view(&self) -> &UIView {
        &self.view
    }
    fn bar_button_symbol(&self) -> Option<&str> {
        (!self.symbol.is_empty()).then_some(&self.symbol)
    }
    fn set_prop(&mut self, name: &str, value: Option<&JsonValue>) {
        if name == "symbol" {
            self.symbol = value.and_then(JsonValue::as_str).unwrap_or_default().into();
        }
    }
}
pub fn register(components: &Components) {
    components.register(
        "smude.header-symbol",
        Rc::new(|_| {
            Box::new(Symbol {
                view: UIView::new(MainThreadMarker::new().expect("UIKit main thread")),
                symbol: String::new(),
            })
        }),
    );
}
