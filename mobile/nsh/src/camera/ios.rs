use nsh_core::JsonValue;
use nsh_platform_ios::{ComponentContext, Components, NativeComponent};
use objc2::rc::Retained;
use objc2_ui_kit::UIView;
use std::{
    ffi::{CStr, c_char, c_void},
    rc::Rc,
};
unsafe extern "C" {
    fn smude_camera_create(
        context: *mut c_void,
        event: extern "C" fn(*mut c_void, *const c_char),
    ) -> *mut UIView;
    fn smude_camera_command(view: *mut UIView, command: *const c_char, json: *const c_char);
    fn smude_camera_shutdown(view: *mut UIView);
    fn smude_viewer_policy_create() -> *mut UIView;
    fn smude_viewer_policy_active(view: *mut UIView, active: bool);
    fn smude_viewer_policy_shutdown(view: *mut UIView);
}
extern "C" fn event(context: *mut c_void, json: *const c_char) {
    if context.is_null() || json.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let context = unsafe { &*(context as *const ComponentContext) };
        let data = serde_json::from_str::<JsonValue>(
            unsafe { CStr::from_ptr(json) }.to_str().unwrap_or("{}"),
        )
        .unwrap_or_default();
        let name = data["event"].as_str().unwrap_or("status").to_owned();
        context.emit(&name, data);
    }));
}
struct Camera {
    view: Retained<UIView>,
    _context: Box<ComponentContext>,
}
impl NativeComponent for Camera {
    fn view(&self) -> &UIView {
        &self.view
    }
    fn set_prop(&mut self, name: &str, value: Option<&JsonValue>) {
        if name == "active" {
            self.command(
                if value.and_then(JsonValue::as_bool).unwrap_or(false) {
                    "activate"
                } else {
                    "deactivate"
                },
                &JsonValue::Null,
            )
        }
    }
    fn command(&mut self, name: &str, args: &JsonValue) {
        if let (Ok(name), Ok(json)) = (
            std::ffi::CString::new(name),
            std::ffi::CString::new(args.to_string()),
        ) {
            unsafe {
                smude_camera_command(
                    Retained::as_ptr(&self.view) as *mut UIView,
                    name.as_ptr(),
                    json.as_ptr(),
                )
            }
        }
    }
}
impl Drop for Camera {
    fn drop(&mut self) {
        unsafe { smude_camera_shutdown(Retained::as_ptr(&self.view) as *mut UIView) }
    }
}
pub fn register(components: &Components) {
    components.register(
        "smude.viewer-gestures",
        Rc::new(|_| {
            Box::new(ViewerPolicy(
                unsafe { Retained::from_raw(smude_viewer_policy_create()) }
                    .expect("gesture policy view"),
            ))
        }),
    );
    components.register(
        "smude.camera",
        Rc::new(|context| {
            let mut context = Box::new(context);
            let view = unsafe {
                Retained::from_raw(smude_camera_create(
                    (&mut *context as *mut ComponentContext).cast(),
                    event,
                ))
            }
            .expect("camera view allocation");
            Box::new(Camera {
                view,
                _context: context,
            })
        }),
    );
}

struct ViewerPolicy(Retained<UIView>);
impl NativeComponent for ViewerPolicy {
    fn view(&self) -> &UIView {
        &self.0
    }
    fn set_prop(&mut self, name: &str, value: Option<&JsonValue>) {
        if name == "active" {
            unsafe {
                smude_viewer_policy_active(
                    Retained::as_ptr(&self.0) as *mut UIView,
                    value.and_then(JsonValue::as_bool).unwrap_or(false),
                )
            }
        }
    }
}
impl Drop for ViewerPolicy {
    fn drop(&mut self) {
        unsafe { smude_viewer_policy_shutdown(Retained::as_ptr(&self.0) as *mut UIView) }
    }
}
