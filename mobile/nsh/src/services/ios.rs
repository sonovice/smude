//! Native OS pickers, sharing and the private archive directory. All visible
//! application screens remain Rust NSH; Swift owns the system controllers.
use nsh_core::JsonValue;
use nsh_platform_ios::{ComponentContext, Components, NativeComponent};
use objc2::rc::Retained;
use objc2_ui_kit::UIView;
use std::{
    ffi::{CStr, CString, c_char, c_void},
    rc::Rc,
};

unsafe extern "C" {
    fn smude_services_create(
        context: *mut c_void,
        event: extern "C" fn(*mut c_void, *const c_char),
    ) -> *mut UIView;
    fn smude_services_command(view: *mut UIView, name: *const c_char, json: *const c_char);
    fn smude_services_shutdown(view: *mut UIView);
}

extern "C" fn event(context: *mut c_void, json: *const c_char) {
    if context.is_null() || json.is_null() {
        return;
    }
    // Swift emits only on the main thread and clears its callback synchronously
    // in shutdown before the boxed ComponentContext is released.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let context = unsafe { &*(context as *const ComponentContext) };
        let Ok(json) = unsafe { CStr::from_ptr(json) }.to_str() else {
            return;
        };
        let Ok(data) = serde_json::from_str::<JsonValue>(json) else {
            return;
        };
        if let Some(name) = data["event"].as_str().map(str::to_owned) {
            context.emit(&name, data);
        }
    }));
}

struct Services {
    view: Retained<UIView>,
    _context: Box<ComponentContext>,
}
impl NativeComponent for Services {
    fn view(&self) -> &UIView {
        &self.view
    }
    fn set_prop(&mut self, _: &str, _: Option<&JsonValue>) {}
    fn command(&mut self, name: &str, args: &JsonValue) {
        if let (Ok(name), Ok(json)) = (CString::new(name), CString::new(args.to_string())) {
            unsafe {
                smude_services_command(
                    Retained::as_ptr(&self.view) as *mut UIView,
                    name.as_ptr(),
                    json.as_ptr(),
                );
            }
        }
    }
}
impl Drop for Services {
    fn drop(&mut self) {
        unsafe { smude_services_shutdown(Retained::as_ptr(&self.view) as *mut UIView) }
    }
}

pub fn register(components: &Components) {
    components.register(
        "smude.services",
        Rc::new(|context| {
            let mut context = Box::new(context);
            let view = unsafe {
                Retained::from_raw(smude_services_create(
                    (&mut *context as *mut ComponentContext).cast(),
                    event,
                ))
            }
            .expect("services view allocation");
            Box::new(Services {
                view,
                _context: context,
            })
        }),
    );
}
