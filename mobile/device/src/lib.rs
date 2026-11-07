//! OS image codecs, camera and document export. Image processing stays in Rust.
#[cfg(any(target_os = "ios", target_os = "android"))]
use tauri::Manager;
use tauri::Runtime;

#[cfg(target_os = "ios")]
tauri::ios_plugin_binding!(init_plugin_smude_device);

#[cfg(any(target_os = "ios", target_os = "android"))]
struct Device<R: Runtime>(tauri::plugin::PluginHandle<R>);

pub fn call<R: Runtime>(
    app: &tauri::AppHandle<R>,
    command: &str,
    args: serde_json::Value,
) -> Result<serde_json::Value, String> {
    #[cfg(any(target_os = "ios", target_os = "android"))]
    return app
        .state::<Device<R>>()
        .0
        .run_mobile_plugin(command, args)
        .map_err(|e| e.to_string());
    #[cfg(not(any(target_os = "ios", target_os = "android")))]
    {
        let _ = (app, command, args);
        Err("Diese Funktion benötigt Android oder iOS.".into())
    }
}

pub fn init<R: Runtime>() -> tauri::plugin::TauriPlugin<R> {
    tauri::plugin::Builder::<R, ()>::new("smude-device")
        .setup(|_app, _api| {
            #[cfg(target_os = "android")]
            let handle =
                _api.register_android_plugin("com.gridnotelabs.smude.mobile", "DevicePlugin")?;
            #[cfg(target_os = "ios")]
            let handle = _api.register_ios_plugin(init_plugin_smude_device)?;
            #[cfg(any(target_os = "ios", target_os = "android"))]
            _app.manage(Device(handle));
            Ok(())
        })
        .build()
}
