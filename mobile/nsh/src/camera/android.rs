//! Android's direct-buffer bridge. Java physically rotates and crops the analysis
//! image to the shared CameraX ViewPort before calling this function.
use jni::{
    JNIEnv,
    objects::{JByteBuffer, JClass},
    sys::{jint, jstring},
};

#[unsafe(no_mangle)]
pub extern "system" fn Java_com_gridnotelabs_smude_design_CameraInference_detect(
    env: JNIEnv<'_>,
    _class: JClass<'_>,
    buffer: JByteBuffer<'_>,
    width: jint,
    height: jint,
    stride: jint,
    clockwise_turns: jint,
) -> jstring {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let (Ok(width), Ok(height), Ok(stride)) = (
            usize::try_from(width),
            usize::try_from(height),
            usize::try_from(stride),
        ) else {
            return serde_json::json!({"error":"Invalid camera image"});
        };
        let (Ok(ptr), Ok(capacity)) = (
            env.get_direct_buffer_address(&buffer),
            env.get_direct_buffer_capacity(&buffer),
        ) else {
            return serde_json::json!({"error":"Invalid camera buffer"});
        };
        if ptr.is_null()
            || width > 2048
            || height > 2048
            || capacity > 32 * 1024 * 1024
            || stride
                .checked_mul(height)
                .is_none_or(|length| length > capacity)
        {
            return serde_json::json!({"error":"Invalid camera buffer"});
        }
        // JNI owns this buffer for the duration of the synchronous call; the
        // Java analysis executor serializes its reuse and closes each frame.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, capacity) };
        super::inference::detect_oriented(
            bytes,
            width,
            height,
            stride,
            false,
            clockwise_turns as u32,
        )
    }))
    .unwrap_or_else(|_| serde_json::json!({"error":"Page detection failed"}));
    env.new_string(result.to_string())
        .map(|s| s.into_raw())
        .unwrap_or(std::ptr::null_mut())
}
