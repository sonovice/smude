NSH 0.4.0 UIKit packaging fix

The published nsh-platform-ios crate references ../../nsh-platform-macos/src/speech.rs, which is outside the published package. This copy preserves the upstream crate and includes that exact file from nsh-platform-macos 0.4.0 under src/speech.rs, with its path attribute removed. No runtime/navigation behavior is changed. Remove this patch when upstream packages the shared file. Source: https://github.com/longbridge/nsh/tree/v0.4.0 . Original licenses are retained.

The vendored archive omits upstream TSX example test applications; they are not inputs to this crate. Apache-2.0 license text is included.
