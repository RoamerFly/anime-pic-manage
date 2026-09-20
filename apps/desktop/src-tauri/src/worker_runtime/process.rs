use crate::worker_runtime::types::WorkerRuntimeError;
use std::process::{Command, Stdio};

pub fn uv_available() -> bool {
    let mut command = Command::new("uv");
    command
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    configure_background_command(&mut command);
    command.status().is_ok_and(|status| status.success())
}

/// A structured Worker error is a valid response and must not tear down the
/// model process.  Transport/protocol errors indicate that the stream can no
/// longer be trusted and require a clean restart on the next request.
pub fn should_clear_process(error: &WorkerRuntimeError) -> bool {
    !matches!(error, WorkerRuntimeError::WorkerResponse { .. })
}

#[cfg(windows)]
const CREATE_NO_WINDOW: u32 = 0x0800_0000;
/// Keep inference below the desktop and the user's other apps so a long scan
/// never makes the machine feel frozen.
#[cfg(windows)]
const BELOW_NORMAL_PRIORITY_CLASS: u32 = 0x0000_4000;

/// Return the platform-specific creation flags for application-owned
/// background processes.  Making this decision explicit and testable keeps a
/// future non-Windows refactor from accidentally importing the Windows API.
pub fn background_creation_flags() -> Option<u32> {
    #[cfg(windows)]
    {
        Some(CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS)
    }
    #[cfg(not(windows))]
    {
        None
    }
}

/// Configure a Worker-related child process so Windows does not allocate a
/// console window for it, and lower its CPU priority unless the user asked for
/// full speed.  On other platforms this is deliberately a no-op.
///
/// Keeping this in one helper is important: both the long-lived Worker and the
/// short-lived `uv --version` probe are application-owned background processes.
/// The latter used to be easy to overlook when adding a Windows-only flag.
pub fn configure_background_command_with(command: &mut Command, below_normal: bool) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let flags = if below_normal {
            CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS
        } else {
            CREATE_NO_WINDOW
        };
        command.creation_flags(flags);
    }

    #[cfg(not(windows))]
    let _ = (command, below_normal);
}

/// Default background policy: visible to nobody, but polite to the desktop.
pub fn configure_background_command(command: &mut Command) {
    configure_background_command_with(command, true);
}
