use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{Manager, RunEvent, State};

/// Windows single-instance guard.
///
/// Hand-rolled instead of `tauri-plugin-single-instance` on purpose: a plugin
/// is installed on the `Builder`, but `run()` spawns the Python sidecar
/// *before* the builder exists. A duplicate launch would therefore boot a full
/// sidecar, get rejected, and orphan that interpreter. This guard runs before
/// any side effect, so a rejected launch leaves nothing behind.
///
/// The two processes also cannot share WebView2's user-data folder: the second
/// one either panics (`HRESULT 0x8007139F`) or attaches its webview to the first
/// process' browser group and turns into a white shell when that group exits
/// (diagnosed 2026-11-25). Blocking at process start is the only reliable fix.
#[cfg(windows)]
mod single_instance {
    use std::ffi::c_void;
    use std::sync::OnceLock;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn CreateMutexW(attrs: *const c_void, initial_owner: i32, name: *const u16) -> *mut c_void;
        fn CloseHandle(handle: *mut c_void) -> i32;
        fn SetLastError(code: u32);
        fn GetLastError() -> u32;
    }

    #[link(name = "user32")]
    unsafe extern "system" {
        fn FindWindowW(class: *const u16, title: *const u16) -> *mut c_void;
        fn ShowWindow(hwnd: *mut c_void, cmd: i32) -> i32;
        fn IsIconic(hwnd: *mut c_void) -> i32;
        fn SetForegroundWindow(hwnd: *mut c_void) -> i32;
    }

    const ERROR_ALREADY_EXISTS: u32 = 183;
    const SW_SHOW: i32 = 5;
    const SW_RESTORE: i32 = 9;

    /// Held for the whole process lifetime: closing the handle would release
    /// the lock while the app is still running. The OS reclaims it on exit.
    static GUARD: OnceLock<usize> = OnceLock::new();

    enum AcquireResult {
        Acquired,
        Busy,
        Unavailable,
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    fn acquire() -> AcquireResult {
        // `Local\` (session-scoped) rather than `Global\`: two logged-in users
        // must be able to run their own copy.
        let name = wide(r"Local\ai-text-detector-single-instance");
        // SAFETY: plain Win32 calls with NUL-terminated UTF-16 literals; the
        // returned handle is leaked on purpose (see GUARD).
        unsafe {
            // CreateMutexW only writes the last error on the "already exists"
            // path, so a stale value from an earlier call would make every
            // launch look like a duplicate.
            SetLastError(0);
            let handle = CreateMutexW(std::ptr::null(), 0, name.as_ptr());
            let err = GetLastError();
            if handle.is_null() {
                // Fail open: the guard must never be the reason the app cannot
                // start (restricted token, AV interference, ...).
                eprintln!(
                    "[shell] single-instance guard unavailable (error {err}); continuing unguarded"
                );
                return AcquireResult::Unavailable;
            }
            if err == ERROR_ALREADY_EXISTS {
                // This is only a borrowed handle to the existing object. Close
                // it before retrying; the owner keeps the named object alive
                // until its own process has fully exited.
                let _ = CloseHandle(handle);
                return AcquireResult::Busy;
            }
            let _ = GUARD.set(handle as usize);
            AcquireResult::Acquired
        }
    }

    /// Best-effort focus of the already-running window.
    fn wake_existing() -> bool {
        let class = wide("Tauri Window");
        let title = wide("AI Text Detector");
        // SAFETY: null returns are checked before any follow-up call.
        unsafe {
            let mut hwnd = FindWindowW(class.as_ptr(), title.as_ptr());
            if hwnd.is_null() {
                // Title-only fallback: the class name is a Tauri implementation
                // detail and may change between releases.
                hwnd = FindWindowW(std::ptr::null(), title.as_ptr());
            }
            if hwnd.is_null() {
                return false;
            }
            if IsIconic(hwnd) != 0 {
                ShowWindow(hwnd, SW_RESTORE);
            } else {
                ShowWindow(hwnd, SW_SHOW);
            }
            SetForegroundWindow(hwnd);
            true
        }
    }

    /// Must be the very first statement of `run()`.
    pub fn enforce() {
        if std::env::var_os("AIGC_ALLOW_MULTI").is_some() {
            eprintln!("[shell] AIGC_ALLOW_MULTI is set: single-instance guard skipped");
            return;
        }
        // Window teardown + sidecar kill can outlive the visible window. During
        // this quiet window we only retry the mutex and never call FindWindow:
        // a closing window still exists briefly and must not be mistaken for a
        // healthy instance that should be woken.
        let mut saw_busy = false;
        for attempt in 0..40 {
            match acquire() {
                AcquireResult::Acquired => {
                    if saw_busy {
                        eprintln!("[shell] previous instance finished; relaunch continues");
                    }
                    return;
                }
                AcquireResult::Unavailable => return,
                AcquireResult::Busy => saw_busy = true,
            }

            // 1.5s is deliberately conservative. A genuine double-click on a
            // live app pays this small delay once; a rapid relaunch instead
            // takes over as soon as the old mutex is released.
            if attempt < 15 {
                std::thread::sleep(std::time::Duration::from_millis(100));
                continue;
            }
            if wake_existing() {
                eprintln!("[shell] another instance is running; exiting duplicate");
                // Nothing has been spawned yet, so this leaves no orphan sidecar.
                std::process::exit(0);
            }
            std::thread::sleep(std::time::Duration::from_millis(150));
        }
        eprintln!(
            "[shell] another instance is still running without a recoverable window; exiting"
        );
        std::process::exit(0);
    }
}

#[cfg(windows)]
mod console_visibility {
    use std::ffi::c_void;
    use std::os::windows::process::CommandExt;
    use std::path::PathBuf;
    use std::process::{Child, Command};
    use std::sync::{Mutex, OnceLock};

    #[derive(serde::Deserialize, serde::Serialize, Default)]
    struct ConsolePreference {
        show_console: Option<bool>,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetConsoleWindow() -> *mut c_void;
    }

    #[link(name = "user32")]
    unsafe extern "system" {
        fn ShowWindow(hwnd: *mut c_void, cmd: i32) -> i32;
    }

    const SW_HIDE: i32 = 0;
    const CREATE_NEW_CONSOLE: u32 = 0x0000_0010;
    static DEBUG_LOG_CHILD: OnceLock<Mutex<Option<Child>>> = OnceLock::new();

    fn debug_log_child() -> &'static Mutex<Option<Child>> {
        DEBUG_LOG_CHILD.get_or_init(|| Mutex::new(None))
    }

    fn settings_path() -> Option<PathBuf> {
        let base = std::env::var_os("LOCALAPPDATA")?;
        Some(
            PathBuf::from(base)
                .join("com.aitextdetector.app")
                .join("console-visibility.json"),
        )
    }

    fn preferred_visible() -> bool {
        if std::env::var("AIGC_SHOW_CONSOLE").ok().as_deref() == Some("1") {
            return true;
        }
        let Some(path) = settings_path() else {
            return false;
        };
        let Ok(raw) = std::fs::read_to_string(path) else {
            return false;
        };
        serde_json::from_str::<ConsolePreference>(&raw)
            .ok()
            .and_then(|pref| pref.show_console)
            .unwrap_or(false)
    }

    fn write_preference(visible: bool) -> Result<(), String> {
        let path = settings_path().ok_or("LOCALAPPDATA is unavailable")?;
        let parent = path.parent().ok_or("invalid console settings path")?;
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("cannot create console settings directory: {e}"))?;
        let raw = serde_json::to_string_pretty(&ConsolePreference {
            show_console: Some(visible),
        })
        .map_err(|e| format!("cannot serialize console preference: {e}"))?;
        std::fs::write(path, raw).map_err(|e| format!("cannot write console preference: {e}"))
    }

    fn hide_attached_console() {
        // The Tauri debug executable is console-subsystem. Keep that window
        // hidden; all visible developer logs are handled by a separate child.
        unsafe {
            let hwnd = GetConsoleWindow();
            if !hwnd.is_null() {
                ShowWindow(hwnd, SW_HIDE);
            }
        }
    }

    fn stop_helper() {
        if let Ok(mut guard) = debug_log_child().lock() {
            if let Some(mut child) = guard.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }

    fn helper_running() -> bool {
        let Ok(mut guard) = debug_log_child().lock() else {
            return false;
        };
        let Some(child) = guard.as_mut() else {
            return false;
        };
        match child.try_wait() {
            Ok(None) => true,
            _ => {
                *guard = None;
                false
            }
        }
    }

    fn start_helper() -> Result<(), String> {
        stop_helper();
        #[cfg(debug_assertions)]
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .ok_or("manifest has no parent")?
            .to_path_buf();
        #[cfg(not(debug_assertions))]
        let root = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .ok_or("installed executable has no parent")?;
        let root_ps = root.to_string_lossy().replace('\'', "''");
        let command = r#"
$Host.UI.RawUI.WindowTitle = 'AI Text Detector Debug Log'
Write-Host 'AI Text Detector debug log viewer'
Write-Host 'Closing this window does not close the application.'
Write-Host ''
$root = '__ROOT__'
$logs = Join-Path $root 'logs'
while ($true) {
  $latest = Get-ChildItem -LiteralPath $logs -Filter 'sidecar-*.log' -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime |
    Select-Object -Last 1
  if ($latest) {
    Get-Content -LiteralPath $latest.FullName -Tail 120 -Wait
  } else {
    Write-Host 'Waiting for sidecar log...'
    Start-Sleep -Seconds 2
  }
}
"#
        .replace("__ROOT__", &root_ps);
        let mut cmd = Command::new("powershell.exe");
        cmd.current_dir(&root)
            .args([
                "-NoLogo",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                &command,
            ])
            .creation_flags(CREATE_NEW_CONSOLE);
        let child = cmd
            .spawn()
            .map_err(|e| format!("cannot start debug log window: {e}"))?;
        *debug_log_child()
            .lock()
            .map_err(|_| "debug log state lock poisoned".to_string())? = Some(child);
        Ok(())
    }

    pub fn apply_saved_preference() {
        hide_attached_console();
        if preferred_visible() {
            if let Err(e) = start_helper() {
                eprintln!("[shell] {e}");
            }
        }
    }

    pub fn status() -> serde_json::Value {
        serde_json::json!({
            "supported": cfg!(debug_assertions),
            "visible": helper_running(),
            "preference": preferred_visible(),
        })
    }

    pub fn set_visible(visible: bool) -> Result<serde_json::Value, String> {
        write_preference(visible)?;
        hide_attached_console();
        if visible {
            start_helper()?;
        } else {
            stop_helper();
        }
        Ok(status())
    }

    pub fn shutdown() {
        stop_helper();
        hide_attached_console();
    }
}

#[cfg(not(windows))]
mod console_visibility {
    pub fn apply_saved_preference() {}

    pub fn status() -> serde_json::Value {
        serde_json::json!({
            "supported": false,
            "visible": false,
            "preference": false,
        })
    }

    pub fn set_visible(_visible: bool) -> Result<serde_json::Value, String> {
        Err("console visibility is only implemented on Windows".to_string())
    }

    pub fn shutdown() {}
}

/// The project only ships for Windows; other targets keep a no-op so a stray
/// `cargo check` on another host still compiles.
#[cfg(not(windows))]
mod single_instance {
    pub fn enforce() {}
}

/// Shared sidecar bookkeeping: child handle for kill-on-exit, handshake result.
struct SidecarState {
    child: Mutex<Option<Child>>,
    conn: Arc<OnceLock<(u16, String)>>,
    startup_error: Arc<OnceLock<String>>,
}

struct SidecarLaunch {
    child: Option<Child>,
    reader: Option<BufReader<std::process::ChildStdout>>,
    token: String,
    error: Option<String>,
}

/// Invoke target for the WebView: returns the WS endpoint once handshake landed.
#[tauri::command]
fn sidecar_info(state: State<SidecarState>) -> Result<serde_json::Value, String> {
    if let Some((port, token)) = state.conn.get() {
        return Ok(serde_json::json!({ "port": port, "token": token }));
    }
    if let Some(error) = state.startup_error.get() {
        return Err(error.clone());
    }
    Err("sidecar handshake not ready".to_string())
}

/// round-30: write the exported log to the path picked through the dialog
/// plugin. The webview cannot touch the filesystem, and pulling in the fs
/// plugin (with its whole scope model) for one button is not worth it.
#[tauri::command]
fn save_text_file(path: String, contents: String) -> Result<u64, String> {
    let bytes = contents.as_bytes();
    std::fs::write(&path, bytes).map_err(|e| format!("{path}: {e}"))?;
    Ok(bytes.len() as u64)
}

/// Open an official/mirror download page in the user's default browser.
///
/// The command is deliberately allowlisted: the webview never gets a generic
/// "open any URL/path" primitive that could be abused if a protocol payload is
/// damaged. Driver and runtime installers stay user-driven, outside the shell.
fn validated_external_url(url: &str) -> Result<url::Url, String> {
    const HOSTS: &[&str] = &[
        "github.com",
        "pytorch.org",
        "download.pytorch.org",
        "pypi.org",
        "files.pythonhosted.org",
        "pypi.tuna.tsinghua.edu.cn",
        "huggingface.co",
        "hf-mirror.com",
        "www.python.org",
        "www.nvidia.com",
        "www.nvidia.cn",
        "www.microsoft.com",
        "learn.microsoft.com",
        "aka.ms",
    ];
    let parsed = url::Url::parse(url).map_err(|e| format!("invalid download URL: {e}"))?;
    if parsed.scheme() != "https" {
        return Err("only https download links are allowed".to_string());
    }
    let host = parsed
        .host_str()
        .ok_or_else(|| "download URL has no host".to_string())?;
    if !HOSTS.contains(&host) {
        return Err(format!("download host is not allowlisted: {host}"));
    }
    Ok(parsed)
}

#[tauri::command]
fn open_external_url(url: String) -> Result<(), String> {
    let parsed = validated_external_url(&url)?;
    open::that(parsed.as_str()).map_err(|e| format!("cannot open browser: {e}"))
}

#[tauri::command]
fn console_visibility_status() -> serde_json::Value {
    console_visibility::status()
}

#[tauri::command]
fn set_console_visibility(visible: bool) -> Result<serde_json::Value, String> {
    console_visibility::set_visible(visible)
}

fn new_sidecar_token() -> String {
    format!(
        "{:016x}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock before epoch")
            .as_nanos() as u64
            ^ u64::from(std::process::id())
    )
}

// Keep the compile-time source path out of the release binary. A runtime
// `cfg!` branch is not enough: `env!` would still be embedded even when the
// branch is never executed.
#[cfg(debug_assertions)]
fn sidecar_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("manifest without parent")
        .to_path_buf()
}

#[cfg(not(debug_assertions))]
fn sidecar_root() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| PathBuf::from("."))
}

fn python_candidates(root: &std::path::Path) -> Vec<PathBuf> {
    if cfg!(debug_assertions) {
        vec![
            root.join(".venv").join("Scripts").join("python.exe"),
            root.join("runtime").join("python").join("python.exe"),
        ]
    } else {
        vec![root.join("runtime").join("python").join("python.exe")]
    }
}

fn write_startup_breadcrumb(root: &std::path::Path, detail: &str) {
    // Release builds have no console (windows_subsystem = "windows"), so leave
    // a useful breadcrumb next to the exe. An install directory can be
    // read-only under some deployment policies; that must not turn a
    // diagnosable startup failure into a second panic.
    if std::fs::write(root.join("startup-error.txt"), detail).is_err() {
        let fallback = std::env::temp_dir().join("ai-text-detector-startup-error.txt");
        let _ = std::fs::write(fallback, detail);
    }
}

fn sidecar_failure(root: &std::path::Path, detail: String) -> SidecarLaunch {
    eprintln!("[shell] {detail}");
    write_startup_breadcrumb(root, &detail);
    SidecarLaunch {
        child: None,
        reader: None,
        token: new_sidecar_token(),
        error: Some(detail),
    }
}

/// Spawn backend/server.py with a session token; stdout carries the handshake
/// JSON line ({"port":N,"status":"ready"}) - server never logs to stdout (D2).
///
/// A missing or broken runtime is a first-run state, not a reason to terminate
/// the shell. Returning a launch result lets the WebView render the startup
/// gate and the runtime component page instead of flashing before any window
/// exists.
fn spawn_sidecar() -> SidecarLaunch {
    let token = new_sidecar_token();
    let root = sidecar_root();
    let candidate = python_candidates(&root)
        .into_iter()
        .find(|candidate| candidate.is_file());
    let Some(python) = candidate else {
        let expected = python_candidates(&root)
            .into_iter()
            .map(|p| p.display().to_string())
            .collect::<Vec<_>>()
            .join(" 或 ");
        return sidecar_failure(
            &root,
            format!("找不到 Python 运行时。已检查：{expected}。请修复基础服务组件后重新启动应用。"),
        );
    };
    let server = root.join("backend").join("server.py");
    if !server.is_file() {
        return sidecar_failure(
            &root,
            format!(
                "找不到后端服务文件：{}。请检查安装包资源是否完整。",
                server.display()
            ),
        );
    }

    let mut cmd = Command::new(&python);
    cmd.current_dir(&root)
        .arg("backend/server.py") // single argv: server.py relative to cwd
        .env("SIDE_TOKEN", &token)
        .env("PYTHONIOENCODING", "utf-8")
        // Never mutate the signed/read-only resource tree with .pyc files.
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    // A GUI Tauri process must never cause the console-subsystem Python
    // interpreter to allocate a second visible console window. Standard
    // handles stay inherited, so stderr still reaches the debug console and
    // the daily log file keeps recording when that console is hidden.
    #[cfg(windows)]
    cmd.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    // Dev WebView origin == devUrl (http://localhost:1420); the prod origin
    // (http://tauri.localhost / tauri://localhost) is already in the allowlist.
    if cfg!(debug_assertions) {
        cmd.env("SIDE_EXTRA_ORIGINS", "http://localhost:1420");
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            return sidecar_failure(
                &root,
                format!(
                    "启动 Python sidecar 失败：{e}。解释器：{}。请检查运行时文件和 VC++ 运行库。",
                    python.display()
                ),
            );
        }
    };
    let Some(stdout) = child.stdout.take() else {
        let _ = child.kill();
        let _ = child.wait();
        return sidecar_failure(
            &root,
            "Python sidecar 启动成功但没有可读取的 stdout 握手通道。".to_string(),
        );
    };
    SidecarLaunch {
        child: Some(child),
        reader: Some(BufReader::new(stdout)),
        token,
        error: None,
    }
}

fn spawn_handshake_reader(
    reader: BufReader<std::process::ChildStdout>,
    conn: Arc<OnceLock<(u16, String)>>,
    startup_error: Arc<OnceLock<String>>,
    token: String,
) {
    std::thread::spawn(move || {
        let mut saw_handshake = false;
        for line in reader.lines().map_while(Result::ok) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                if let Some(port) = v["port"].as_u64() {
                    let _ = conn.set((port as u16, token));
                    saw_handshake = true;
                    break;
                }
            }
        }
        if !saw_handshake && conn.get().is_none() {
            let _ = startup_error.set(
                "Python sidecar 在完成握手前退出。请打开开发者日志或基础服务组件检查运行时、VC++ 运行库与后端日志。"
                    .to_string(),
            );
        }
    });
}

/// Bring the (single) window back: minimised, hidden behind something, or the
/// user just clicked the tray icon.
fn focus_window(app: &tauri::AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        let _ = w.show();
        let _ = w.unminimize();
        let _ = w.set_focus();
    }
}

/// Reload the page from the *outside*.
///
/// The frontend blocks F5/Ctrl+R on purpose (a reload kills the socket that
/// owns the running detect, and one such reload left WebView2 painting white),
/// so this is the deliberate route back when the page itself is unresponsive.
///
/// Uses `reload()` (browser-level) rather than `eval("location.reload()")`:
/// eval needs a live JS world, which is exactly what is missing when the page
/// is white. `reload()` re-runs the initialization scripts, so it also repairs
/// a webview whose script context died.
fn reload_ui(app: &tauri::AppHandle) {
    if let Some(w) = app.get_webview_window("main") {
        if w.reload().is_err() {
            let _ = w.eval("window.location.reload()");
        }
    }
}

/// Tray menu instead of a native menu bar.
///
/// `decorations: false` + the in-webview TitleBar means a Win32 menu bar would
/// sit on top of the custom title bar and wreck the layout. The tray keeps a
/// recovery entry point that survives a white/unresponsive webview, and it is
/// what P5 (tray) was going to add anyway.
fn install_tray(app: &tauri::AppHandle) -> tauri::Result<()> {
    let show = MenuItem::with_id(app, "show", "显示窗口", true, None::<&str>)?;
    let reload = MenuItem::with_id(app, "reload", "重新加载界面", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "退出", true, None::<&str>)?;
    let menu = Menu::with_items(
        app,
        &[&show, &reload, &PredefinedMenuItem::separator(app)?, &quit],
    )?;
    let mut builder = TrayIconBuilder::with_id("main-tray")
        .tooltip("AI Text Detector")
        .menu(&menu)
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id().as_ref() {
            "show" => focus_window(app),
            "reload" => {
                focus_window(app);
                reload_ui(app);
            }
            "quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                focus_window(tray.app_handle());
            }
        });
    match app.default_window_icon() {
        Some(icon) => builder = builder.icon(icon.clone()),
        // a tray without an icon is an invisible escape hatch - say so loudly
        None => eprintln!("[shell] no bundled window icon: the tray entry will be invisible"),
    }
    builder.build(app)?;
    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    console_visibility::apply_saved_preference();
    // Before anything else: a duplicate launch must neither spawn a sidecar nor
    // create a second WebView2 against the same user-data folder - that is what
    // turned the window into a white shell on 2026-11-25.
    single_instance::enforce();
    let launch = spawn_sidecar();
    let token = launch.token;
    let startup_error = Arc::new(OnceLock::new());
    if let Some(error) = launch.error {
        let _ = startup_error.set(error);
    }
    let conn: Arc<OnceLock<(u16, String)>> = Arc::new(OnceLock::new());
    let conn_for_reader = Arc::clone(&conn);
    let startup_error_for_reader = Arc::clone(&startup_error);
    // Handshake reader thread: first (and only) stdout line carries the port.
    if let Some(reader) = launch.reader {
        spawn_handshake_reader(reader, conn_for_reader, startup_error_for_reader, token);
    } else if startup_error.get().is_none() {
        let _ = startup_error.set("sidecar 启动器没有返回可读取的握手通道。".to_string());
    }
    let state = SidecarState {
        child: Mutex::new(launch.child),
        conn,
        startup_error,
    };
    tauri::Builder::default()
        .manage(state)
        .plugin(tauri_plugin_clipboard_manager::init())
        // D40: the composer's "import file" button needs a native picker; a
        // webview <input type="file"> cannot hand back a path the sidecar can
        // open, so the dialog plugin is the only workable route on Windows.
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            sidecar_info,
            save_text_file,
            open_external_url,
            console_visibility_status,
            set_console_visibility
        ])
        .setup(|app| {
            install_tray(app.handle())?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|_handle, event| {
            // Kill-on-exit: no orphaned python sidecars after window close (P0 gate).
            if let RunEvent::Exit = event {
                console_visibility::shutdown();
                let state: State<SidecarState> = _handle.state();
                if let Some(mut child) = state.child.lock().expect("poisoned child lock").take() {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::validated_external_url;

    #[test]
    fn external_url_allowlist_accepts_official_sources() {
        assert!(validated_external_url("https://pytorch.org/get-started/locally/").is_ok());
        assert!(validated_external_url("https://pypi.tuna.tsinghua.edu.cn/simple/torch/").is_ok());
        assert!(validated_external_url("https://www.nvidia.cn/geforce/drivers/").is_ok());
    }

    #[test]
    fn external_url_allowlist_rejects_lookalikes_and_plain_http() {
        assert!(validated_external_url("https://pytorch.org.evil.example/file").is_err());
        assert!(validated_external_url("http://pytorch.org/get-started/locally/").is_err());
    }
}
