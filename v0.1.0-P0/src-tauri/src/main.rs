use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use tauri::{Emitter, Manager, RunEvent, WindowEvent};

/// Sidecar process handle held for the whole app lifetime.
struct Sidecar(Mutex<Option<Child>>);

/// One-shot hex token (D3 threat model: loopback self-connection only).
fn hex_token() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seed = nanos ^ ((std::process::id() as u128) << 64);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in seed.to_be_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    let mut s = String::with_capacity(64);
    for _ in 0..4 {
        h ^= h << 13;
        h ^= h >> 7;
        h ^= h << 17;
        s.push_str(&format!("{h:016x}"));
    }
    s
}

/// Spawn `backend/server.py` on stdout-guided handshake (D2).
fn spawn_sidecar(app: &tauri::AppHandle) {
    let token = hex_token();
    let python = std::env::var("SIDE_PYTHON").unwrap_or_else(|_| "python".into());
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/../backend/server.py");
    match Command::new(&python)
        .args(["-u", script])
        .env("SIDE_TOKEN", &token)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
    {
        Ok(mut child) => {
            let stdout = child.stdout.take().expect("piped stdout");
            let emitter = app.clone();
            let token2 = token.clone();
            std::thread::spawn(move || {
                let mut line = String::new();
                let ok = BufReader::new(stdout)
                    .read_line(&mut line)
                    .map(|n| n > 0)
                    .unwrap_or(false);
                match ok
                    .then(|| serde_json::from_str::<serde_json::Value>(line.trim()).ok())
                    .flatten()
                    .and_then(|v| v["port"].as_u64().map(|p| (p, token2)))
                {
                    Some((port, tok)) => {
                        let _ = emitter.emit("sidecar-ready", serde_json::json!({ "port": port, "token": tok }));
                    }
                    None => {
                        let _ = emitter.emit("sidecar-error", "handshake failed (bad stdout)");
                    }
                }
            });
            app.state::<Sidecar>().0.lock().unwrap().replace(child);
        }
        Err(e) => {
            let _ = app.emit("sidecar-error", format!("spawn failed: {e}"));
        }
    }
}

fn kill_sidecar(app: &tauri::AppHandle) {
    if let Some(c) = app.state::<Sidecar>().0.lock().unwrap().as_mut() {
        let _ = c.kill();
        let _ = c.wait();
    }
}

fn main() {
    tauri::Builder::default()
        .manage(Sidecar(Mutex::new(None)))
        .setup(|app| {
            spawn_sidecar(app.handle());
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| match event {
            RunEvent::WindowEvent {
                event: WindowEvent::CloseRequested { .. },
                ..
            }
            | RunEvent::Exit => kill_sidecar(app),
            _ => {}
        });
}
