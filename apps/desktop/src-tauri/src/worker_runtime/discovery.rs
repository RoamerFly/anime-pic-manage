use crate::worker_runtime::types::{WorkerLaunchSpec, WorkerRuntimeError, WorkerRuntimeMode};
use std::fs;
use std::path::{Path, PathBuf};

/// Resolve an installed executable first, then the source checkout used by
/// local development.  All returned paths are canonical existing files or
/// directories, so a stale resource path cannot silently become a command.
#[allow(dead_code)]
pub fn resolve_worker_candidate(
    resource_dir: Option<&Path>,
    local_project_dir: &Path,
    local_models_root: &Path,
) -> Result<WorkerLaunchSpec, WorkerRuntimeError> {
    if let Some(resource_dir) = resource_dir {
        for executable in resource_executable_candidates(resource_dir) {
            if let Some(executable) = canonical_file(&executable) {
                let working_dir = executable
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| resource_dir.to_path_buf());
                let models_root = resource_models_root(resource_dir, local_models_root);
                return Ok(WorkerLaunchSpec::Executable {
                    executable,
                    working_dir,
                    models_root,
                });
            }
        }
        for script in [
            resource_dir.join("run_worker.py"),
            resource_dir.join("ai-worker").join("run_worker.py"),
        ] {
            if let Some(script) = canonical_file(&script) {
                let project_dir = script
                    .parent()
                    .and_then(canonical_dir)
                    .unwrap_or_else(|| resource_dir.to_path_buf());
                let models_root = resource_models_root(resource_dir, local_models_root);
                return Ok(source_worker_spec(project_dir, script, models_root));
            }
        }
    }

    let project_dir = canonical_dir(local_project_dir).ok_or(WorkerRuntimeError::NotFound)?;
    let script = project_dir.join("run_worker.py");
    let script = canonical_file(&script).ok_or(WorkerRuntimeError::NotFound)?;
    let models_root =
        canonical_dir(local_models_root).unwrap_or_else(|| local_models_root.to_path_buf());
    Ok(source_worker_spec(project_dir, script, models_root))
}

/// Resolve a Worker according to the persisted runtime preference.  The
/// executable mode is intentionally strict so selecting it never silently
/// falls back to a Python environment.  Embedded ENV mode first checks the
/// portable `resource_dir/env` layout, then the local source checkout where a
/// project `.venv` (or the existing uv fallback) remains supported.
pub fn resolve_worker_candidate_for_mode(
    resource_dir: Option<&Path>,
    local_project_dir: &Path,
    local_models_root: &Path,
    mode: WorkerRuntimeMode,
) -> Result<WorkerLaunchSpec, WorkerRuntimeError> {
    match mode {
        WorkerRuntimeMode::Executable => resource_dir
            .and_then(|resource_dir| executable_worker_spec(resource_dir, local_models_root))
            .ok_or(WorkerRuntimeError::NotFound),
        WorkerRuntimeMode::EmbeddedEnv => {
            if let Some(resource_dir) = resource_dir {
                if let Some(spec) = resource_env_worker_spec(resource_dir, local_models_root) {
                    return Ok(spec);
                }
                // The standard installer ships the packaged Worker without the
                // optional fallback ENV, so the EXE has to be considered before
                // giving up on the application folder.
                if let Some(spec) = executable_worker_spec(resource_dir, local_models_root) {
                    return Ok(spec);
                }
                if let Some(spec) = resource_source_worker_spec(resource_dir, local_models_root) {
                    return Ok(spec);
                }
                // A packaged application must never fall back to the developer
                // checkout the binary was compiled in: that path belongs to the
                // build machine and writing there would leave the application
                // folder silently incomplete.
                if is_packaged_layout(resource_dir) {
                    return Err(WorkerRuntimeError::NotFound);
                }
            }
            let project_dir =
                canonical_dir(local_project_dir).ok_or(WorkerRuntimeError::NotFound)?;
            let script = canonical_file(&project_dir.join("run_worker.py"))
                .ok_or(WorkerRuntimeError::NotFound)?;
            let models_root =
                canonical_dir(local_models_root).unwrap_or_else(|| local_models_root.to_path_buf());
            Ok(source_worker_spec(project_dir, script, models_root))
        }
    }
}

pub fn executable_worker_spec(
    resource_dir: &Path,
    local_models_root: &Path,
) -> Option<WorkerLaunchSpec> {
    resource_executable_candidates(resource_dir)
        .into_iter()
        .find_map(|candidate| {
            canonical_file(&candidate).map(|executable| {
                let working_dir = executable
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| resource_dir.to_path_buf());
                let models_root = resource_models_root(resource_dir, local_models_root);
                WorkerLaunchSpec::Executable {
                    executable,
                    working_dir,
                    models_root,
                }
            })
        })
}

pub fn resource_source_worker_spec(
    resource_dir: &Path,
    local_models_root: &Path,
) -> Option<WorkerLaunchSpec> {
    resource_roots(resource_dir).into_iter().find_map(|root| {
        [
            root.join("run_worker.py"),
            root.join("ai-worker").join("run_worker.py"),
        ]
        .into_iter()
        .find_map(|script| {
            let script = canonical_file(&script)?;
            let project_dir = script.parent().and_then(canonical_dir)?;
            let models_root = resource_models_root(resource_dir, local_models_root);
            Some(source_worker_spec(project_dir, script, models_root))
        })
    })
}

pub fn resource_env_worker_spec(
    resource_dir: &Path,
    local_models_root: &Path,
) -> Option<WorkerLaunchSpec> {
    resource_roots(resource_dir).into_iter().find_map(|root| {
        let env_dir = root.join("env");
        let interpreter = canonical_file(&env_dir.join(venv_python_relative()))?;
        [
            env_dir.join("worker").join("run_worker.py"),
            env_dir.join("run_worker.py"),
        ]
        .into_iter()
        .find_map(|script| {
            let script = canonical_file(&script)?;
            let project_dir = script.parent().and_then(canonical_dir)?;
            let models_root = resource_models_root(resource_dir, local_models_root);
            Some(WorkerLaunchSpec::PythonScript {
                interpreter: interpreter.clone(),
                project_dir,
                script,
                models_root,
            })
        })
    })
}

pub fn source_worker_spec(
    project_dir: PathBuf,
    script: PathBuf,
    models_root: PathBuf,
) -> WorkerLaunchSpec {
    if let Some(interpreter) = venv_python(&project_dir) {
        WorkerLaunchSpec::PythonScript {
            interpreter,
            project_dir,
            script,
            models_root,
        }
    } else {
        WorkerLaunchSpec::UvScript {
            project_dir,
            script,
            models_root,
        }
    }
}

pub fn venv_python(project_dir: &Path) -> Option<PathBuf> {
    canonical_file(&project_dir.join(Path::new(".venv").join(venv_python_relative())))
}

pub fn venv_python_relative() -> PathBuf {
    if cfg!(windows) {
        PathBuf::from("Scripts").join("python.exe")
    } else {
        PathBuf::from("bin").join("python")
    }
}

pub fn resource_executable_candidates(resource_dir: &Path) -> Vec<PathBuf> {
    let names = if cfg!(windows) {
        ["ai-worker.exe", "anime-pic-ai-worker.exe"]
    } else {
        ["ai-worker", "anime-pic-ai-worker"]
    };
    let mut candidates = Vec::new();
    for root in resource_roots(resource_dir) {
        for name in names {
            candidates.push(root.join(name));
            candidates.push(root.join("runtime").join(name));
            candidates.push(root.join("ai-worker").join(name));
        }
    }
    candidates
}

pub fn resource_models_root(resource_dir: &Path, fallback: &Path) -> PathBuf {
    for root in resource_roots(resource_dir) {
        let candidate = root.join("models");
        if candidate.is_dir() {
            return candidate;
        }
    }
    // A packaged layout keeps its models next to the program even when the
    // folder does not exist yet: a fresh install downloads them on demand from
    // 设置 → 模型配置, and they must land inside the application folder.
    if is_packaged_layout(resource_dir) {
        if let Some(root) = resource_roots(resource_dir)
            .into_iter()
            .find(|root| root.join("BUILD_FLAVOR.txt").is_file() || root.join("app").is_dir())
        {
            return root.join("models");
        }
        return resource_dir.join("models");
    }
    fallback.to_path_buf()
}

/// True when `resource_dir` is a released package (`app\` next to the program).
pub fn is_packaged_layout(resource_dir: &Path) -> bool {
    resource_dir.join("app").is_dir()
        || resource_dir.join("BUILD_FLAVOR.txt").is_file()
        || resource_dir
            .parent()
            .is_some_and(|parent| parent.join("BUILD_FLAVOR.txt").is_file())
}

pub fn resource_roots(resource_dir: &Path) -> Vec<PathBuf> {
    let mut roots = vec![resource_dir.to_path_buf()];
    // Released packages keep the program files in `app\`, while `models\` and
    // `resources\` stay in the package root; both layouts are searched.
    let app_dir = resource_dir.join("app");
    if app_dir.is_dir() {
        roots.push(app_dir);
    }
    if let Some(parent) = resource_dir.parent() {
        if parent != resource_dir {
            roots.push(parent.to_path_buf());
        }
    }
    roots
}

pub fn canonical_file(path: &Path) -> Option<PathBuf> {
    fs::canonicalize(path).ok().filter(|path| path.is_file())
}

pub fn canonical_dir(path: &Path) -> Option<PathBuf> {
    fs::canonicalize(path).ok().filter(|path| path.is_dir())
}

pub fn local_project_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../ai-worker")
}

pub fn local_models_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../models")
}
