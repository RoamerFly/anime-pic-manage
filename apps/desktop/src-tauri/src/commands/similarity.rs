use crate::database::{self, Database};
use crate::ipc::{core_error, failure, normalize_request_id, success, IpcEnvelope};
use crate::models::allow_asset_directory;
use crate::result_store::{self, ResultKind};
use crate::state::{AppState, ScanControl, ScanControlResponse};
use crate::worker_runtime::{self, WorkerManager};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager, State};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityScanArgs {
    pub path: String,
    pub threshold: Option<f64>,
    pub include_subfolders: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityScanProgress {
    pub phase: String,
    pub current: usize,
    pub total: usize,
    pub path: Option<String>,
    pub message: String,
    #[serde(default)]
    pub directory: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityScanState {
    pub running: bool,
    pub directory: Option<String>,
    pub phase: String,
    pub current: usize,
    pub total: usize,
    pub path: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityScanPayload {
    pub directory: String,
    pub total_scanned: usize,
    pub groups: Vec<database::SimilarityGroup>,
    pub duplicates_count: usize,
    pub potential_space_saved: u64,
    pub cancelled: bool,
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityItemDecision {
    pub path: String,
    pub action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityDecisionRequest {
    pub archive_directory: Option<String>,
    pub archive_directory_name: Option<String>,
    pub result_directory: Option<String>,
    pub decisions: Vec<SimilarityItemDecision>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityProcessedItem {
    pub path: String,
    pub action: String,
    pub destination: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityItemError {
    pub path: String,
    pub error: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityDecisionResult {
    pub success_count: usize,
    pub failed_count: usize,
    pub processed: Vec<SimilarityProcessedItem>,
    pub errors: Vec<SimilarityItemError>,
    pub freed_bytes: u64,
}

/// Animation-capable files need the multi-frame feature payload.
///
/// Caches written before that change only hold the first frame, so they are
/// treated as a miss once and re-extracted.
fn feature_cache_is_current(cached: &database::ImageSimilarityFeatureRecord, path: &Path) -> bool {
    if cached.feature_version >= database::SIMILARITY_FEATURE_VERSION {
        return true;
    }
    let animation_capable = matches!(
        path.extension()
            .and_then(|value| value.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("gif") | Some("webp")
    );
    !animation_capable
}

fn run_similarity_scan(
    app: AppHandle,
    worker: Arc<Mutex<WorkerManager>>,
    database: Arc<Mutex<Database>>,
    control: Arc<ScanControl>,
    directory: PathBuf,
    threshold: f64,
    include_subfolders: bool,
    max_workers: u32,
) -> Result<SimilarityScanPayload, worker_runtime::WorkerRuntimeError> {
    let emit_progress =
        |phase: &str, current: usize, total: usize, path: Option<&str>, message: &str| {
            control.update_phase_progress(current, total, path.unwrap_or(""), message, phase);
            let _ = app.emit(
                "similarity://progress",
                SimilarityScanProgress {
                    phase: phase.to_string(),
                    current,
                    total,
                    path: path.map(str::to_string),
                    message: message.to_string(),
                    directory: Some(directory.to_string_lossy().into_owned()),
                },
            );
        };

    emit_progress("enumerating", 0, 0, None, "正在发现目录中的图片文件...");

    let enumerate_payload = {
        let mut manager = worker.lock().map_err(|_| {
            worker_runtime::WorkerRuntimeError::Io("Worker 状态锁已中毒".to_string())
        })?;
        manager.request(
            "images.enumerate",
            json!({
                "directory": directory.to_string_lossy(),
                "recursive": include_subfolders,
                "include_hidden": false,
            }),
        )?
    };

    let image_values = enumerate_payload
        .get("images")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    let image_paths: Vec<String> = image_values
        .iter()
        .filter_map(|img| img.get("path").and_then(Value::as_str).map(str::to_string))
        .collect();

    let total = image_paths.len();
    if total < 2 {
        emit_progress(
            "finalizing",
            total,
            total,
            None,
            "扫描完成，正在写入结果文件...",
        );
        return Ok(SimilarityScanPayload {
            directory: directory.to_string_lossy().into_owned(),
            total_scanned: total,
            groups: Vec::new(),
            duplicates_count: 0,
            potential_space_saved: 0,
            cancelled: false,
            completed_at: None,
        });
    }

    emit_progress("checking_cache", 0, total, None, "正在比对已有特征缓存...");

    let cached_records = if let Ok(db) = database.lock() {
        db.get_similarity_features_by_paths(&image_paths)
            .unwrap_or_default()
    } else {
        Vec::new()
    };

    let mut cached_map: std::collections::HashMap<String, database::ImageSimilarityFeatureRecord> =
        std::collections::HashMap::with_capacity(cached_records.len());
    for rec in cached_records {
        cached_map.insert(rec.path.clone(), rec);
    }

    let mut all_features: Vec<Value> = Vec::with_capacity(total);
    let mut uncached_paths: Vec<String> = Vec::new();

    for p in &image_paths {
        let path_buf = Path::new(p);
        let actual_size = fs::metadata(path_buf).map(|m| m.len() as i64).unwrap_or(-1);
        if let Some(cached) = cached_map.get(p) {
            let feature_cache_current = feature_cache_is_current(cached, path_buf);
            if cached.file_size == actual_size && actual_size > 0 && feature_cache_current {
                let color_hist: Value =
                    serde_json::from_str(&cached.color_hist_json).unwrap_or_else(|_| json!([]));
                let frames: Value = cached
                    .frames_json
                    .as_deref()
                    .and_then(|raw| serde_json::from_str(raw).ok())
                    .unwrap_or(Value::Null);
                all_features.push(json!({
                    "path": cached.path,
                    "file_size": cached.file_size,
                    "dimensions": [cached.width, cached.height],
                    "format": cached.format,
                    "clarity_score": cached.clarity_score,
                    "phash": cached.phash,
                    "dhash": cached.dhash,
                    "color_histogram": color_hist,
                    "frame_count": cached.frame_count.max(1),
                    "is_animated": cached.frame_count > 1,
                    "frames": frames,
                }));
                continue;
            }
        }
        uncached_paths.push(p.clone());
    }

    let cached_count = all_features.len();
    let uncached_count = uncached_paths.len();
    emit_progress(
        "extracting",
        cached_count,
        total,
        None,
        &format!("已命中缓存 {cached_count} 张，正在提取剩余 {uncached_count} 张图片的视觉特征..."),
    );

    let batch_size = 30;
    let mut processed_so_far = cached_count;
    let mut newly_extracted_records = Vec::new();

    for chunk in uncached_paths.chunks(batch_size) {
        if control.is_cancelled() {
            emit_progress(
                "cancelling",
                processed_so_far,
                total,
                None,
                "用户已取消相似度扫描，正在保存已扫描结果...",
            );
            return Ok(SimilarityScanPayload {
                directory: directory.to_string_lossy().into_owned(),
                total_scanned: processed_so_far,
                groups: Vec::new(),
                duplicates_count: 0,
                potential_space_saved: 0,
                cancelled: true,
                completed_at: None,
            });
        }

        let first_path = chunk.first().cloned();
        emit_progress(
            "extracting",
            processed_so_far,
            total,
            first_path.as_deref(),
            &format!(
                "正在提取特征 ({}/{total})...",
                processed_so_far + chunk.len()
            ),
        );

        let extract_res = {
            let mut manager = worker.lock().map_err(|_| {
                worker_runtime::WorkerRuntimeError::Io("Worker 状态锁已中毒".to_string())
            })?;
            manager.request(
                "similarity.features.extract",
                json!({
                    "paths": chunk,
                    "max_workers": max_workers,
                }),
            )?
        };

        if let Some(features_array) = extract_res.get("features").and_then(Value::as_array) {
            for f in features_array {
                all_features.push(f.clone());

                if let (
                    Some(path),
                    Some(file_size),
                    Some(dims),
                    Some(fmt),
                    Some(clarity),
                    Some(phash),
                    Some(dhash),
                    Some(color_hist),
                ) = (
                    f.get("path").and_then(Value::as_str),
                    f.get("file_size").and_then(Value::as_i64),
                    f.get("dimensions").and_then(Value::as_array),
                    f.get("format").and_then(Value::as_str),
                    f.get("clarity_score").and_then(Value::as_f64),
                    f.get("phash").and_then(Value::as_str),
                    f.get("dhash").and_then(Value::as_str),
                    f.get("color_histogram"),
                ) {
                    let w = dims.get(0).and_then(Value::as_i64).unwrap_or(0);
                    let h = dims.get(1).and_then(Value::as_i64).unwrap_or(0);
                    let color_hist_json = color_hist.to_string();
                    let frames_json = f.get("frames").map(Value::to_string);
                    let frame_count = f
                        .get("frame_count")
                        .and_then(Value::as_i64)
                        .unwrap_or(1)
                        .max(1);
                    newly_extracted_records.push(database::ImageSimilarityFeatureRecord {
                        path: path.to_string(),
                        file_size,
                        width: w,
                        height: h,
                        format: fmt.to_string(),
                        clarity_score: clarity,
                        phash: phash.to_string(),
                        dhash: dhash.to_string(),
                        color_hist_json,
                        frames_json,
                        frame_count,
                        feature_version: database::SIMILARITY_FEATURE_VERSION,
                    });
                }
            }
        }

        processed_so_far += chunk.len();
    }

    if !newly_extracted_records.is_empty() {
        if let Ok(mut db) = database.lock() {
            let _ = db.save_similarity_features(&newly_extracted_records);
        }
    }

    if control.is_cancelled() {
        emit_progress(
            "cancelling",
            processed_so_far,
            total,
            None,
            "用户已取消相似度扫描，正在保存已扫描结果...",
        );
        return Ok(SimilarityScanPayload {
            directory: directory.to_string_lossy().into_owned(),
            total_scanned: processed_so_far,
            groups: Vec::new(),
            duplicates_count: 0,
            potential_space_saved: 0,
            cancelled: true,
            completed_at: None,
        });
    }

    emit_progress(
        "clustering",
        total,
        total,
        None,
        "正在分析视觉特征并聚类相似图片组...",
    );

    let cluster_res = {
        let mut manager = worker.lock().map_err(|_| {
            worker_runtime::WorkerRuntimeError::Io("Worker 状态锁已中毒".to_string())
        })?;
        manager.request(
            "similarity.cluster",
            json!({
                "features": all_features,
                "threshold": threshold,
            }),
        )?
    };

    let mut groups: Vec<database::SimilarityGroup> = Vec::new();
    if let Some(raw_groups) = cluster_res.get("groups").and_then(Value::as_array) {
        for rg in raw_groups {
            if let Ok(g) = serde_json::from_value::<database::SimilarityGroup>(rg.clone()) {
                groups.push(g);
            }
        }
    }

    let duplicates_count = cluster_res
        .get("duplicates_count")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    let potential_space_saved = cluster_res
        .get("potential_space_saved")
        .and_then(Value::as_u64)
        .unwrap_or(0);

    emit_progress(
        "finalizing",
        total,
        total,
        None,
        &format!(
            "扫描完成，发现 {} 个相似分组，正在写入结果文件...",
            groups.len(),
        ),
    );

    Ok(SimilarityScanPayload {
        directory: directory.to_string_lossy().into_owned(),
        total_scanned: total,
        groups,
        duplicates_count,
        potential_space_saved,
        cancelled: false,
        completed_at: None,
    })
}

#[tauri::command]
pub async fn scan_similarity_preview(
    app: AppHandle,
    state: State<'_, AppState>,
    args: SimilarityScanArgs,
    request_id: Option<String>,
) -> Result<IpcEnvelope<SimilarityScanPayload>, String> {
    let request_id = normalize_request_id(request_id);
    let candidate = Path::new(&args.path);
    if !candidate.is_absolute() || !candidate.is_dir() {
        return Ok(failure(
            "similarity.scan.start",
            request_id.clone(),
            core_error(
                &request_id,
                "INVALID_DIRECTORY",
                "请选择一个存在的本地文件夹。",
                Some("path must be an absolute directory".to_string()),
                false,
            ),
        ));
    }

    let directory = match fs::canonicalize(candidate) {
        Ok(dir) if dir.is_dir() => dir,
        Ok(_) => {
            return Ok(failure(
                "similarity.scan.start",
                request_id.clone(),
                core_error(
                    &request_id,
                    "INVALID_DIRECTORY",
                    "选择的路径不是文件夹。",
                    None,
                    false,
                ),
            ));
        }
        Err(e) => {
            return Ok(failure(
                "similarity.scan.start",
                request_id.clone(),
                core_error(
                    &request_id,
                    "INVALID_DIRECTORY",
                    "无法访问选择的图片文件夹。",
                    Some(e.to_string()),
                    false,
                ),
            ));
        }
    };

    if let Err(detail) = allow_asset_directory(&app, &directory) {
        return Ok(failure(
            "similarity.scan.start",
            request_id.clone(),
            core_error(
                &request_id,
                "ASSET_SCOPE_FAILED",
                "无法授权预览所选图片文件夹。",
                Some(detail),
                true,
            ),
        ));
    }

    if !state.similarity_scan_control.try_start() {
        return Ok(failure(
            "similarity.scan.start",
            request_id.clone(),
            core_error(
                &request_id,
                "SIMILARITY_SCAN_ALREADY_RUNNING",
                "已有相似度扫描任务正在运行，请等待当前任务完成。",
                None,
                true,
            ),
        ));
    }

    let control = Arc::clone(&state.similarity_scan_control);
    let worker = Arc::clone(&state.worker);
    let database = Arc::clone(&state.database);
    let scan_app = app.clone();
    let threshold = args.threshold.unwrap_or(0.85);
    let include_subfolders = args.include_subfolders.unwrap_or(true);
    let max_workers = match state.database.lock() {
        Ok(database) => crate::app_settings::load_app_settings(&database)
            .map(|settings| settings.similarity_workers)
            .unwrap_or_else(|_| crate::app_settings::default_similarity_workers()),
        Err(_) => crate::app_settings::default_similarity_workers(),
    };
    let scan_directory = directory.to_string_lossy().into_owned();
    state.similarity_scan_control.set_directory(&scan_directory);

    let emit_terminal = |phase: &str, current: usize, total: usize, message: String| {
        let _ = app.emit(
            "similarity://progress",
            SimilarityScanProgress {
                phase: phase.to_string(),
                current,
                total,
                path: None,
                message,
                directory: Some(scan_directory.clone()),
            },
        );
    };

    let scan = tauri::async_runtime::spawn_blocking(move || {
        run_similarity_scan(
            scan_app,
            worker,
            database,
            control,
            directory,
            threshold,
            include_subfolders,
            max_workers,
        )
    })
    .await;

    state.similarity_scan_control.finish();

    match scan {
        Ok(Ok(mut payload)) => {
            let completed_at = result_store::now_rfc3339();
            payload.completed_at = Some(completed_at.clone());
            if let Err(error) = result_store::save_versioned(
                Path::new(&state.data_dir),
                ResultKind::Similarity,
                &payload.directory,
                &completed_at,
                &payload,
            ) {
                emit_terminal(
                    "error",
                    payload.total_scanned,
                    payload.total_scanned,
                    "相似度扫描已完成，但结果文件保存失败。".to_string(),
                );
                return Ok(failure(
                    "similarity.scan.start",
                    request_id.clone(),
                    core_error(
                        &request_id,
                        "SIMILARITY_RESULT_SAVE_FAILED",
                        "相似度扫描已完成，但结果文件保存失败。",
                        Some(error.to_string()),
                        true,
                    ),
                ));
            }
            // The terminal event is emitted only after the result file exists so
            // a page that re-mounts mid-scan can reload a complete result.
            if payload.cancelled {
                emit_terminal(
                    "cancelled",
                    payload.total_scanned,
                    payload.total_scanned,
                    format!(
                        "扫描已取消，已保存 {} 张图片的部分结果。",
                        payload.total_scanned
                    ),
                );
            } else {
                emit_terminal(
                    "completed",
                    payload.total_scanned,
                    payload.total_scanned,
                    format!(
                        "扫描完成！发现 {} 个相似分组，共 {} 张重复/变体图片。",
                        payload.groups.len(),
                        payload.duplicates_count
                    ),
                );
            }
            Ok(success("similarity.scan.start", request_id, payload))
        }
        Ok(Err(error)) => {
            emit_terminal("error", 0, 0, format!("相似度扫描失败：{error}"));
            Ok(failure(
                "similarity.scan.start",
                request_id.clone(),
                core_error(
                    &request_id,
                    "SIMILARITY_SCAN_FAILED",
                    "相似度扫描失败。",
                    Some(error.to_string()),
                    true,
                ),
            ))
        }
        Err(error) => {
            emit_terminal("error", 0, 0, format!("相似度扫描任务异常中止：{error}"));
            Ok(failure(
                "similarity.scan.start",
                request_id.clone(),
                core_error(
                    &request_id,
                    "SIMILARITY_SCAN_TASK_FAILED",
                    "相似度扫描任务异常中止。",
                    Some(error.to_string()),
                    true,
                ),
            ))
        }
    }
}

#[tauri::command]
pub fn cancel_similarity_scan(
    app: AppHandle,
    state: State<'_, AppState>,
    request_id: Option<String>,
) -> IpcEnvelope<ScanControlResponse> {
    let request_id = normalize_request_id(request_id);
    let running = state.similarity_scan_control.request_cancel();
    let _ = app.emit(
        "similarity://progress",
        SimilarityScanProgress {
            // Not a terminal phase: the scan still has to persist partial
            // results, and only then is the real "cancelled" event emitted.
            phase: "cancelling".to_string(),
            current: 0,
            total: 0,
            path: None,
            message: "已请求取消相似度扫描，正在停止并保存已扫描结果。".to_string(),
            directory: None,
        },
    );
    success(
        "similarity.scan.cancel",
        request_id,
        ScanControlResponse {
            status: if running { "cancelling" } else { "idle" }.to_string(),
            running,
            message: if running {
                "已请求取消相似度扫描，正在停止。"
            } else {
                "当前没有运行中的相似度扫描。"
            }
            .to_string(),
        },
    )
}

/// Report whether a similarity scan is still running.
///
/// The desktop page is unmounted whenever the user leaves it, which loses the
/// in-flight scan promise. Re-mounting therefore re-attaches to the backend
/// through this snapshot instead of reloading a previous (possibly huge)
/// result file while a new scan is still writing.
#[tauri::command]
pub fn get_similarity_scan_state(
    state: State<'_, AppState>,
    request_id: Option<String>,
) -> IpcEnvelope<SimilarityScanState> {
    let request_id = normalize_request_id(request_id);
    let snapshot = state.similarity_scan_control.snapshot();
    success(
        "similarity.scan.state",
        request_id,
        SimilarityScanState {
            running: snapshot.running,
            directory: Some(snapshot.directory).filter(|value| !value.is_empty()),
            phase: snapshot.phase,
            current: snapshot.current,
            total: snapshot.total,
            path: Some(snapshot.path).filter(|value| !value.is_empty()),
            message: snapshot.message,
        },
    )
}

#[tauri::command]
pub async fn get_cached_similarity_groups(
    app: AppHandle,
    request_id: Option<String>,
) -> IpcEnvelope<Vec<database::SimilarityGroup>> {
    let request_id = normalize_request_id(request_id);
    let database = Arc::clone(&app.state::<AppState>().database);
    let groups = tauri::async_runtime::spawn_blocking(move || match database.lock() {
        Ok(db) => db.get_latest_similarity_groups().unwrap_or_default(),
        Err(_) => Vec::new(),
    })
    .await
    .unwrap_or_default();
    success("similarity.group.list", request_id, groups)
}

#[tauri::command]
pub async fn get_latest_similarity_result(
    app: AppHandle,
    path: String,
    request_id: Option<String>,
) -> IpcEnvelope<Option<SimilarityScanPayload>> {
    let request_id = normalize_request_id(request_id);
    let data_dir = PathBuf::from(&app.state::<AppState>().data_dir);
    let loaded = tauri::async_runtime::spawn_blocking(move || {
        result_store::load_latest(&data_dir, ResultKind::Similarity, &path)
    })
    .await
    .unwrap_or_else(|error| Err(std::io::Error::other(error.to_string())));
    match loaded {
        Ok(payload) => success("similarity.group.list", request_id, payload),
        Err(error) => failure(
            "similarity.group.list",
            request_id.clone(),
            core_error(
                &request_id,
                "SIMILARITY_RESULT_READ_FAILED",
                "无法读取该图库的相似度结果文件。",
                Some(error.to_string()),
                true,
            ),
        ),
    }
}

fn apply_decisions_to_similarity_result(
    data_dir: &Path,
    directory: &str,
    decisions: &[SimilarityItemDecision],
    processed: &[SimilarityProcessedItem],
) -> std::io::Result<bool> {
    let decision_by_path = decisions
        .iter()
        .map(|item| (item.path.as_str(), item.action.as_str()))
        .collect::<std::collections::HashMap<_, _>>();
    let removed = processed
        .iter()
        .filter(|item| item.action == "archive" || item.action == "delete")
        .map(|item| item.path.as_str())
        .collect::<std::collections::HashSet<_>>();

    result_store::update_latest::<SimilarityScanPayload>(
        data_dir,
        ResultKind::Similarity,
        directory,
        |payload| {
            for group in &mut payload.groups {
                group.items.retain_mut(|item| {
                    if removed.contains(item.path.as_str()) {
                        return false;
                    }
                    if let Some(action) = decision_by_path.get(item.path.as_str()) {
                        item.decision = Some((*action).to_string());
                    }
                    true
                });
            }
            payload.groups.retain(|group| group.items.len() >= 2);
            payload.duplicates_count = payload
                .groups
                .iter()
                .map(|group| group.items.len().saturating_sub(1))
                .sum();
            payload.potential_space_saved = payload
                .groups
                .iter()
                .flat_map(|group| group.items.iter())
                .filter(|item| item.decision.as_deref() == Some("delete"))
                .map(|item| item.file_size.max(0) as u64)
                .sum();
        },
    )
}

#[tauri::command]
pub async fn save_similarity_result_decisions(
    app: AppHandle,
    path: String,
    decisions: Vec<SimilarityItemDecision>,
    request_id: Option<String>,
) -> IpcEnvelope<bool> {
    let request_id = normalize_request_id(request_id);
    let data_dir = PathBuf::from(&app.state::<AppState>().data_dir);
    let saved = tauri::async_runtime::spawn_blocking(move || {
        apply_decisions_to_similarity_result(&data_dir, &path, &decisions, &[])
    })
    .await
    .unwrap_or_else(|error| Err(std::io::Error::other(error.to_string())));
    match saved {
        Ok(saved) => success("similarity.decision.apply", request_id, saved),
        Err(error) => failure(
            "similarity.decision.apply",
            request_id.clone(),
            core_error(
                &request_id,
                "SIMILARITY_RESULT_SAVE_FAILED",
                "无法保存相似度结果选择。",
                Some(error.to_string()),
                true,
            ),
        ),
    }
}

fn archive_similarity_file(path: &Path, archive_dir: &Path) -> Result<PathBuf, std::io::Error> {
    fs::create_dir_all(archive_dir)?;
    let file_name = path
        .file_name()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "无法获取文件名"))?;

    let mut destination = archive_dir.join(file_name);
    if destination.exists() && fs::canonicalize(&destination).ok() == fs::canonicalize(path).ok() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "归档目录与源文件所在目录相同",
        ));
    }
    if destination.exists() {
        let stem = path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("file");
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("");
        let mut counter = 1;
        loop {
            let candidate_name = if extension.is_empty() {
                format!("{stem}_{counter}")
            } else {
                format!("{stem}_{counter}.{extension}")
            };
            let candidate = archive_dir.join(candidate_name);
            if !candidate.exists() {
                destination = candidate;
                break;
            }
            counter += 1;
        }
    }

    fs::rename(path, &destination)
        .or_else(|_| fs::copy(path, &destination).and_then(|_| fs::remove_file(path)))?;
    Ok(destination)
}

fn delete_similarity_file(path: &Path) -> Result<u64, std::io::Error> {
    let file_size = fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    fs::remove_file(path)?;
    Ok(file_size)
}

fn resolve_archive_directory(request: &SimilarityDecisionRequest) -> Option<PathBuf> {
    if let Some(directory) = request
        .archive_directory
        .as_deref()
        .filter(|directory| !directory.trim().is_empty())
    {
        return Some(PathBuf::from(directory));
    }

    let base_directory = request
        .result_directory
        .as_deref()
        .filter(|directory| !directory.trim().is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            request
                .decisions
                .first()
                .and_then(|decision| Path::new(&decision.path).parent().map(Path::to_path_buf))
        });

    let archive_name = request
        .archive_directory_name
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .unwrap_or("_duplicates");

    base_directory.map(|directory| directory.join(archive_name))
}

fn remove_processed_items_from_groups(
    groups: &mut Vec<database::SimilarityGroup>,
    processed: &[SimilarityProcessedItem],
) {
    let processed_paths: std::collections::HashSet<&str> = processed
        .iter()
        .filter(|item| item.action == "archive" || item.action == "delete")
        .map(|item| item.path.as_str())
        .collect();
    if processed_paths.is_empty() {
        return;
    }

    for group in groups.iter_mut() {
        group
            .items
            .retain(|item| !processed_paths.contains(item.path.as_str()));
    }
    groups.retain(|group| group.items.len() >= 2);
}

#[tauri::command]
pub async fn apply_similarity_decisions(
    app: AppHandle,
    state: State<'_, AppState>,
    request: SimilarityDecisionRequest,
    request_id: Option<String>,
) -> Result<IpcEnvelope<SimilarityDecisionResult>, String> {
    let request_id = normalize_request_id(request_id);

    let result_directory = request.result_directory.clone();
    let submitted_decisions = request.decisions.clone();
    let target_archive_dir = resolve_archive_directory(&request);

    if let Some(ref dir) = target_archive_dir {
        let _ = allow_asset_directory(&app, dir);
    }

    let mut processed = Vec::new();
    let mut errors = Vec::new();
    let mut success_count = 0;
    let mut failed_count = 0;
    let mut freed_bytes = 0u64;

    for item in request.decisions {
        let path = Path::new(&item.path);
        match item.action.as_str() {
            "keep" => {
                success_count += 1;
                processed.push(SimilarityProcessedItem {
                    path: item.path,
                    action: "keep".to_string(),
                    destination: None,
                });
            }
            "archive" => {
                if !path.exists() {
                    failed_count += 1;
                    errors.push(SimilarityItemError {
                        path: item.path.clone(),
                        error: "目标文件不存在".to_string(),
                    });
                    continue;
                }
                let Some(ref archive_dir) = target_archive_dir else {
                    failed_count += 1;
                    errors.push(SimilarityItemError {
                        path: item.path,
                        error: "未指定归档目录".to_string(),
                    });
                    continue;
                };

                match archive_similarity_file(path, archive_dir) {
                    Ok(destination) => {
                        success_count += 1;
                        processed.push(SimilarityProcessedItem {
                            path: item.path,
                            action: "archive".to_string(),
                            destination: Some(destination.to_string_lossy().into_owned()),
                        });
                    }
                    Err(e) => {
                        failed_count += 1;
                        errors.push(SimilarityItemError {
                            path: item.path,
                            error: format!("归档移动失败: {e}"),
                        });
                    }
                }
            }
            "delete" => {
                if !path.exists() {
                    failed_count += 1;
                    errors.push(SimilarityItemError {
                        path: item.path.clone(),
                        error: "目标文件不存在".to_string(),
                    });
                    continue;
                }
                match delete_similarity_file(path) {
                    Ok(file_size) => {
                        success_count += 1;
                        freed_bytes += file_size;
                        processed.push(SimilarityProcessedItem {
                            path: item.path,
                            action: "delete".to_string(),
                            destination: None,
                        });
                    }
                    Err(e) => {
                        failed_count += 1;
                        errors.push(SimilarityItemError {
                            path: item.path,
                            error: format!("删除失败: {e}"),
                        });
                    }
                }
            }
            _ => {
                failed_count += 1;
                errors.push(SimilarityItemError {
                    path: item.path,
                    error: format!("不支持的治理动作: {}", item.action),
                });
            }
        }
    }

    if let Some(directory) = result_directory.as_deref() {
        if let Err(error) = apply_decisions_to_similarity_result(
            Path::new(&state.data_dir),
            directory,
            &submitted_decisions,
            &processed,
        ) {
            failed_count += 1;
            errors.push(SimilarityItemError {
                path: directory.to_string(),
                error: format!("结果文件更新失败: {error}"),
            });
        }
    } else if !processed.is_empty() {
        // Compatibility for result sets created by older versions.
        if let Ok(mut database) = state.database.lock() {
            if let Ok(mut groups) = database.get_latest_similarity_groups() {
                remove_processed_items_from_groups(&mut groups, &processed);
                let _ = database.save_similarity_groups(&groups);
            }
        }
    }

    Ok(success(
        "similarity.decision.apply",
        request_id,
        SimilarityDecisionResult {
            success_count,
            failed_count,
            processed,
            errors,
            freed_bytes,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::database::{SimilarityGroup, SimilarityGroupItem};

    fn cached_feature(
        path: &str,
        frame_count: i64,
        feature_version: i64,
    ) -> database::ImageSimilarityFeatureRecord {
        database::ImageSimilarityFeatureRecord {
            path: path.to_string(),
            file_size: 1024,
            width: 100,
            height: 100,
            format: "gif".to_string(),
            clarity_score: 1.0,
            phash: "0000000000000000".to_string(),
            dhash: "0000000000000000".to_string(),
            color_hist_json: "[]".to_string(),
            frames_json: None,
            frame_count,
            feature_version,
        }
    }

    #[test]
    fn stale_animation_feature_cache_is_refreshed_once() {
        let legacy = database::SIMILARITY_FEATURE_VERSION - 1;

        // Animation-capable files force a re-extraction when the cache predates
        // multi-frame extraction.
        assert!(!feature_cache_is_current(
            &cached_feature("D:/images/a.gif", 5, legacy),
            Path::new("D:/images/a.gif")
        ));
        assert!(!feature_cache_is_current(
            &cached_feature("D:/images/a.webp", 5, legacy),
            Path::new("D:/images/a.webp")
        ));

        // Still images keep their existing cache.
        assert!(feature_cache_is_current(
            &cached_feature("D:/images/a.png", 1, legacy),
            Path::new("D:/images/a.png")
        ));

        // Current caches are always reused.
        assert!(feature_cache_is_current(
            &cached_feature("D:/images/a.gif", 5, database::SIMILARITY_FEATURE_VERSION),
            Path::new("D:/images/a.gif")
        ));
    }

    #[test]
    fn governance_moves_deletes_and_prunes_processed_files() {
        let root = std::env::temp_dir().join(format!(
            "anime-pic-manage-similarity-{}",
            uuid::Uuid::new_v4()
        ));
        let archive_directory = root.join("archive");
        fs::create_dir_all(&root).expect("temporary test directory should be created");

        let archived_source = root.join("archive-me.png");
        let deleted_source = root.join("delete-me.png");
        let same_directory_source = root.join("stay-here.png");
        fs::write(&archived_source, b"archive").expect("archive fixture should be written");
        fs::write(&deleted_source, b"delete").expect("delete fixture should be written");
        fs::write(&same_directory_source, b"same")
            .expect("same-directory fixture should be written");

        let same_directory_error = archive_similarity_file(&same_directory_source, &root)
            .expect_err("archiving into the source directory must not report success");
        assert_eq!(
            same_directory_error.kind(),
            std::io::ErrorKind::InvalidInput
        );
        assert!(same_directory_source.exists());

        let destination = archive_similarity_file(&archived_source, &archive_directory)
            .expect("archive should move the source file");
        let deleted_bytes =
            delete_similarity_file(&deleted_source).expect("delete should remove the source file");
        assert_eq!(deleted_bytes, 6);
        assert!(!archived_source.exists());
        assert_eq!(destination, archive_directory.join("archive-me.png"));
        assert!(destination.exists());
        assert!(!deleted_source.exists());

        let item = |path: String| SimilarityGroupItem {
            path,
            file_size: 1,
            dimensions: [1, 1],
            format: "png".to_string(),
            clarity_score: 1.0,
            is_recommended: false,
            recommend_reason: None,
            decision: Some("keep".to_string()),
        };
        let mut groups = vec![SimilarityGroup {
            group_id: "group-1".to_string(),
            group_type: "similar".to_string(),
            average_similarity: 0.9,
            items: vec![
                item(root.join("keep.png").to_string_lossy().into_owned()),
                item(archived_source.to_string_lossy().into_owned()),
                item(deleted_source.to_string_lossy().into_owned()),
            ],
        }];
        let processed = vec![
            SimilarityProcessedItem {
                path: archived_source.to_string_lossy().into_owned(),
                action: "archive".to_string(),
                destination: Some(destination.to_string_lossy().into_owned()),
            },
            SimilarityProcessedItem {
                path: deleted_source.to_string_lossy().into_owned(),
                action: "delete".to_string(),
                destination: None,
            },
        ];
        remove_processed_items_from_groups(&mut groups, &processed);
        assert!(
            groups.is_empty(),
            "groups with fewer than two items are removed"
        );

        fs::remove_dir_all(&root).expect("temporary test directory should be removed");
    }

    #[test]
    fn similarity_keep_decisions_are_saved_without_overwriting_history() {
        let root = std::env::temp_dir().join(format!(
            "anime-pic-manage-similarity-results-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&root).expect("temporary test directory should be created");
        let directory = root.join("library").to_string_lossy().into_owned();
        let item = |name: &str| SimilarityGroupItem {
            path: root.join(name).to_string_lossy().into_owned(),
            file_size: 10,
            dimensions: [1, 1],
            format: "png".to_string(),
            clarity_score: 1.0,
            is_recommended: name == "keep.png",
            recommend_reason: None,
            decision: None,
        };
        let payload = SimilarityScanPayload {
            directory: directory.clone(),
            total_scanned: 2,
            groups: vec![SimilarityGroup {
                group_id: "group-1".to_string(),
                group_type: "similar".to_string(),
                average_similarity: 0.9,
                items: vec![item("keep.png"), item("other.png")],
            }],
            duplicates_count: 1,
            potential_space_saved: 10,
            cancelled: false,
            completed_at: Some("2026-09-19T00:00:00.000Z".to_string()),
        };
        result_store::save_versioned(
            &root,
            ResultKind::Similarity,
            &directory,
            "2026-09-19T00:00:00.000Z",
            &payload,
        )
        .unwrap();
        result_store::save_versioned(
            &root,
            ResultKind::Similarity,
            &directory,
            "2026-09-19T00:01:00.000Z",
            &payload,
        )
        .unwrap();

        let decisions = vec![SimilarityItemDecision {
            path: root.join("keep.png").to_string_lossy().into_owned(),
            action: "keep".to_string(),
        }];
        assert!(apply_decisions_to_similarity_result(&root, &directory, &decisions, &[],).unwrap());
        let latest: SimilarityScanPayload =
            result_store::load_latest(&root, ResultKind::Similarity, &directory)
                .unwrap()
                .unwrap();
        assert_eq!(latest.groups[0].items[0].decision.as_deref(), Some("keep"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn archive_subdirectory_is_joined_by_the_native_path_api() {
        let request = SimilarityDecisionRequest {
            archive_directory: None,
            archive_directory_name: Some("_duplicates".to_string()),
            result_directory: Some(PathBuf::from("library").to_string_lossy().into_owned()),
            decisions: Vec::new(),
        };

        assert_eq!(
            resolve_archive_directory(&request),
            Some(PathBuf::from("library").join("_duplicates"))
        );
    }

    #[cfg(windows)]
    #[test]
    fn archive_subdirectory_preserves_windows_verbatim_paths() {
        let library = r"\\?\G:\P\涩图\画师\A\_ImagesTool";
        let request = SimilarityDecisionRequest {
            archive_directory: None,
            archive_directory_name: Some("_duplicates".to_string()),
            result_directory: Some(library.to_string()),
            decisions: Vec::new(),
        };

        let resolved = resolve_archive_directory(&request).unwrap();
        assert_eq!(resolved, PathBuf::from(library).join("_duplicates"));
        assert_eq!(
            resolved.to_string_lossy(),
            r"\\?\G:\P\涩图\画师\A\_ImagesTool\_duplicates"
        );
    }

    #[cfg(windows)]
    #[test]
    fn archive_moves_a_file_with_windows_verbatim_paths() {
        let root = std::env::temp_dir().join(format!(
            "anime-pic-manage-verbatim-{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&root).unwrap();
        let canonical_root = fs::canonicalize(&root).unwrap();
        let source = canonical_root.join("source.png");
        let archive = canonical_root.join("_duplicates");
        fs::write(&source, b"image").unwrap();

        let destination = archive_similarity_file(&source, &archive).unwrap();
        assert_eq!(destination, archive.join("source.png"));
        assert!(destination.exists());
        assert!(!source.exists());
        fs::remove_dir_all(root).unwrap();
    }
}
