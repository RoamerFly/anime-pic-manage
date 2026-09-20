pub mod app_settings;
pub mod comfy;
pub mod commands;
pub mod database;
pub mod ipc;
pub mod kohya;
pub mod kohya_runner;
pub mod models;
pub mod portable;
pub mod result_store;
pub mod state;
pub mod worker_runtime;

#[cfg(test)]
mod tests;

use commands::*;
use database::{Database, WORKER_COMPUTE_DEVICE_SETTING, WORKER_RUNTIME_MODE_SETTING};
use state::AppState;
use std::fs;
use std::path::PathBuf;
use tauri::Manager;
use thiserror::Error;
use worker_runtime::{WorkerComputeDevice, WorkerManager, WorkerRuntimeMode};

#[derive(Debug, Error)]
pub enum CoreInitError {
    #[error("无法定位应用数据目录")]
    DataDirectory,
    #[error("{0}")]
    Database(#[from] database::DatabaseError),
}

pub fn resolve_portable_data_dir(app: &tauri::App) -> Result<PathBuf, CoreInitError> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()));

    let target_data_dir = if let Some(dir) = exe_dir {
        let data_folder = dir.join("data");
        let temp_folder = dir.join("temp");
        let _ = fs::create_dir_all(&data_folder);
        let _ = fs::create_dir_all(&temp_folder);
        data_folder
    } else {
        app.path()
            .app_data_dir()
            .map_err(|_| CoreInitError::DataDirectory)?
    };

    let target_db = target_data_dir.join("anime-pic-manage.sqlite3");

    // Safe automatic migration from legacy AppData if target database does not exist yet
    if !target_db.exists() {
        if let Ok(old_data_dir) = app.path().app_data_dir() {
            let old_db = old_data_dir.join("anime-pic-manage.sqlite3");
            if old_db.exists() {
                let _ = fs::copy(&old_db, &target_db);
            }
        }
    }

    Ok(target_data_dir)
}

pub fn init_state(app: &tauri::App) -> Result<AppState, CoreInitError> {
    let data_dir = resolve_portable_data_dir(app)?;
    if let Some(parent) = data_dir.parent() {
        let _ = fs::create_dir_all(parent.join("temp"));
    }
    let _ = fs::create_dir_all(data_dir.join("temp"));
    let database = Database::open(data_dir.join("anime-pic-manage.sqlite3"))?;
    let runtime_mode = database
        .get_setting_string(WORKER_RUNTIME_MODE_SETTING)?
        .and_then(|value| WorkerRuntimeMode::parse(&value))
        .unwrap_or_default();
    let compute_device = database
        .get_setting_string(WORKER_COMPUTE_DEVICE_SETTING)?
        .and_then(|value| WorkerComputeDevice::parse(&value))
        .unwrap_or_default();
    let app_settings = app_settings::load_app_settings(&database)
        .unwrap_or_else(|_| app_settings::AppSettings::default());
    let resource_dir = app.path().resource_dir().ok();
    let mut worker = WorkerManager::with_settings(
        resource_dir,
        runtime_mode,
        compute_device,
        app_settings.recognition_onnx_threads,
    );
    worker.set_cuda_runtime_dir(Some(app_settings.cuda_runtime_dir.clone()));
    worker.set_data_dir(Some(data_dir.to_string_lossy().into_owned()));
    worker.set_below_normal_priority(app_settings.background_priority != "normal");
    let state = AppState::new(database, data_dir.to_string_lossy().into_owned(), worker);
    if let Ok(mut kohya) = state.kohya.lock() {
        kohya.set_hf_cache_dir(Some(data_dir.join("hf-cache")));
    }
    Ok(state)
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .setup(|app| {
            let state = init_state(app)
                .map_err(|error| Box::<dyn std::error::Error>::from(error.to_string()))?;
            app.manage(state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_runtime_status,
            check_worker_health,
            get_gpu_devices,
            check_worker_compute,
            get_worker_runtime_settings,
            set_worker_runtime_mode,
            set_worker_compute_device,
            get_app_settings,
            update_app_settings,
            comfy_status,
            comfy_start,
            comfy_stop,
            comfy_models,
            comfy_templates,
            comfy_generate,
            comfy_cancel,
            comfy_open_output,
            comfy_template_detail,
            comfy_save_template_overrides,
            get_lora_export_candidates,
            export_lora_dataset,
            kohya_status,
            kohya_probe_environment,
            kohya_start_training,
            kohya_training_status,
            kohya_stop_training,
            kohya_install_lora,
            register_library_selection,
            register_preview_file,
            list_image_annotations,
            save_image_annotations,
            list_character_identities,
            get_personal_training_status,
            train_personal_model,
            activate_personal_model,
            rollback_personal_model,
            delete_personal_model,
            scan_library_preview,
            get_latest_library_scan_result,
            delete_latest_library_scan_result,
            cancel_library_scan,
            pause_library_scan,
            resume_library_scan,
            scan_similarity_preview,
            cancel_similarity_scan,
            get_similarity_scan_state,
            apply_similarity_decisions,
            get_cached_similarity_groups,
            get_latest_similarity_result,
            save_similarity_result_decisions,
            move_library_files,
            show_item_in_folder,
            open_external_url,
            cuda_runtime_status,
            install_cuda_runtime,
            get_model_inventory,
            set_active_recognizer,
            install_catalog_model,
            delete_catalog_model,
            adopt_legacy_model_cache,
            get_reference_library_status,
            build_reference_library,
            clear_reference_library,
            delete_library_file,
            check_for_update,
            install_update,
            cancel_update_download
        ])
        .run(tauri::generate_context!())
        .expect("error while running anime-pic-manage");
}
