#[cfg(test)]
mod tests {
    use crate::worker_runtime::discovery::*;
    use crate::worker_runtime::manager::WorkerManager;
    use crate::worker_runtime::process::*;
    use crate::worker_runtime::protocol::*;
    use crate::worker_runtime::types::*;
    use serde_json::json;
    use std::fs;
    use std::path::Path;
    use std::process::Command;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn finds_the_worker_and_env_inside_the_new_app_layout() {
        let root = std::env::temp_dir().join(format!("anime-layout-{}", uuid::Uuid::new_v4()));
        let app = root.join("app");
        fs::create_dir_all(app.join("runtime")).unwrap();
        fs::create_dir_all(app.join("env").join("Scripts")).unwrap();
        fs::create_dir_all(app.join("env").join("worker")).unwrap();
        fs::create_dir_all(root.join("models")).unwrap();
        fs::write(app.join("runtime").join("ai-worker.exe"), b"# worker").unwrap();
        fs::write(
            app.join("env").join("Scripts").join(python_name()),
            b"# python",
        )
        .unwrap();
        fs::write(
            app.join("env").join("worker").join("run_worker.py"),
            b"# launcher",
        )
        .unwrap();

        let candidates = resource_executable_candidates(&root);
        assert!(
            candidates.iter().any(
                |path| path.ends_with(Path::new("app").join("runtime").join(executable_name()))
            ),
            "candidates were {candidates:?}"
        );
        let spec = resolve_worker_candidate_for_mode(
            Some(&root),
            &root,
            &root.join("models"),
            WorkerRuntimeMode::Executable,
        )
        .expect("the packaged worker is discovered");
        match &spec {
            WorkerLaunchSpec::Executable { executable, .. } => {
                assert!(executable.ends_with(executable_name()))
            }
            other => panic!("unexpected spec: {other:?}"),
        }
        assert_eq!(spec.models_root(), root.join("models"));

        let env_spec = resolve_worker_candidate_for_mode(
            Some(&root),
            &root,
            &root.join("models"),
            WorkerRuntimeMode::EmbeddedEnv,
        )
        .expect("the bundled compatibility env is discovered");
        match env_spec {
            WorkerLaunchSpec::PythonScript { interpreter, .. } => {
                assert!(interpreter.ends_with(python_name()))
            }
            other => panic!("unexpected spec: {other:?}"),
        }
        fs::remove_dir_all(root).unwrap();
    }

    fn executable_name() -> &'static str {
        if cfg!(windows) {
            "ai-worker.exe"
        } else {
            "ai-worker"
        }
    }

    #[test]
    fn installer_layout_uses_the_packaged_worker_without_the_fallback_env() {
        // The NSIS installer ships `app\runtime\ai-worker.exe` and no `app\env`.
        let root = std::env::temp_dir().join(format!("anime-install-{}", uuid::Uuid::new_v4()));
        let app = root.join("app");
        fs::create_dir_all(app.join("runtime")).unwrap();
        fs::write(app.join("runtime").join(executable_name()), b"# worker").unwrap();

        let spec = resolve_worker_candidate_for_mode(
            Some(&root),
            &root,
            &root.join("models"),
            WorkerRuntimeMode::EmbeddedEnv,
        )
        .expect("the default mode still finds the packaged worker");
        match spec {
            WorkerLaunchSpec::Executable { executable, .. } => {
                assert!(executable.ends_with(executable_name()))
            }
            other => panic!("unexpected spec: {other:?}"),
        }

        // Models must land inside the application folder even before the folder
        // exists: a fresh install downloads them on demand.
        let models_root = resource_models_root(&root, Path::new("C:/build-machine/models"));
        assert_eq!(models_root, root.join("models"));

        // A packaged app never borrows the checkout it was compiled in.
        let missing = resolve_worker_candidate_for_mode(
            Some(&root.join("subdir")),
            &root,
            &root.join("models"),
            WorkerRuntimeMode::EmbeddedEnv,
        );
        assert!(missing.is_err());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn source_checkout_keeps_using_its_own_models_root() {
        // Development runs have no `app\` folder next to the binary, so the
        // configured checkout is still honoured.
        let root = std::env::temp_dir().join(format!("anime-dev-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let fallback = root.join("checkout-models");
        assert_eq!(resource_models_root(&root, &fallback), fallback);
        fs::remove_dir_all(root).unwrap();
    }

    fn python_name() -> &'static str {
        if cfg!(windows) {
            "python.exe"
        } else {
            "python"
        }
    }

    #[test]
    fn parses_compute_device_values_and_labels() {
        assert_eq!(
            WorkerComputeDevice::parse("auto"),
            Some(WorkerComputeDevice::Auto)
        );
        assert_eq!(
            WorkerComputeDevice::parse(" CPU "),
            Some(WorkerComputeDevice::Cpu)
        );
        assert_eq!(
            WorkerComputeDevice::parse("CUDA"),
            Some(WorkerComputeDevice::Cuda)
        );
        assert_eq!(
            WorkerComputeDevice::parse("gpu"),
            Some(WorkerComputeDevice::Cuda)
        );
        assert_eq!(WorkerComputeDevice::parse("tpu"), None);
        assert_eq!(WorkerComputeDevice::default(), WorkerComputeDevice::Auto);
        assert_eq!(WorkerComputeDevice::Cuda.as_str(), "cuda");
        assert!(WorkerComputeDevice::Cuda.label().contains("CUDA"));
    }

    #[test]
    fn recognition_payloads_carry_the_configured_compute_device() {
        let mut manager = WorkerManager::with_settings(
            None,
            WorkerRuntimeMode::Executable,
            WorkerComputeDevice::Cuda,
            4,
        );
        let payload =
            manager.with_compute_device("recognition.image", json!({"image_path": "sample.png"}));
        assert_eq!(payload["device"], "cuda");
        assert_eq!(payload["image_path"], "sample.png");

        // An explicit per-request device wins over the stored setting.
        let explicit = manager.with_compute_device(
            "recognition.image",
            json!({"image_path": "sample.png", "device": "cpu"}),
        );
        assert_eq!(explicit["device"], "cpu");

        // Unrelated message types are forwarded untouched.
        let enumerate =
            manager.with_compute_device("images.enumerate", json!({"directory": "D:\\"}));
        assert!(enumerate.get("device").is_none());

        manager.set_compute_device(WorkerComputeDevice::Cpu);
        assert_eq!(manager.compute_device(), WorkerComputeDevice::Cpu);
        let rewritten =
            manager.with_compute_device("recognition.embedding", json!({"image_path": "x.png"}));
        assert_eq!(rewritten["device"], "cpu");
        assert_eq!(rewritten["onnx_threads"], 4);

        manager.set_onnx_threads(7);
        assert_eq!(manager.onnx_threads(), 7);
        let threaded =
            manager.with_compute_device("recognition.image", json!({"image_path": "y.png"}));
        assert_eq!(threaded["onnx_threads"], 7);
    }

    #[test]
    fn parses_and_validates_health_response() {
        let line = r#"{"schema_version":"1.0","request_id":"r1","task_id":"t1","message_type":"health.result","payload":{"status":"ok","worker_version":"0.1.0","schema_version":"1.0","model_count":2},"error":null}"#;
        let result = parse_health_response(line, "r1", "t1").expect("health should parse");
        assert_eq!(result.status, "ok");
        assert_eq!(result.version, "0.1.0");
        assert_eq!(result.model_count, 2);
    }

    #[test]
    fn rejects_wrong_request_and_worker_error() {
        let line = r#"{"schema_version":"1.0","request_id":"other","task_id":"t1","message_type":"health.result","payload":{},"error":null}"#;
        assert!(matches!(
            parse_health_response(line, "r1", "t1"),
            Err(WorkerRuntimeError::Protocol(_))
        ));
        let line = r#"{"schema_version":"1.0","request_id":"r1","task_id":"t1","message_type":"health.result","payload":{},"error":{"code":"BOOT_FAILED","message":"bad","detail":null,"retryable":true}}"#;
        assert!(matches!(
            parse_health_response(line, "r1", "t1"),
            Err(WorkerRuntimeError::WorkerResponse { code, .. }) if code == "BOOT_FAILED"
        ));
    }

    #[test]
    fn parses_generic_response_payload_and_message_type() {
        let line = r#"{"schema_version":"1.0","request_id":"r1","task_id":"t1","message_type":"recognition.image.result","payload":{"path":"image.png","people":[]},"error":null}"#;
        let payload = parse_worker_response(line, "r1", "t1", "recognition.image.result")
            .expect("generic response should parse");
        assert_eq!(payload["path"], "image.png");
        assert!(payload["people"].is_array());

        let wrong_type = parse_worker_response(line, "r1", "t1", "images.enumerate.result");
        assert!(matches!(
            wrong_type,
            Err(WorkerRuntimeError::Protocol(message)) if message.contains("message_type")
        ));
    }

    #[test]
    fn parses_generic_worker_error() {
        let line = r#"{"schema_version":"1.0","request_id":"r1","task_id":"t1","message_type":"recognition.image.result","payload":{},"error":{"code":"IMAGE_READ_FAILED","message":"bad image","detail":"details","retryable":true}}"#;
        assert!(matches!(
            parse_worker_response(line, "r1", "t1", "recognition.image.result"),
            Err(WorkerRuntimeError::WorkerResponse { code, message })
                if code == "IMAGE_READ_FAILED" && message == "bad image"
        ));
    }

    #[test]
    fn parses_runtime_capabilities_and_required_features() {
        let payload = json!({
            "status": "ok",
            "ready": true,
            "dghs_imgutils": {
                "distribution": "dghs-imgutils",
                "version": "0.19.0",
                "status": "ready",
                "error": null
            },
            "onnxruntime": {
                "distribution": "onnxruntime",
                "version": "1.20.1",
                "status": "ready",
                "error": null
            },
            "features": {
                "imgutils.preprocess.pillow": {
                    "module": "imgutils.preprocess.pillow",
                    "status": "ready",
                    "error": null
                },
                "imgutils.generic.yolo": {
                    "module": "imgutils.generic.yolo",
                    "status": "ready",
                    "error": null
                },
                "imgutils.data": {
                    "module": "imgutils.data",
                    "status": "ready",
                    "error": null
                }
            },
            "errors": []
        });
        let capabilities = parse_capabilities_payload(payload).expect("capabilities should parse");
        assert!(capabilities.ready);
        assert_eq!(
            capabilities.dghs_imgutils.version.as_deref(),
            Some("0.19.0")
        );
        assert_eq!(capabilities.onnxruntime.version.as_deref(), Some("1.20.1"));
        assert_eq!(capabilities.features.len(), 3);
        assert!(capabilities.compute.is_none());
    }

    #[test]
    fn parses_compute_capability_report() {
        let payload = json!({
            "status": "ok",
            "ready": true,
            "dghs_imgutils": {
                "distribution": "dghs-imgutils",
                "version": "0.19.0",
                "status": "ready",
                "error": null
            },
            "onnxruntime": {
                "distribution": "onnxruntime-gpu",
                "version": "1.20.1",
                "status": "ready",
                "error": null
            },
            "compute": {
                "distribution": "onnxruntime-gpu",
                "available_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "cuda_available": true,
                "error": null
            },
            "features": {
                "imgutils.preprocess.pillow": {
                    "module": "imgutils.preprocess.pillow",
                    "status": "ready",
                    "error": null
                },
                "imgutils.generic.yolo": {
                    "module": "imgutils.generic.yolo",
                    "status": "ready",
                    "error": null
                },
                "imgutils.data": {
                    "module": "imgutils.data",
                    "status": "ready",
                    "error": null
                }
            },
            "errors": []
        });
        let capabilities = parse_capabilities_payload(payload).expect("capabilities should parse");
        let compute = capabilities.compute.expect("compute report should parse");
        assert!(compute.cuda_available);
        assert_eq!(compute.distribution.as_deref(), Some("onnxruntime-gpu"));
        assert_eq!(compute.available_providers.len(), 2);
    }

    #[test]
    fn parses_unavailable_runtime_capability_without_claiming_ready() {
        let payload = json!({
            "status": "unavailable",
            "ready": false,
            "dghs_imgutils": {
                "distribution": "dghs-imgutils",
                "version": null,
                "status": "unavailable",
                "error": {
                    "code": "DEPENDENCY_NOT_INSTALLED",
                    "message": "missing",
                    "detail": null,
                    "retryable": false
                }
            },
            "onnxruntime": {
                "distribution": "onnxruntime",
                "version": "1.20.1",
                "status": "ready",
                "error": null
            },
            "features": {
                "imgutils.preprocess.pillow": {"module": "imgutils.preprocess.pillow", "status": "ready", "error": null},
                "imgutils.generic.yolo": {"module": "imgutils.generic.yolo", "status": "ready", "error": null},
                "imgutils.data": {"module": "imgutils.data", "status": "ready", "error": null}
            },
            "errors": [{
                "code": "DEPENDENCY_NOT_INSTALLED",
                "message": "missing",
                "detail": null,
                "retryable": false
            }]
        });
        let capabilities =
            parse_capabilities_payload(payload).expect("unavailable report should parse");
        assert!(!capabilities.ready);
        assert_eq!(capabilities.status, "unavailable");
        assert_eq!(capabilities.errors[0].code, "DEPENDENCY_NOT_INSTALLED");
    }

    #[test]
    fn rejects_capabilities_missing_required_feature() {
        let payload = json!({
            "status": "ok",
            "ready": true,
            "dghs_imgutils": {"distribution": "dghs-imgutils", "version": "0.19.0", "status": "ready", "error": null},
            "onnxruntime": {"distribution": "onnxruntime", "version": "1.20.1", "status": "ready", "error": null},
            "features": {},
            "errors": []
        });
        assert!(matches!(
            parse_capabilities_payload(payload),
            Err(WorkerRuntimeError::Protocol(message)) if message.contains("缺少必需 feature")
        ));
    }

    #[test]
    fn only_connection_or_protocol_errors_clear_worker_process() {
        assert!(!should_clear_process(&WorkerRuntimeError::WorkerResponse {
            code: "IMAGE_READ_FAILED".to_string(),
            message: "bad image".to_string(),
        }));
        assert!(should_clear_process(&WorkerRuntimeError::Io(
            "closed".to_string()
        )));
        assert!(should_clear_process(&WorkerRuntimeError::Protocol(
            "invalid".to_string()
        )));
        assert!(should_clear_process(&WorkerRuntimeError::Spawn(
            "failed".to_string()
        )));
        assert!(should_clear_process(&WorkerRuntimeError::NotFound));
        assert!(should_clear_process(&WorkerRuntimeError::UvMissing));
    }

    #[test]
    fn background_process_flags_are_platform_specific() {
        assert_eq!(
            background_creation_flags(),
            if cfg!(windows) {
                // CREATE_NO_WINDOW | BELOW_NORMAL_PRIORITY_CLASS: inference
                // must not starve the desktop or the user's other apps.
                Some(0x0800_0000 | 0x0000_4000)
            } else {
                None
            }
        );

        // This also exercises the non-Windows no-op implementation and keeps
        // the Windows `CommandExt` call covered by a cross-platform test.
        let mut command = Command::new("worker-test-placeholder");
        configure_background_command(&mut command);
    }

    /// The Worker's Hugging Face cache has to follow the package, because the
    /// settings page reports this path and users copy the folder around.
    #[test]
    fn worker_cache_follows_the_application_data_folder() {
        let mut manager = WorkerManager::new(None);
        assert!(manager.hf_cache_dir().is_none());

        manager.set_data_dir(Some(r"D:\Apps\AnimePicManage\data".to_string()));
        assert_eq!(
            manager.hf_cache_dir(),
            Some(std::path::PathBuf::from(
                r"D:\Apps\AnimePicManage\data\hf-cache"
            ))
        );

        // An unset or blank folder must not silently become a relative path.
        manager.set_data_dir(Some("   ".to_string()));
        assert!(manager.hf_cache_dir().is_none());
    }

    #[test]
    fn source_candidate_requires_verified_paths() {
        let root = std::env::temp_dir().join(format!(
            "anime-pic-worker-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let project = root.join("ai-worker");
        let models = root.join("models");
        fs::create_dir_all(&project).expect("project dir");
        fs::create_dir_all(&models).expect("models dir");
        fs::write(project.join("run_worker.py"), "print('ok')").expect("script");
        let spec = resolve_worker_candidate(None, &project, &models).expect("source candidate");
        assert!(matches!(spec, WorkerLaunchSpec::UvScript { .. }));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn source_candidate_prefers_project_venv_python() {
        let root = std::env::temp_dir().join(format!(
            "anime-pic-worker-venv-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let project = root.join("ai-worker");
        let models = root.join("models");
        fs::create_dir_all(&project).expect("project dir");
        fs::create_dir_all(&models).expect("models dir");
        fs::write(project.join("run_worker.py"), "print('ok')").expect("script");

        let interpreter = project.join(if cfg!(windows) {
            ".venv/Scripts/python.exe"
        } else {
            ".venv/bin/python"
        });
        fs::create_dir_all(interpreter.parent().expect("interpreter parent")).expect("venv dir");
        fs::write(&interpreter, "placeholder").expect("interpreter");

        let spec = resolve_worker_candidate(None, &project, &models).expect("source candidate");
        match spec {
            WorkerLaunchSpec::PythonScript {
                interpreter: resolved,
                project_dir,
                script,
                ..
            } => {
                assert_eq!(
                    resolved,
                    fs::canonicalize(interpreter).expect("canonical interpreter")
                );
                assert_eq!(
                    project_dir,
                    fs::canonicalize(&project).expect("canonical project")
                );
                assert_eq!(
                    script,
                    fs::canonicalize(project.join("run_worker.py")).expect("canonical script")
                );
            }
            other => panic!("expected direct venv Python launch, got {other:?}"),
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn embedded_env_candidate_uses_portable_resource_layout() {
        let root = std::env::temp_dir().join(format!(
            "anime-pic-worker-resource-env-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let env = root.join("env");
        let script = env.join("worker/run_worker.py");
        let interpreter = env.join(venv_python_relative());
        fs::create_dir_all(script.parent().expect("script parent")).expect("worker dir");
        fs::create_dir_all(interpreter.parent().expect("interpreter parent"))
            .expect("interpreter dir");
        fs::write(&script, "print('ok')").expect("script");
        fs::write(&interpreter, "placeholder").expect("interpreter");

        let spec = resolve_worker_candidate_for_mode(
            Some(&root),
            &root.join("missing-local-project"),
            &root.join("models"),
            WorkerRuntimeMode::EmbeddedEnv,
        )
        .expect("resource env candidate");
        match spec {
            WorkerLaunchSpec::PythonScript {
                interpreter: resolved,
                project_dir,
                script: resolved_script,
                ..
            } => {
                assert_eq!(
                    resolved,
                    fs::canonicalize(interpreter).expect("canonical interpreter")
                );
                assert_eq!(
                    project_dir,
                    fs::canonicalize(env.join("worker")).expect("canonical worker dir")
                );
                assert_eq!(
                    resolved_script,
                    fs::canonicalize(script).expect("canonical script")
                );
            }
            other => panic!("expected portable env launch, got {other:?}"),
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn executable_mode_does_not_fall_back_to_python() {
        let root = std::env::temp_dir().join(format!(
            "anime-pic-worker-exe-mode-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let project = root.join("ai-worker");
        fs::create_dir_all(&project).expect("project dir");
        fs::write(project.join("run_worker.py"), "print('ok')").expect("script");
        fs::create_dir_all(
            project
                .join(".venv")
                .join(venv_python_relative())
                .parent()
                .expect("venv parent"),
        )
        .expect("venv dir");
        fs::write(
            project.join(".venv").join(venv_python_relative()),
            "placeholder",
        )
        .expect("interpreter");
        let error = resolve_worker_candidate_for_mode(
            Some(&root),
            &project,
            &root.join("models"),
            WorkerRuntimeMode::Executable,
        )
        .expect_err("executable mode should require an exe");
        assert!(matches!(error, WorkerRuntimeError::NotFound));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn executable_mode_finds_internal_runtime_worker() {
        let root = std::env::temp_dir().join(format!(
            "anime-pic-worker-runtime-layout-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("clock")
                .as_nanos()
        ));
        let runtime = root.join("runtime");
        let executable = runtime.join(if cfg!(windows) {
            "ai-worker.exe"
        } else {
            "ai-worker"
        });
        fs::create_dir_all(&runtime).expect("runtime dir");
        fs::write(&executable, "placeholder").expect("worker executable");

        let spec = resolve_worker_candidate_for_mode(
            Some(&root),
            &root.join("missing-local-project"),
            &root.join("models"),
            WorkerRuntimeMode::Executable,
        )
        .expect("internal runtime worker");
        match spec {
            WorkerLaunchSpec::Executable {
                executable: resolved,
                working_dir,
                ..
            } => {
                assert_eq!(
                    resolved,
                    fs::canonicalize(&executable).expect("canonical worker")
                );
                assert_eq!(
                    working_dir,
                    fs::canonicalize(&runtime).expect("canonical runtime dir")
                );
            }
            other => panic!("expected internal runtime executable launch, got {other:?}"),
        }
        let _ = fs::remove_dir_all(root);
    }
}
