use crate::commands::annotations::{
    annotation_records, worker_annotation_payload, MAX_ANNOTATIONS_PER_IMAGE,
};
use crate::commands::library::{
    clamp_scan_limit, merge_personal_model_artifact, total_discovered_from_payload,
};
use crate::database::{
    AnnotationBbox, AnnotationInput, AnnotationRecord, Database, PersonalPrototype,
    PERSONAL_EMBEDDING_DIMENSION,
};
use crate::ipc::core_error;
use crate::models::ScanLibraryResult;
use crate::state::{ScanControl, TrainingControl};
use serde_json::{json, Value};
use std::sync::Arc;

fn test_embedding(first: f32) -> Vec<f32> {
    let mut embedding = vec![0.0; PERSONAL_EMBEDDING_DIMENSION];
    embedding[0] = first;
    embedding
}

fn test_artifact() -> Value {
    json!({
        "schema_version": "1.0",
        "algorithm": "normalized_centroid_v1",
        "version": "personal-v7",
        "strict_threshold": 0.92,
        "min_margin": 0.12,
        "prototypes": [{
            "identity_id": "identity-a",
            "display_name": "旧名称",
            "embedding": test_embedding(1.0),
            "sample_count": 2,
        }],
    })
}

#[test]
fn merging_live_prototypes_overrides_and_appends_without_changing_calibration() {
    let artifact = test_artifact();
    let original = artifact.clone();
    let live = vec![
        PersonalPrototype {
            identity_id: "identity-a".to_string(),
            display_name: "新名称".to_string(),
            embedding: test_embedding(0.5),
            sample_count: 5,
        },
        PersonalPrototype {
            identity_id: "identity-b".to_string(),
            display_name: "新增身份".to_string(),
            embedding: test_embedding(0.25),
            sample_count: 1,
        },
    ];

    let merged = merge_personal_model_artifact(artifact, &live);
    assert_eq!(merged["version"], original["version"]);
    assert_eq!(merged["strict_threshold"], original["strict_threshold"]);
    assert_eq!(merged["min_margin"], original["min_margin"]);
    assert_eq!(merged["prototypes"].as_array().unwrap().len(), 2);
    assert_eq!(merged["prototypes"][0]["display_name"], "新名称");
    assert_eq!(merged["prototypes"][0]["sample_count"], 5);
    assert_eq!(merged["prototypes"][1]["identity_id"], "identity-b");
    assert_eq!(merged["prototypes"][1]["sample_count"], 1);
    assert_eq!(original["prototypes"][0]["display_name"], "旧名称");
    assert_eq!(original["prototypes"][0]["sample_count"], 2);
}

#[test]
fn invalid_artifact_or_live_prototype_is_left_untouched() {
    let invalid_artifact = json!({"version": "broken", "prototypes": []});
    let live = PersonalPrototype {
        identity_id: "identity-a".to_string(),
        display_name: "新名称".to_string(),
        embedding: test_embedding(1.0),
        sample_count: 1,
    };
    assert_eq!(
        merge_personal_model_artifact(invalid_artifact.clone(), &[live]),
        invalid_artifact
    );

    let artifact = test_artifact();
    let invalid_live = PersonalPrototype {
        identity_id: "identity-a".to_string(),
        display_name: "".to_string(),
        embedding: test_embedding(1.0),
        sample_count: 1,
    };
    assert_eq!(
        merge_personal_model_artifact(artifact.clone(), &[invalid_live]),
        artifact
    );
}

#[test]
fn envelope_contains_structured_error_fields() {
    let error = core_error("request-1", "TEST", "测试错误", None, false);
    assert_eq!(error.code, "TEST");
    assert_eq!(error.request_id, "request-1");
    assert!(!error.retryable);
}

#[test]
fn scan_limit_defaults_and_clamps_to_safe_preview_range() {
    assert_eq!(clamp_scan_limit(None), None);
    assert_eq!(clamp_scan_limit(Some(0)), None);
    assert_eq!(clamp_scan_limit(Some(12)), Some(12));
    assert_eq!(clamp_scan_limit(Some(999)), Some(999));
}

#[test]
fn total_discovered_prefers_safe_total_count_and_falls_back_to_images() {
    assert_eq!(
        total_discovered_from_payload(&json!({"total_count": 123}), 12),
        123
    );
    assert_eq!(
        total_discovered_from_payload(&json!({"total_count": -1}), 12),
        12
    );
    assert_eq!(
        total_discovered_from_payload(&json!({"total_count": "123"}), 12),
        12
    );
    assert_eq!(total_discovered_from_payload(&json!({}), 12), 12);
}

#[test]
fn scan_control_prevents_concurrent_runs_and_resets_cancel_state() {
    let control = ScanControl::new();
    assert!(control.try_start());
    assert!(!control.try_start());
    assert!(!control.is_cancelled());
    assert!(control.request_cancel());
    assert!(control.is_cancelled());
    control.finish();
    assert!(!control.request_cancel());
    assert!(control.try_start());
    assert!(!control.is_cancelled());
    control.finish();
}

#[test]
fn scan_control_pause_resume_blocks_at_boundary_and_resets_on_finish() {
    let control = Arc::new(ScanControl::new());
    assert!(control.try_start());
    control.update_progress(2, 6, "frame.png", "正在识别图片…");
    assert_eq!(control.request_pause().status, "pausing");
    assert_eq!(control.request_pause().status, "paused");
    let waiting = Arc::clone(&control);
    let (sender, receiver) = std::sync::mpsc::channel();
    let thread = std::thread::spawn(move || {
        let result = waiting.wait_if_paused_with(|_| {});
        sender.send(result).expect("pause waiter should report");
    });
    assert!(receiver
        .recv_timeout(std::time::Duration::from_millis(40))
        .is_err());
    assert_eq!(control.request_resume().status, "resumed");
    assert_eq!(control.request_resume().status, "running");
    let result = receiver
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("resume should wake paused scan");
    assert!(result.was_paused);
    assert!(!result.cancelled);
    thread.join().expect("pause waiter should exit");
    control.finish();
    assert_eq!(control.request_pause().status, "idle");
    assert_eq!(control.request_resume().status, "idle");
    assert!(!control.is_cancelled());
}

#[test]
fn scan_control_cancel_wakes_a_paused_scan() {
    let control = Arc::new(ScanControl::new());
    assert!(control.try_start());
    assert_eq!(control.request_pause().status, "pausing");
    let waiting = Arc::clone(&control);
    let (sender, receiver) = std::sync::mpsc::channel();
    let thread = std::thread::spawn(move || {
        let result = waiting.wait_if_paused_with(|_| {});
        sender.send(result).expect("cancel waiter should report");
    });
    assert!(receiver
        .recv_timeout(std::time::Duration::from_millis(40))
        .is_err());
    assert!(control.request_cancel());
    let result = receiver
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("cancel should wake paused scan");
    assert!(result.was_paused);
    assert!(result.cancelled);
    thread.join().expect("cancel waiter should exit");
    control.finish();
    assert!(!control.is_cancelled());
}

#[test]
fn training_control_rejects_concurrent_runs_and_releases_on_drop() {
    let control = TrainingControl::new();
    let guard = control.try_start().expect("first training should start");
    assert!(control.is_running());
    assert!(control.try_start().is_none());
    drop(guard);
    assert!(!control.is_running());
    let second = control
        .try_start()
        .expect("training should be startable after guard drop");
    assert!(control.is_running());
    drop(second);
    assert!(!control.is_running());
}

#[test]
fn worker_annotation_payload_converts_normalized_xywh_to_xyxy() {
    let annotation = AnnotationRecord {
        id: "annotation-1".to_string(),
        identity_id: "identity-1".to_string(),
        label_name: "边界分类".to_string(),
        bbox: AnnotationBbox {
            x: 0.0,
            y: 0.25,
            width: 1.0,
            height: 0.75,
        },
        source: "manual".to_string(),
        manually_adjusted: true,
    };
    let payload = worker_annotation_payload(&annotation);
    assert_eq!(payload["annotation_id"], "annotation-1");
    assert_eq!(payload["bbox"]["x1"], 0.0);
    assert_eq!(payload["bbox"]["y1"], 0.25);
    assert_eq!(payload["bbox"]["x2"], 1.0);
    assert_eq!(payload["bbox"]["y2"], 1.0);
    assert!(payload["bbox"].get("x").is_none());
    assert!(payload["bbox"].get("width").is_none());
}

#[test]
fn annotation_records_reuses_identity_for_same_normalized_name_in_one_request() {
    let database = Database::open_in_memory().expect("database should migrate");
    let annotations = vec![
        AnnotationInput {
            id: None,
            identity_id: None,
            label_name: "  分类甲  ".to_string(),
            bbox: AnnotationBbox {
                x: 0.0,
                y: 0.0,
                width: 0.2,
                height: 0.2,
            },
            source: "manual".to_string(),
            manually_adjusted: true,
        },
        AnnotationInput {
            id: None,
            identity_id: None,
            label_name: "分类甲".to_string(),
            bbox: AnnotationBbox {
                x: 0.3,
                y: 0.3,
                width: 0.2,
                height: 0.2,
            },
            source: "manual".to_string(),
            manually_adjusted: true,
        },
    ];
    let records = annotation_records(&database, annotations).expect("annotations valid");
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].identity_id, records[1].identity_id);
    assert_ne!(records[0].id, records[1].id);
}

#[test]
fn annotation_records_rejects_more_than_worker_limit() {
    let database = Database::open_in_memory().expect("database should migrate");
    let annotations = (0..=MAX_ANNOTATIONS_PER_IMAGE)
        .map(|index| AnnotationInput {
            id: None,
            identity_id: None,
            label_name: format!("分类{index}"),
            bbox: AnnotationBbox {
                x: 0.0,
                y: 0.0,
                width: 0.1,
                height: 0.1,
            },
            source: "manual".to_string(),
            manually_adjusted: true,
        })
        .collect();
    let error = annotation_records(&database, annotations).expect_err("limit must apply");
    assert!(error.contains("100"));
}

#[test]
fn scan_library_result_serializes_with_skip_annotated() {
    let result = ScanLibraryResult {
        directory: "C:/images".to_string(),
        total_discovered: 5,
        processed: 5,
        cancelled: false,
        results: vec![],
        errors: vec![],
        model_version: Some(2),
        model_name: Some("个人模型 v2".to_string()),
        skip_annotated: Some(true),
        completed_at: Some("2026-09-19T00:00:00.000Z".to_string()),
    };
    let val = serde_json::to_value(&result).expect("must serialize");
    assert_eq!(val["skip_annotated"], true);
}
