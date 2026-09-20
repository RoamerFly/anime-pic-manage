use super::*;

#[test]
fn initial_migration_creates_core_tables() {
    let database = Database::open_in_memory().expect("in-memory database should migrate");
    let health = database
        .health()
        .expect("database health should be readable");
    assert_eq!(health.schema_version, CURRENT_SCHEMA_VERSION);
    assert!(health.table_count >= 12);
    for table in [
        "libraries",
        "images",
        "detections",
        "recognitions",
        "classification_plans",
        "file_operations",
        "similarity_embeddings",
        "similarity_groups",
        "jobs",
        "settings",
        "character_identities",
        "annotation_images",
        "annotation_revisions",
        "annotation_revision_items",
        "image_annotations_current",
        "personal_embeddings",
        "personal_model_versions",
        "personal_training_state",
        "personal_training_settings",
        "image_similarity_features",
        "similarity_group_records",
    ] {
        assert!(
            database
                .table_exists(table)
                .expect("table lookup should work"),
            "missing {table}"
        );
    }
}

#[test]
fn worker_runtime_setting_round_trips_as_versioned_json() {
    let database = Database::open_in_memory().expect("database should migrate");
    assert_eq!(
        database
            .get_setting_string(WORKER_RUNTIME_MODE_SETTING)
            .expect("setting lookup should work"),
        None
    );
    database
        .set_setting_string(WORKER_RUNTIME_MODE_SETTING, "embedded_env")
        .expect("setting write should work");
    assert_eq!(
        database
            .get_setting_string(WORKER_RUNTIME_MODE_SETTING)
            .expect("setting lookup should work")
            .as_deref(),
        Some("embedded_env")
    );
    database
        .set_setting_string(WORKER_RUNTIME_MODE_SETTING, "executable")
        .expect("setting update should work");
    assert_eq!(
        database
            .get_setting_string(WORKER_RUNTIME_MODE_SETTING)
            .expect("setting lookup should work")
            .as_deref(),
        Some("executable")
    );
}

#[test]
fn annotation_revisions_are_optimistic_and_soft_delete_current_rows() {
    let mut database = Database::open_in_memory().expect("database should migrate");
    let identity_id = uuid::Uuid::new_v4().to_string();
    let annotation_id = uuid::Uuid::new_v4().to_string();
    let annotation = AnnotationRecord {
        id: annotation_id.clone(),
        identity_id: identity_id.clone(),
        label_name: "测试分类".to_string(),
        bbox: AnnotationBbox {
            x: 0.1,
            y: 0.2,
            width: 0.3,
            height: 0.4,
        },
        source: "manual".to_string(),
        manually_adjusted: true,
    };
    let embedding = vec![1.0_f32; PERSONAL_EMBEDDING_DIMENSION];
    let first = database
        .save_image_annotations(
            "C:/images/a.png",
            [100, 80],
            std::slice::from_ref(&annotation),
            &[EmbeddingSample {
                annotation_id: annotation_id.clone(),
                identity_id,
                embedding,
                manually_verified: true,
            }],
            Some(0),
        )
        .expect("first annotation save should succeed");
    assert_eq!(first.revision, 1);
    assert_eq!(first.annotations.len(), 1);
    assert_eq!(first.learned_samples, 1);
    let prototypes = database
        .personal_prototypes()
        .expect("prototype query should succeed");
    assert_eq!(prototypes.len(), 1);
    assert_eq!(prototypes[0].sample_count, 1);
    assert_eq!(prototypes[0].embedding.len(), PERSONAL_EMBEDDING_DIMENSION);
    assert!(matches!(
        database.save_image_annotations("C:/images/a.png", [100, 80], &[], &[], Some(0)),
        Err(DatabaseError::RevisionConflict {
            current_revision: 1
        })
    ));
    let second = database
        .save_image_annotations("C:/images/a.png", [100, 80], &[], &[], Some(1))
        .expect("deletion save should succeed");
    assert_eq!(second.revision, 2);
    assert!(second.annotations.is_empty());
    assert_eq!(
            database
                .connection
                .query_row(
                    "SELECT deleted FROM image_annotations_current WHERE image_path = ?1 AND annotation_id = ?2",
                    rusqlite::params!["C:/images/a.png", annotation_id],
                    |row| row.get::<_, i64>(0),
                )
                .expect("soft deleted row should remain"),
            1
        );
    assert_eq!(
        database.personal_prototypes().expect("prototype query"),
        Vec::new()
    );
}

#[test]
fn v3_training_status_requires_three_current_samples_per_class() {
    let mut database = Database::open_in_memory().expect("database should migrate");
    let identity_id = uuid::Uuid::new_v4().to_string();
    let embedding = vec![1.0_f32; PERSONAL_EMBEDDING_DIMENSION];
    for index in 0..2 {
        let annotation_id = uuid::Uuid::new_v4().to_string();
        let annotation = AnnotationRecord {
            id: annotation_id.clone(),
            identity_id: identity_id.clone(),
            label_name: "同一分类".to_string(),
            bbox: AnnotationBbox {
                x: 0.1,
                y: 0.1,
                width: 0.5,
                height: 0.5,
            },
            source: "manual".to_string(),
            manually_adjusted: true,
        };
        database
            .save_image_annotations(
                &format!("C:/images/{index}.png"),
                [100, 100],
                std::slice::from_ref(&annotation),
                &[EmbeddingSample {
                    annotation_id,
                    identity_id: identity_id.clone(),
                    embedding: embedding.clone(),
                    manually_verified: true,
                }],
                Some(0),
            )
            .expect("sample should save");
    }
    let readiness = database
        .training_readiness()
        .expect("readiness should be readable");
    assert_eq!(readiness.verified_sample_count, 2);
    assert_eq!(readiness.verified_class_count, 1);
    assert!(!readiness.ready);
    assert_eq!(readiness.eligible_class_count, 0);
}

#[test]
fn v3_trained_versions_are_transactional_and_only_valid_artifacts_activate() {
    let mut database = Database::open_in_memory().expect("database should migrate");
    let artifact = serde_json::json!({
        "schema_version": "1.0",
        "algorithm": "normalized_centroid_v1",
        "version": "personal-v2",
        "strict_threshold": 0.86,
        "min_margin": 0.08,
        "prototypes": [{
            "identity_id": "identity-1",
            "display_name": "分类一",
            "embedding": vec![1.0_f32; PERSONAL_EMBEDDING_DIMENSION],
            "sample_count": 3
        }]
    });
    let model = TrainedPersonalModel {
        artifact: artifact.clone(),
        metrics: serde_json::json!({"sample_count": 3, "class_count": 1}),
        algorithm: "normalized_centroid_v1".to_string(),
        warnings: vec!["测试警告".to_string()],
        eligible_for_activation: true,
        sample_count: 3,
        class_count: 1,
    };
    let status = database
        .save_trained_personal_model(&model, 2, true, false)
        .expect("model should save");
    assert_eq!(status.active_version, Some(2));
    assert_eq!(status.versions[0].status, "active");
    assert_eq!(
        database
            .active_personal_model_artifact()
            .expect("artifact should read"),
        Some(artifact.clone())
    );
    assert!(matches!(
        database.save_trained_personal_model(&model, 3, true, false),
        Err(rusqlite::Error::InvalidQuery)
    ));
    let second = database
        .save_trained_personal_model(
            &TrainedPersonalModel {
                artifact: {
                    let mut artifact = artifact.clone();
                    artifact["version"] = serde_json::json!("personal-v3");
                    artifact
                },
                eligible_for_activation: false,
                ..model.clone()
            },
            3,
            true,
            false,
        )
        .expect("draft model should save");
    assert_eq!(second.active_version, Some(2));
    assert_eq!(second.versions[0].status, "draft");
    let activated_at: Option<String> = database
        .connection
        .query_row(
            "SELECT activated_at FROM personal_model_versions WHERE version = 3",
            [],
            |row| row.get(0),
        )
        .expect("draft activation timestamp should be queryable");
    assert!(activated_at.is_none());
    assert!(database.activate_personal_model(3).is_err());
    let rolled_back = database
        .rollback_personal_model(Some(2))
        .expect("rollback should activate valid version");
    assert_eq!(rolled_back.active_version, Some(2));
}

#[test]
fn delete_personal_model_version_removes_inactive_and_protects_active() {
    let mut database = Database::open_in_memory().expect("database should migrate");
    let artifact = serde_json::json!({
        "schema_version": "1.0",
        "algorithm": "normalized_centroid_v1",
        "version": "personal-v2",
        "strict_threshold": 0.86,
        "min_margin": 0.08,
        "prototypes": [{
            "identity_id": "identity-1",
            "display_name": "分类一",
            "embedding": vec![1.0_f32; PERSONAL_EMBEDDING_DIMENSION],
            "sample_count": 3
        }]
    });
    let model = TrainedPersonalModel {
        artifact: artifact.clone(),
        metrics: serde_json::json!({"sample_count": 3, "class_count": 1}),
        algorithm: "normalized_centroid_v1".to_string(),
        warnings: vec![],
        eligible_for_activation: true,
        sample_count: 3,
        class_count: 1,
    };
    database
        .save_trained_personal_model(&model, 2, true, false)
        .expect("v2 should save as active");

    let mut artifact_v3 = artifact.clone();
    artifact_v3["version"] = serde_json::json!("personal-v3");
    let model_v3 = TrainedPersonalModel {
        artifact: artifact_v3,
        ..model.clone()
    };
    database
        .save_trained_personal_model(&model_v3, 3, false, false)
        .expect("v3 should save as draft");

    assert!(matches!(
        database.delete_personal_model_version(2),
        Err(rusqlite::Error::InvalidQuery)
    ));

    let status = database
        .delete_personal_model_version(3)
        .expect("v3 should be deleted");
    assert!(!status.versions.iter().any(|v| v.version == 3));

    assert!(matches!(
        database.delete_personal_model_version(99),
        Err(rusqlite::Error::QueryReturnedNoRows)
    ));
}

#[test]
fn overwrite_personal_model_version_updates_v2_and_protects_v1() {
    let mut database = Database::open_in_memory().expect("database should migrate");
    let artifact = serde_json::json!({
        "schema_version": "1.0",
        "algorithm": "normalized_centroid_v1",
        "version": "personal-v2",
        "strict_threshold": 0.86,
        "min_margin": 0.08,
        "prototypes": [{
            "identity_id": "identity-1",
            "display_name": "分类一",
            "embedding": vec![1.0_f32; PERSONAL_EMBEDDING_DIMENSION],
            "sample_count": 3
        }]
    });
    let model = TrainedPersonalModel {
        artifact: artifact.clone(),
        metrics: serde_json::json!({"sample_count": 3, "class_count": 1}),
        algorithm: "normalized_centroid_v1".to_string(),
        warnings: vec![],
        eligible_for_activation: true,
        sample_count: 3,
        class_count: 1,
    };

    // Attempting to overwrite v1 must fail
    let mut artifact_v1 = artifact.clone();
    artifact_v1["version"] = serde_json::json!("personal-v1");
    let model_v1 = TrainedPersonalModel {
        artifact: artifact_v1,
        ..model.clone()
    };
    assert!(matches!(
        database.save_trained_personal_model(&model_v1, 1, true, true),
        Err(rusqlite::Error::InvalidQuery)
    ));

    // Save v2 normally
    database
        .save_trained_personal_model(&model, 2, true, false)
        .expect("v2 should save as active");

    // Overwrite v2 with updated sample count
    let model_v2_updated = TrainedPersonalModel {
        sample_count: 10,
        ..model.clone()
    };
    let status = database
        .save_trained_personal_model(&model_v2_updated, 2, true, true)
        .expect("v2 should overwrite successfully");

    assert_eq!(status.active_version, Some(2));
    assert_eq!(status.versions.len(), 2); // builtin-v1 and personal-v2
    let v2 = status.versions.iter().find(|v| v.version == 2).unwrap();
    assert_eq!(v2.sample_count, 10);
}

#[test]
fn annotated_paths_and_update_image_path_tracks_movement() {
    let mut database = Database::open_in_memory().expect("database should migrate");
    let identity_id = "char-1".to_string();
    let annotation_id = uuid::Uuid::new_v4().to_string();
    let annotation = AnnotationRecord {
        id: annotation_id.clone(),
        identity_id: identity_id.clone(),
        label_name: "测试角色".to_string(),
        bbox: AnnotationBbox {
            x: 0.2,
            y: 0.2,
            width: 0.4,
            height: 0.4,
        },
        source: "manual".to_string(),
        manually_adjusted: true,
    };
    database
        .save_image_annotations(
            "C:/images/test.png",
            [200, 150],
            std::slice::from_ref(&annotation),
            &[EmbeddingSample {
                annotation_id,
                identity_id,
                embedding: vec![0.5_f32; PERSONAL_EMBEDDING_DIMENSION],
                manually_verified: true,
            }],
            Some(0),
        )
        .expect("annotation should save");

    let paths = database
        .list_annotated_image_paths("C:/images")
        .expect("should list paths");
    assert!(paths.contains("C:/images/test.png"));

    let scan_result = database
        .get_annotation_as_scan_result("C:/images/test.png")
        .expect("should construct scan result")
        .expect("should have result");
    assert_eq!(scan_result["path"], "C:/images/test.png");
    assert_eq!(scan_result["people"].as_array().unwrap().len(), 1);

    // Test updating image path on move
    database
        .update_image_path("C:/images/test.png", "D:/moved/test.png")
        .expect("update image path should succeed");

    let old_annos = database
        .list_image_annotations("C:/images/test.png")
        .expect("should read old path");
    assert!(old_annos.annotations.is_empty());

    let new_annos = database
        .list_image_annotations("D:/moved/test.png")
        .expect("should read new path");
    assert_eq!(new_annos.annotations.len(), 1);
    assert_eq!(new_annos.annotations[0].label_name, "测试角色");
}

#[test]
fn identity_sample_queries_feed_the_lora_dataset_export() {
    let mut database = Database::open_in_memory().expect("database should migrate");

    let annotation =
        |identity_id: &str, label_name: &str, source: &str, adjusted: bool| AnnotationRecord {
            id: uuid::Uuid::new_v4().to_string(),
            identity_id: identity_id.to_string(),
            label_name: label_name.to_string(),
            bbox: AnnotationBbox {
                x: 0.1,
                y: 0.2,
                width: 0.5,
                height: 0.6,
            },
            source: source.to_string(),
            manually_adjusted: adjusted,
        };

    database
        .save_image_annotations(
            "C:/images/one.png",
            [800, 1200],
            &[annotation("char-1", "character_alpha", "manual", true)],
            &[],
            Some(0),
        )
        .expect("first annotation should save");
    database
        .save_image_annotations(
            "C:/images/two.png",
            [800, 1200],
            &[
                // Same character twice in one picture still counts as one sample.
                annotation("char-1", "character_alpha", "model", false),
                annotation("char-2", "character_beta", "model", false),
            ],
            &[],
            Some(0),
        )
        .expect("second annotation should save");

    let summaries = database
        .list_identity_sample_summaries()
        .expect("summaries should be readable");
    assert_eq!(summaries.len(), 2);
    assert_eq!(summaries[0].identity_id, "char-1");
    assert_eq!(summaries[0].image_count, 2);
    assert_eq!(summaries[0].manual_count, 1);
    assert_eq!(summaries[1].identity_id, "char-2");
    assert_eq!(summaries[1].image_count, 1);
    assert_eq!(summaries[1].manual_count, 0);

    let samples = database
        .list_identity_samples("char-1")
        .expect("samples should be readable");
    assert_eq!(samples.len(), 2);
    assert_eq!(samples[0].image_width, 800);
    assert_eq!(samples[0].image_height, 1200);
    assert_eq!(samples[0].bbox_width, 0.5);
    assert!(samples.iter().any(|sample| sample.manually_adjusted));

    // Moving a file keeps the exported sample list in sync with the new path.
    database
        .update_image_path("C:/images/two.png", "D:/moved/two.png")
        .expect("moving the annotated file should succeed");
    let moved = database
        .list_identity_samples("char-1")
        .expect("samples should still be readable");
    assert!(moved.iter().any(|sample| sample.path == "D:/moved/two.png"));
    assert!(moved
        .iter()
        .all(|sample| sample.path != "C:/images/two.png"));
}

#[test]
fn similarity_feature_cache_and_group_round_trips() {
    let mut database = Database::open_in_memory().expect("database should migrate");
    let feature = ImageSimilarityFeatureRecord {
        path: "C:/images/test1.png".to_string(),
        file_size: 10240,
        width: 800,
        height: 600,
        format: "png".to_string(),
        clarity_score: 125.5,
        phash: "1122334455667788".to_string(),
        dhash: "8877665544332211".to_string(),
        color_hist_json: "[0.1, 0.2]".to_string(),
        frames_json: Some("[{\"frame_index\":0}]".to_string()),
        frame_count: 2,
        feature_version: SIMILARITY_FEATURE_VERSION,
    };
    database
        .save_similarity_features(&[feature.clone()])
        .expect("features should save");

    let cached = database
        .get_similarity_features_by_paths(&["C:/images/test1.png".to_string()])
        .expect("should query cached");
    assert_eq!(cached.len(), 1);
    assert_eq!(cached[0].phash, "1122334455667788");
    assert_eq!(cached[0].frame_count, 2);
    assert_eq!(cached[0].feature_version, SIMILARITY_FEATURE_VERSION);
    assert!(cached[0].frames_json.is_some());

    let group = SimilarityGroup {
        group_id: "group-001".to_string(),
        group_type: "exact_duplicate".to_string(),
        average_similarity: 0.98,
        items: vec![
            SimilarityGroupItem {
                path: "C:/images/test1.png".to_string(),
                file_size: 10240,
                dimensions: [800, 600],
                format: "png".to_string(),
                clarity_score: 125.5,
                is_recommended: true,
                recommend_reason: Some("最佳画质".to_string()),
                decision: Some("keep".to_string()),
            },
            SimilarityGroupItem {
                path: "C:/images/test2.png".to_string(),
                file_size: 5120,
                dimensions: [400, 300],
                format: "jpg".to_string(),
                clarity_score: 60.0,
                is_recommended: false,
                recommend_reason: Some("低清版本".to_string()),
                decision: Some("archive".to_string()),
            },
        ],
    };

    database
        .save_similarity_groups(&[group.clone()])
        .expect("groups should save");

    let loaded = database
        .get_latest_similarity_groups()
        .expect("should load groups");
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].group_id, "group-001");
    assert_eq!(loaded[0].items.len(), 2);
    assert!(loaded[0].items[0].is_recommended);
}
