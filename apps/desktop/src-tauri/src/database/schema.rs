use rusqlite::{params, Connection};

pub fn run_migrations(connection: &Connection) -> Result<(), rusqlite::Error> {
    connection.execute_batch(
        "CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );",
    )?;
    let current: i64 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if current < 1 {
        connection.execute_batch(
            "BEGIN;
            CREATE TABLE IF NOT EXISTS libraries (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                root_path TEXT NOT NULL UNIQUE,
                output_path TEXT,
                operation_mode TEXT NOT NULL DEFAULT 'copy',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_scanned_at TEXT
            );
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                library_id TEXT NOT NULL REFERENCES libraries(id) ON DELETE CASCADE,
                original_path TEXT NOT NULL,
                current_path TEXT NOT NULL,
                relative_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                extension TEXT NOT NULL,
                file_size INTEGER NOT NULL DEFAULT 0,
                width INTEGER,
                height INTEGER,
                modified_at TEXT,
                sha256 TEXT,
                quick_fingerprint TEXT,
                phash TEXT,
                scan_status TEXT NOT NULL DEFAULT 'unscanned',
                content_version TEXT,
                error_code TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_images_library_status ON images(library_id, scan_status);
            CREATE INDEX IF NOT EXISTS idx_images_content ON images(file_size, modified_at, quick_fingerprint);
            CREATE TABLE IF NOT EXISTS detections (
                id TEXT PRIMARY KEY,
                image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                person_index INTEGER NOT NULL,
                bbox_x REAL NOT NULL,
                bbox_y REAL NOT NULL,
                bbox_w REAL NOT NULL,
                bbox_h REAL NOT NULL,
                detect_confidence REAL NOT NULL,
                pose_hint TEXT,
                quality_score REAL,
                final_character_tag TEXT,
                final_confidence REAL,
                review_status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_detections_image ON detections(image_id, person_index);
            CREATE TABLE IF NOT EXISTS recognitions (
                id TEXT PRIMARY KEY,
                detection_id TEXT NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
                crop_type TEXT NOT NULL,
                model_id TEXT NOT NULL,
                character_tag TEXT NOT NULL,
                score REAL NOT NULL,
                rank INTEGER NOT NULL,
                is_in_active_character_set INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_recognitions_detection ON recognitions(detection_id, crop_type, rank);
            CREATE TABLE IF NOT EXISTS classification_plans (
                id TEXT PRIMARY KEY,
                image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                source_path TEXT NOT NULL,
                target_path TEXT NOT NULL,
                operation_type TEXT NOT NULL,
                planned_file_name TEXT NOT NULL,
                plan_status TEXT NOT NULL DEFAULT 'draft',
                warning_code TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                approved_at TEXT,
                executed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS file_operations (
                id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL REFERENCES classification_plans(id) ON DELETE CASCADE,
                source_path TEXT NOT NULL,
                target_path TEXT NOT NULL,
                operation TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'planned',
                started_at TEXT,
                completed_at TEXT,
                undo_status TEXT,
                undone_at TEXT,
                error_code TEXT,
                error_message TEXT
            );
            CREATE TABLE IF NOT EXISTS similarity_embeddings (
                id TEXT PRIMARY KEY,
                image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
                model_id TEXT NOT NULL,
                model_version TEXT NOT NULL,
                vector_path TEXT,
                vector_dimension INTEGER,
                content_version TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(image_id, model_id, model_version, content_version)
            );
            CREATE TABLE IF NOT EXISTS similarity_groups (
                id TEXT PRIMARY KEY,
                algorithm_version TEXT NOT NULL,
                group_status TEXT NOT NULL DEFAULT 'pending',
                decision TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                decided_at TEXT
            );
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                job_type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                total_count INTEGER NOT NULL DEFAULT 0,
                completed_count INTEGER NOT NULL DEFAULT 0,
                success_count INTEGER NOT NULL DEFAULT 0,
                failed_count INTEGER NOT NULL DEFAULT 0,
                skipped_count INTEGER NOT NULL DEFAULT 0,
                current_stage TEXT,
                cursor TEXT,
                started_at TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                error_code TEXT,
                error_message TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, updated_at);
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value_json TEXT NOT NULL,
                value_version INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO schema_migrations(version) VALUES (1);
            PRAGMA user_version = 1;
            COMMIT;",
        )?;
    }
    let current: i64 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if current < 2 {
        connection.execute_batch(
            "BEGIN;
            CREATE TABLE IF NOT EXISTS character_identities (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                normalized_name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                deleted_at TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_character_identities_name
                ON character_identities(normalized_name) WHERE deleted_at IS NULL;
            CREATE TABLE IF NOT EXISTS annotation_images (
                path TEXT PRIMARY KEY,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                current_revision INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS annotation_revisions (
                id TEXT PRIMARY KEY,
                image_path TEXT NOT NULL REFERENCES annotation_images(path) ON DELETE CASCADE,
                revision INTEGER NOT NULL,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(image_path, revision)
            );
            CREATE INDEX IF NOT EXISTS idx_annotation_revisions_image
                ON annotation_revisions(image_path, revision DESC);
            CREATE TABLE IF NOT EXISTS annotation_revision_items (
                revision_id TEXT NOT NULL REFERENCES annotation_revisions(id) ON DELETE CASCADE,
                annotation_id TEXT NOT NULL,
                identity_id TEXT NOT NULL REFERENCES character_identities(id),
                label_name TEXT NOT NULL,
                bbox_x REAL NOT NULL,
                bbox_y REAL NOT NULL,
                bbox_width REAL NOT NULL,
                bbox_height REAL NOT NULL,
                source TEXT NOT NULL,
                manually_adjusted INTEGER NOT NULL DEFAULT 0,
                deleted INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(revision_id, annotation_id)
            );
            CREATE TABLE IF NOT EXISTS image_annotations_current (
                image_path TEXT NOT NULL REFERENCES annotation_images(path) ON DELETE CASCADE,
                annotation_id TEXT NOT NULL,
                identity_id TEXT NOT NULL REFERENCES character_identities(id),
                label_name TEXT NOT NULL,
                bbox_x REAL NOT NULL,
                bbox_y REAL NOT NULL,
                bbox_width REAL NOT NULL,
                bbox_height REAL NOT NULL,
                source TEXT NOT NULL,
                manually_adjusted INTEGER NOT NULL DEFAULT 0,
                deleted INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(image_path, annotation_id)
            );
            CREATE INDEX IF NOT EXISTS idx_image_annotations_current_visible
                ON image_annotations_current(image_path, deleted);
            CREATE TABLE IF NOT EXISTS personal_embeddings (
                id TEXT PRIMARY KEY,
                image_path TEXT NOT NULL REFERENCES annotation_images(path) ON DELETE CASCADE,
                revision INTEGER NOT NULL,
                annotation_id TEXT NOT NULL,
                identity_id TEXT NOT NULL REFERENCES character_identities(id),
                embedding_json TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                manually_verified INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(image_path, revision, annotation_id)
            );
            CREATE INDEX IF NOT EXISTS idx_personal_embeddings_identity
                ON personal_embeddings(identity_id, created_at);
            CREATE TABLE IF NOT EXISTS personal_model_versions (
                id TEXT PRIMARY KEY,
                version INTEGER NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'active',
                sample_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                activated_at TEXT,
                metadata_json TEXT
            );
            INSERT OR IGNORE INTO personal_model_versions(id, version, status, sample_count)
                VALUES ('builtin-v1', 1, 'active', 0);
            INSERT INTO schema_migrations(version) VALUES (2);
            PRAGMA user_version = 2;
            COMMIT;",
        )?;
    }
    let current: i64 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if current < 3 {
        connection.execute_batch("BEGIN;")?;
        for (column, definition) in [
            ("artifact_json", "TEXT"),
            ("metrics_json", "TEXT NOT NULL DEFAULT '{}'"),
            ("algorithm", "TEXT NOT NULL DEFAULT 'prototype_centroid_v1'"),
            ("warnings_json", "TEXT NOT NULL DEFAULT '[]'"),
            ("class_count", "INTEGER NOT NULL DEFAULT 0"),
            ("eligible_for_activation", "INTEGER NOT NULL DEFAULT 0"),
            ("error_code", "TEXT"),
            ("error_message", "TEXT"),
        ] {
            let exists: bool = connection.query_row(
                "SELECT EXISTS(
                   SELECT 1 FROM pragma_table_info('personal_model_versions')
                   WHERE name = ?1
                 )",
                params![column],
                |row| row.get(0),
            )?;
            if !exists {
                connection.execute_batch(&format!(
                    "ALTER TABLE personal_model_versions ADD COLUMN {column} {definition};"
                ))?;
            }
        }
        connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS personal_training_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                status TEXT NOT NULL DEFAULT 'idle',
                message TEXT NOT NULL DEFAULT '',
                current_job_id TEXT,
                last_version INTEGER,
                error_code TEXT,
                error_message TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                started_at TEXT,
                completed_at TEXT
            );
            INSERT OR IGNORE INTO personal_training_state(id, status, message)
                VALUES (1, 'idle', '尚未训练个人模型。');
            CREATE TABLE IF NOT EXISTS personal_training_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                min_total_samples INTEGER NOT NULL DEFAULT 3,
                min_samples_per_class INTEGER NOT NULL DEFAULT 3,
                auto_activate INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT OR IGNORE INTO personal_training_settings(
                id, min_total_samples, min_samples_per_class, auto_activate)
                VALUES (1, 3, 3, 1);
            CREATE INDEX IF NOT EXISTS idx_personal_model_versions_status
                ON personal_model_versions(status, version DESC);
            CREATE INDEX IF NOT EXISTS idx_personal_embeddings_current_verified
                ON personal_embeddings(image_path, revision, manually_verified);
            INSERT OR IGNORE INTO schema_migrations(version) VALUES (3);
            PRAGMA user_version = 3;
            COMMIT;",
        )?;
    }
    let current: i64 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if current < 4 {
        connection.execute_batch(
            "BEGIN;
            CREATE TABLE IF NOT EXISTS image_similarity_features (
                path TEXT PRIMARY KEY,
                file_size INTEGER NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                format TEXT NOT NULL,
                clarity_score REAL NOT NULL,
                phash TEXT NOT NULL,
                dhash TEXT NOT NULL,
                color_hist_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE INDEX IF NOT EXISTS idx_similarity_features_updated
                ON image_similarity_features(updated_at DESC);
            CREATE TABLE IF NOT EXISTS similarity_group_records (
                id TEXT PRIMARY KEY,
                group_type TEXT NOT NULL,
                average_similarity REAL NOT NULL,
                items_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT OR IGNORE INTO schema_migrations(version) VALUES (4);
            PRAGMA user_version = 4;
            COMMIT;",
        )?;
    }
    let current: i64 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if current < 5 {
        connection.execute_batch(
            "BEGIN;
            ALTER TABLE image_similarity_features ADD COLUMN frames_json TEXT;
            ALTER TABLE image_similarity_features ADD COLUMN frame_count INTEGER NOT NULL DEFAULT 1;
            ALTER TABLE image_similarity_features ADD COLUMN feature_version INTEGER NOT NULL DEFAULT 1;
            INSERT OR IGNORE INTO schema_migrations(version) VALUES (5);
            PRAGMA user_version = 5;
            COMMIT;",
        )?;
    }
    Ok(())
}
