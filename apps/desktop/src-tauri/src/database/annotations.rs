use crate::database::models::{
    AnnotationBbox, AnnotationRecord, CharacterIdentity, Database, DatabaseError, EmbeddingSample,
    IdentitySample, IdentitySampleSummary, ImageAnnotations,
};
use rusqlite::{params, OptionalExtension};
use std::collections::HashSet;

impl Database {
    pub fn list_image_annotations(&self, path: &str) -> Result<ImageAnnotations, rusqlite::Error> {
        let image = self
            .connection
            .query_row(
                "SELECT image_width, image_height, current_revision
                 FROM annotation_images WHERE path = ?1",
                params![path],
                |row| {
                    Ok((
                        row.get::<_, u32>(0)?,
                        row.get::<_, u32>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                },
            )
            .optional()?;
        let Some((width, height, revision)) = image else {
            return Ok(ImageAnnotations {
                path: path.to_string(),
                image_size: [0, 0],
                revision: 0,
                annotations: Vec::new(),
                learned_samples: 0,
                training_recommended: false,
                verified_sample_count: 0,
                verified_class_count: 0,
            });
        };
        let mut statement = self.connection.prepare(
            "SELECT annotation_id, identity_id, label_name, bbox_x, bbox_y,
                    bbox_width, bbox_height, source, manually_adjusted
             FROM image_annotations_current
             WHERE image_path = ?1 AND deleted = 0
             ORDER BY rowid",
        )?;
        let annotations = statement
            .query_map(params![path], |row| {
                Ok(AnnotationRecord {
                    id: row.get(0)?,
                    identity_id: row.get(1)?,
                    label_name: row.get(2)?,
                    bbox: AnnotationBbox {
                        x: row.get(3)?,
                        y: row.get(4)?,
                        width: row.get(5)?,
                        height: row.get(6)?,
                    },
                    source: row.get(7)?,
                    manually_adjusted: row.get::<_, i64>(8)? != 0,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        let readiness = self.training_readiness()?;
        let learned_samples = self.connection.query_row(
            "SELECT COUNT(*) FROM personal_embeddings e
             JOIN annotation_images ai ON ai.path = e.image_path AND ai.current_revision = e.revision
             JOIN image_annotations_current c
               ON c.image_path = e.image_path AND c.annotation_id = e.annotation_id
              AND c.identity_id = e.identity_id
             WHERE e.image_path = ?1 AND c.deleted = 0 AND e.manually_verified = 1",
            params![path],
            |row| row.get::<_, i64>(0),
        )?;
        Ok(ImageAnnotations {
            path: path.to_string(),
            image_size: [width, height],
            revision,
            annotations,
            learned_samples: learned_samples.max(0) as usize,
            training_recommended: readiness.ready,
            verified_sample_count: readiness.verified_sample_count,
            verified_class_count: readiness.verified_class_count,
        })
    }

    pub fn list_annotated_image_paths(
        &self,
        directory: &str,
    ) -> Result<HashSet<String>, rusqlite::Error> {
        let normalized_dir = directory.replace('\\', "/");
        let prefix = if normalized_dir.ends_with('/') {
            normalized_dir.clone()
        } else {
            format!("{normalized_dir}/")
        };
        let mut stmt = self.connection.prepare(
            "SELECT DISTINCT image_path FROM image_annotations_current WHERE deleted = 0",
        )?;
        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
        let mut paths = HashSet::new();
        for path in rows {
            let p = path?;
            let normalized_p = p.replace('\\', "/");
            if normalized_p.starts_with(&prefix) || normalized_p == normalized_dir {
                paths.insert(normalized_p);
            }
        }
        Ok(paths)
    }

    pub fn get_annotation_as_scan_result(
        &self,
        path: &str,
    ) -> Result<Option<serde_json::Value>, rusqlite::Error> {
        let annos = self.list_image_annotations(path)?;
        if annos.annotations.is_empty() {
            return Ok(None);
        }
        let width = annos.image_size[0] as f64;
        let height = annos.image_size[1] as f64;
        let people: Vec<serde_json::Value> = annos
            .annotations
            .iter()
            .enumerate()
            .map(|(idx, a)| {
                let x1 = a.bbox.x * width;
                let y1 = a.bbox.y * height;
                let x2 = (a.bbox.x + a.bbox.width) * width;
                let y2 = (a.bbox.y + a.bbox.height) * height;
                serde_json::json!({
                    "person_index": idx,
                    "box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "confidence": 1.0,
                    },
                    "crops": {},
                    "status": "high_confidence",
                    "top1": {
                        "character_tag": a.label_name,
                        "display_name": a.label_name,
                        "confidence": 1.0,
                        "source": "manual",
                    },
                    "top2": null,
                    "margin": 1.0,
                    "consistency": 1.0,
                    "candidates": [{
                        "character_tag": a.label_name,
                        "display_name": a.label_name,
                        "confidence": 1.0,
                        "source": "manual",
                    }]
                })
            })
            .collect();

        Ok(Some(serde_json::json!({
            "path": path,
            "image_size": annos.image_size,
            "people": people,
        })))
    }

    pub fn update_image_path(
        &mut self,
        old_path: &str,
        new_path: &str,
    ) -> Result<(), rusqlite::Error> {
        let transaction = self.connection.transaction()?;
        let exists: bool = transaction.query_row(
            "SELECT COUNT(1) > 0 FROM annotation_images WHERE path = ?1",
            params![old_path],
            |row| row.get(0),
        )?;
        if !exists {
            return Ok(());
        }

        transaction.execute(
            "INSERT INTO annotation_images (path, image_width, image_height, current_revision, created_at, updated_at)
             SELECT ?1, image_width, image_height, current_revision, created_at, CURRENT_TIMESTAMP
             FROM annotation_images WHERE path = ?2",
            params![new_path, old_path],
        )?;
        transaction.execute(
            "UPDATE annotation_revisions SET image_path = ?1 WHERE image_path = ?2",
            params![new_path, old_path],
        )?;
        transaction.execute(
            "UPDATE image_annotations_current SET image_path = ?1 WHERE image_path = ?2",
            params![new_path, old_path],
        )?;
        transaction.execute(
            "UPDATE personal_embeddings SET image_path = ?1 WHERE image_path = ?2",
            params![new_path, old_path],
        )?;
        transaction.execute(
            "DELETE FROM annotation_images WHERE path = ?1",
            params![old_path],
        )?;
        transaction.commit()?;
        Ok(())
    }

    pub fn delete_image_annotations_for_path(
        &self,
        image_path: &str,
    ) -> Result<(), rusqlite::Error> {
        self.connection.execute(
            "UPDATE image_annotations_current SET deleted = 1, updated_at = CURRENT_TIMESTAMP WHERE image_path = ?1",
            params![image_path],
        )?;
        self.connection.execute(
            "DELETE FROM annotation_images WHERE path = ?1",
            params![image_path],
        )?;
        Ok(())
    }

    pub fn list_character_identities(&self) -> Result<Vec<CharacterIdentity>, rusqlite::Error> {
        let mut statement = self.connection.prepare(
            "SELECT id, display_name, created_at, updated_at
             FROM character_identities WHERE deleted_at IS NULL
             ORDER BY normalized_name, id",
        )?;
        let result = statement
            .query_map([], |row| {
                Ok(CharacterIdentity {
                    id: row.get(0)?,
                    display_name: row.get(1)?,
                    created_at: row.get(2)?,
                    updated_at: row.get(3)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>();
        result
    }

    pub fn identity_id_by_name(
        &self,
        normalized_name: &str,
    ) -> Result<Option<String>, rusqlite::Error> {
        self.connection
            .query_row(
                "SELECT id FROM character_identities
                 WHERE normalized_name = ?1 AND deleted_at IS NULL
                 ORDER BY id LIMIT 1",
                params![normalized_name],
                |row| row.get(0),
            )
            .optional()
    }

    pub(crate) fn save_image_annotations(
        &mut self,
        path: &str,
        image_size: [u32; 2],
        annotations: &[AnnotationRecord],
        embeddings: &[EmbeddingSample],
        base_revision: Option<i64>,
    ) -> Result<ImageAnnotations, DatabaseError> {
        let current_revision = self
            .connection
            .query_row(
                "SELECT current_revision FROM annotation_images WHERE path = ?1",
                params![path],
                |row| row.get::<_, i64>(0),
            )
            .optional()?
            .unwrap_or(0);
        if let Some(base_revision) = base_revision {
            if base_revision < 0 || base_revision != current_revision {
                return Err(DatabaseError::RevisionConflict { current_revision });
            }
        }
        let new_revision = current_revision + 1;
        let revision_id = uuid::Uuid::new_v4().to_string();
        let transaction = self.connection.transaction()?;
        transaction.execute(
            "INSERT INTO annotation_images(path, image_width, image_height, current_revision)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(path) DO UPDATE SET image_width = excluded.image_width,
               image_height = excluded.image_height,
               current_revision = excluded.current_revision,
               updated_at = CURRENT_TIMESTAMP",
            params![path, image_size[0], image_size[1], new_revision],
        )?;
        transaction.execute(
            "INSERT INTO annotation_revisions(id, image_path, revision, image_width, image_height)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                revision_id,
                path,
                new_revision,
                image_size[0],
                image_size[1]
            ],
        )?;

        let mut old_statement = transaction.prepare(
            "SELECT annotation_id, identity_id, label_name, bbox_x, bbox_y,
                    bbox_width, bbox_height, source, manually_adjusted
             FROM image_annotations_current WHERE image_path = ?1 AND deleted = 0",
        )?;
        let old = old_statement
            .query_map(params![path], |row| {
                Ok(AnnotationRecord {
                    id: row.get(0)?,
                    identity_id: row.get(1)?,
                    label_name: row.get(2)?,
                    bbox: AnnotationBbox {
                        x: row.get(3)?,
                        y: row.get(4)?,
                        width: row.get(5)?,
                        height: row.get(6)?,
                    },
                    source: row.get(7)?,
                    manually_adjusted: row.get::<_, i64>(8)? != 0,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        drop(old_statement);
        let visible_ids: std::collections::HashSet<&str> = annotations
            .iter()
            .map(|annotation| annotation.id.as_str())
            .collect();

        for annotation in annotations {
            transaction.execute(
                "INSERT INTO character_identities(id, display_name, normalized_name)
                 VALUES (?1, ?2, ?3)
                 ON CONFLICT(id) DO UPDATE SET
                   display_name = CASE WHEN ?4 = 1 THEN excluded.display_name
                                       ELSE character_identities.display_name END,
                   normalized_name = CASE WHEN ?4 = 1 THEN excluded.normalized_name
                                          ELSE character_identities.normalized_name END,
                   deleted_at = NULL, updated_at = CURRENT_TIMESTAMP",
                params![
                    annotation.identity_id,
                    annotation.label_name,
                    annotation.label_name.to_lowercase(),
                    (annotation.source == "manual" || annotation.manually_adjusted) as i64,
                ],
            )?;
            transaction.execute(
                "INSERT INTO annotation_revision_items(
                   revision_id, annotation_id, identity_id, label_name,
                   bbox_x, bbox_y, bbox_width, bbox_height, source, manually_adjusted, deleted)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 0)",
                params![
                    revision_id,
                    annotation.id,
                    annotation.identity_id,
                    annotation.label_name,
                    annotation.bbox.x,
                    annotation.bbox.y,
                    annotation.bbox.width,
                    annotation.bbox.height,
                    annotation.source,
                    annotation.manually_adjusted as i64,
                ],
            )?;
            transaction.execute(
                "INSERT INTO image_annotations_current(
                   image_path, annotation_id, identity_id, label_name,
                   bbox_x, bbox_y, bbox_width, bbox_height, source, manually_adjusted, deleted)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 0)
                 ON CONFLICT(image_path, annotation_id) DO UPDATE SET
                   identity_id = excluded.identity_id, label_name = excluded.label_name,
                   bbox_x = excluded.bbox_x, bbox_y = excluded.bbox_y,
                   bbox_width = excluded.bbox_width, bbox_height = excluded.bbox_height,
                   source = excluded.source, manually_adjusted = excluded.manually_adjusted,
                   deleted = 0, updated_at = CURRENT_TIMESTAMP",
                params![
                    path,
                    annotation.id,
                    annotation.identity_id,
                    annotation.label_name,
                    annotation.bbox.x,
                    annotation.bbox.y,
                    annotation.bbox.width,
                    annotation.bbox.height,
                    annotation.source,
                    annotation.manually_adjusted as i64,
                ],
            )?;
        }
        for annotation in old
            .iter()
            .filter(|annotation| !visible_ids.contains(annotation.id.as_str()))
        {
            transaction.execute(
                "INSERT INTO annotation_revision_items(
                   revision_id, annotation_id, identity_id, label_name,
                   bbox_x, bbox_y, bbox_width, bbox_height, source, manually_adjusted, deleted)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 1)",
                params![
                    revision_id,
                    annotation.id,
                    annotation.identity_id,
                    annotation.label_name,
                    annotation.bbox.x,
                    annotation.bbox.y,
                    annotation.bbox.width,
                    annotation.bbox.height,
                    annotation.source,
                    annotation.manually_adjusted as i64,
                ],
            )?;
            transaction.execute(
                "UPDATE image_annotations_current SET deleted = 1, updated_at = CURRENT_TIMESTAMP
                 WHERE image_path = ?1 AND annotation_id = ?2",
                params![path, annotation.id],
            )?;
        }
        for sample in embeddings {
            let serialized = serde_json::to_string(&sample.embedding)?;
            transaction.execute(
                "INSERT INTO personal_embeddings(
                   id, image_path, revision, annotation_id, identity_id,
                   embedding_json, dimension, manually_verified)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    uuid::Uuid::new_v4().to_string(),
                    path,
                    new_revision,
                    sample.annotation_id,
                    sample.identity_id,
                    serialized,
                    sample.embedding.len() as i64,
                    sample.manually_verified as i64,
                ],
            )?;
        }
        transaction.execute(
            "UPDATE personal_model_versions SET sample_count = (
                SELECT COUNT(*) FROM personal_embeddings e
                 JOIN annotation_images ai
                   ON ai.path = e.image_path AND ai.current_revision = e.revision
                 JOIN image_annotations_current c
                   ON c.image_path = e.image_path AND c.annotation_id = e.annotation_id
                  AND c.identity_id = e.identity_id AND c.deleted = 0
                 WHERE e.dimension = 2048 AND e.manually_verified = 1),
             class_count = (
                SELECT COUNT(DISTINCT e.identity_id) FROM personal_embeddings e
                 JOIN annotation_images ai
                   ON ai.path = e.image_path AND ai.current_revision = e.revision
                 JOIN image_annotations_current c
                   ON c.image_path = e.image_path AND c.annotation_id = e.annotation_id
                  AND c.identity_id = e.identity_id AND c.deleted = 0
                 WHERE e.dimension = 2048 AND e.manually_verified = 1)
              WHERE version = 1",
            [],
        )?;
        transaction.commit()?;
        Ok(self.list_image_annotations(path)?)
    }

    /// Characters that have annotated images, with how many are usable as
    /// LoRA training material (unique images; manual corrections counted).
    pub fn list_identity_sample_summaries(
        &self,
    ) -> Result<Vec<IdentitySampleSummary>, rusqlite::Error> {
        let mut statement = self.connection.prepare(
            "SELECT c.identity_id,
                    COALESCE(i.display_name, c.label_name),
                    c.label_name,
                    COUNT(DISTINCT c.image_path),
                    COUNT(DISTINCT CASE WHEN c.source = 'manual' OR c.manually_adjusted = 1
                                        THEN c.image_path END)
             FROM image_annotations_current c
             LEFT JOIN character_identities i ON i.id = c.identity_id
             WHERE c.deleted = 0
             GROUP BY c.identity_id, c.label_name
             ORDER BY COUNT(DISTINCT c.image_path) DESC, c.label_name",
        )?;
        let rows = statement.query_map([], |row| {
            Ok(IdentitySampleSummary {
                identity_id: row.get(0)?,
                display_name: row.get(1)?,
                label_name: row.get(2)?,
                image_count: row.get::<_, i64>(3)? as usize,
                manual_count: row.get::<_, i64>(4)? as usize,
            })
        })?;
        rows.collect()
    }

    /// Every current annotation box of one character, for dataset export.
    pub fn list_identity_samples(
        &self,
        identity_id: &str,
    ) -> Result<Vec<IdentitySample>, rusqlite::Error> {
        let mut statement = self.connection.prepare(
            "SELECT c.image_path, c.label_name, ai.image_width, ai.image_height,
                    c.bbox_x, c.bbox_y, c.bbox_width, c.bbox_height,
                    c.source, c.manually_adjusted
             FROM image_annotations_current c
             JOIN annotation_images ai ON ai.path = c.image_path
             WHERE c.deleted = 0 AND c.identity_id = ?1
             ORDER BY c.image_path",
        )?;
        let rows = statement.query_map(params![identity_id], |row| {
            Ok(IdentitySample {
                path: row.get(0)?,
                label_name: row.get(1)?,
                image_width: row.get::<_, i64>(2)? as u32,
                image_height: row.get::<_, i64>(3)? as u32,
                bbox_x: row.get(4)?,
                bbox_y: row.get(5)?,
                bbox_width: row.get(6)?,
                bbox_height: row.get(7)?,
                source: row.get(8)?,
                manually_adjusted: row.get(9)?,
            })
        })?;
        rows.collect()
    }
}
