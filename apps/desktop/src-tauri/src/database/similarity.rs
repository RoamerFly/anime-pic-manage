use crate::database::models::{
    Database, DatabaseError, ImageSimilarityFeatureRecord, SimilarityGroup, SimilarityGroupItem,
};
use rusqlite::params;

impl Database {
    pub fn save_similarity_features(
        &mut self,
        features: &[ImageSimilarityFeatureRecord],
    ) -> Result<(), DatabaseError> {
        if features.is_empty() {
            return Ok(());
        }
        let transaction = self.connection.transaction()?;
        {
            let mut stmt = transaction.prepare(
                "INSERT INTO image_similarity_features (
                    path, file_size, width, height, format, clarity_score, phash, dhash,
                    color_hist_json, frames_json, frame_count, feature_version, updated_at
                ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, CURRENT_TIMESTAMP)
                ON CONFLICT(path) DO UPDATE SET
                    file_size = excluded.file_size,
                    width = excluded.width,
                    height = excluded.height,
                    format = excluded.format,
                    clarity_score = excluded.clarity_score,
                    phash = excluded.phash,
                    dhash = excluded.dhash,
                    color_hist_json = excluded.color_hist_json,
                    frames_json = excluded.frames_json,
                    frame_count = excluded.frame_count,
                    feature_version = excluded.feature_version,
                    updated_at = CURRENT_TIMESTAMP",
            )?;
            for f in features {
                stmt.execute(params![
                    f.path,
                    f.file_size,
                    f.width,
                    f.height,
                    f.format,
                    f.clarity_score,
                    f.phash,
                    f.dhash,
                    f.color_hist_json,
                    f.frames_json,
                    f.frame_count,
                    f.feature_version,
                ])?;
            }
        }
        transaction.commit()?;
        Ok(())
    }

    pub fn get_similarity_features_by_paths(
        &self,
        paths: &[String],
    ) -> Result<Vec<ImageSimilarityFeatureRecord>, DatabaseError> {
        if paths.is_empty() {
            return Ok(Vec::new());
        }
        let mut results = Vec::new();
        for chunk in paths.chunks(500) {
            let placeholders = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT path, file_size, width, height, format, clarity_score, phash, dhash,
                        color_hist_json, frames_json, frame_count, feature_version
                 FROM image_similarity_features
                 WHERE path IN ({placeholders})"
            );
            let mut stmt = self.connection.prepare(&sql)?;
            let params = rusqlite::params_from_iter(chunk.iter());
            let rows = stmt.query_map(params, |row| {
                Ok(ImageSimilarityFeatureRecord {
                    path: row.get(0)?,
                    file_size: row.get(1)?,
                    width: row.get(2)?,
                    height: row.get(3)?,
                    format: row.get(4)?,
                    clarity_score: row.get(5)?,
                    phash: row.get(6)?,
                    dhash: row.get(7)?,
                    color_hist_json: row.get(8)?,
                    frames_json: row.get(9)?,
                    frame_count: row.get(10)?,
                    feature_version: row.get(11)?,
                })
            })?;
            for item in rows {
                results.push(item?);
            }
        }
        Ok(results)
    }

    pub fn save_similarity_groups(
        &mut self,
        groups: &[SimilarityGroup],
    ) -> Result<(), DatabaseError> {
        let transaction = self.connection.transaction()?;
        transaction.execute("DELETE FROM similarity_group_records", [])?;
        {
            let mut stmt = transaction.prepare(
                "INSERT INTO similarity_group_records (
                    id, group_type, average_similarity, items_json, created_at
                ) VALUES (?1, ?2, ?3, ?4, CURRENT_TIMESTAMP)",
            )?;
            for g in groups {
                let items_json = serde_json::to_string(&g.items)?;
                stmt.execute(params![
                    g.group_id,
                    g.group_type,
                    g.average_similarity,
                    items_json,
                ])?;
            }
        }
        transaction.commit()?;
        Ok(())
    }

    pub fn get_latest_similarity_groups(&self) -> Result<Vec<SimilarityGroup>, DatabaseError> {
        let mut stmt = self.connection.prepare(
            "SELECT id, group_type, average_similarity, items_json
             FROM similarity_group_records
             ORDER BY rowid ASC",
        )?;
        let rows = stmt.query_map([], |row| {
            let id: String = row.get(0)?;
            let group_type: String = row.get(1)?;
            let average_similarity: f64 = row.get(2)?;
            let items_json: String = row.get(3)?;
            let items: Vec<SimilarityGroupItem> =
                serde_json::from_str(&items_json).unwrap_or_default();
            Ok(SimilarityGroup {
                group_id: id,
                group_type,
                average_similarity,
                items,
            })
        })?;

        let mut groups = Vec::new();
        for g in rows {
            groups.push(g?);
        }
        Ok(groups)
    }

    pub fn clear_similarity_cache(&mut self) -> Result<(), DatabaseError> {
        let transaction = self.connection.transaction()?;
        transaction.execute("DELETE FROM image_similarity_features", [])?;
        transaction.execute("DELETE FROM similarity_group_records", [])?;
        transaction.commit()?;
        Ok(())
    }
}
