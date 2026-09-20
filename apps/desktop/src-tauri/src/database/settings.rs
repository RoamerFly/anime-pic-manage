use crate::database::models::{Database, DatabaseError};
use rusqlite::{params, OptionalExtension};

impl Database {
    pub fn get_setting_string(&self, key: &str) -> Result<Option<String>, DatabaseError> {
        let value_json: Option<String> = self
            .connection
            .query_row(
                "SELECT value_json FROM settings WHERE key = ?1",
                params![key],
                |row| row.get(0),
            )
            .optional()?;
        value_json
            .map(|value| serde_json::from_str(&value).map_err(DatabaseError::from))
            .transpose()
    }

    pub fn set_setting_string(&self, key: &str, value: &str) -> Result<(), DatabaseError> {
        let value_json = serde_json::to_string(value)?;
        self.connection.execute(
            "INSERT INTO settings(key, value_json, value_version, updated_at)
             VALUES (?1, ?2, 1, CURRENT_TIMESTAMP)
             ON CONFLICT(key) DO UPDATE SET
                 value_json = excluded.value_json,
                 value_version = settings.value_version + 1,
                 updated_at = CURRENT_TIMESTAMP",
            params![key, value_json],
        )?;
        Ok(())
    }
}
