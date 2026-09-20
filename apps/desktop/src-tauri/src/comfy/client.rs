//! Minimal HTTP client for a local ComfyUI server.

use super::types::{ComfyError, ComfyGeneratedImage};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct ComfyClient {
    base_url: String,
    http: reqwest::Client,
}

impl ComfyClient {
    pub fn new(port: u16) -> Result<Self, ComfyError> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|error| ComfyError::Request(error.to_string()))?;
        Ok(Self {
            base_url: format!("http://127.0.0.1:{port}"),
            http,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    async fn get_json(&self, path: &str) -> Result<Value, ComfyError> {
        let response = self
            .http
            .get(format!("{}{path}", self.base_url))
            .send()
            .await
            .map_err(|error| ComfyError::NotRunning(error.to_string()))?;
        Self::decode(response, path).await
    }

    async fn post_json(&self, path: &str, body: Value) -> Result<Value, ComfyError> {
        let response = self
            .http
            .post(format!("{}{path}", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(|error| ComfyError::NotRunning(error.to_string()))?;
        Self::decode(response, path).await
    }

    async fn decode(response: reqwest::Response, path: &str) -> Result<Value, ComfyError> {
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|error| ComfyError::Request(format!("{path}: {error}")))?;
        if !status.is_success() {
            return Err(ComfyError::Response(format!(
                "{path} -> HTTP {}: {}",
                status.as_u16(),
                summarize(&text)
            )));
        }
        if text.trim().is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text)
            .map_err(|error| ComfyError::Response(format!("{path}: {error}: {}", summarize(&text))))
    }

    pub async fn system_stats(&self) -> Result<Value, ComfyError> {
        self.get_json("/system_stats").await
    }

    pub async fn list_models(&self, folder: &str) -> Result<Vec<String>, ComfyError> {
        let value = self.get_json(&format!("/models/{folder}")).await?;
        Ok(value
            .as_array()
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default())
    }

    /// Node input options, e.g. the sampler or scheduler choices.
    pub async fn node_input_options(
        &self,
        node_class: &str,
        input_name: &str,
    ) -> Result<Vec<String>, ComfyError> {
        let value = self.get_json(&format!("/object_info/{node_class}")).await?;
        let options = value
            .get(node_class)
            .and_then(|node| node.get("input"))
            .and_then(|input| {
                input
                    .get("required")
                    .or_else(|| input.get("optional"))
                    .and_then(|group| group.get(input_name))
            })
            .and_then(|entry| entry.get(0))
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_string))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        Ok(options)
    }

    /// Queue a prompt and return its prompt id.
    pub async fn queue_prompt(&self, prompt: Value, client_id: &str) -> Result<String, ComfyError> {
        let response = self
            .post_json(
                "/prompt",
                json!({ "prompt": prompt, "client_id": client_id }),
            )
            .await?;
        if let Some(detail) = response.get("error") {
            let message = response
                .get("node_errors")
                .map(|errors| format!("{detail} {errors}"))
                .unwrap_or_else(|| detail.to_string());
            return Err(ComfyError::Response(summarize(&message)));
        }
        response
            .get("prompt_id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| ComfyError::Response("ComfyUI 未返回 prompt_id".to_string()))
    }

    pub async fn history(&self, prompt_id: &str) -> Result<Option<Value>, ComfyError> {
        let value = self.get_json(&format!("/history/{prompt_id}")).await?;
        Ok(value
            .get(prompt_id)
            .cloned()
            .filter(|entry| !entry.is_null()))
    }

    pub async fn queue(&self) -> Result<Value, ComfyError> {
        self.get_json("/queue").await
    }

    pub async fn interrupt(&self) -> Result<(), ComfyError> {
        self.post_json("/interrupt", json!({})).await.map(|_| ())
    }

    /// Download one generated image into `output_dir`.
    pub async fn download_image(
        &self,
        image: &ComfyGeneratedImage,
        output_dir: &Path,
        timeout_secs: u64,
    ) -> Result<PathBuf, ComfyError> {
        let response = self
            .http
            .get(format!("{}/view", self.base_url))
            .query(&[
                ("filename", image.filename.as_str()),
                ("subfolder", image.subfolder.as_str()),
                ("type", "output"),
            ])
            .timeout(std::time::Duration::from_secs(timeout_secs))
            .send()
            .await
            .map_err(|error| ComfyError::Request(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            return Err(ComfyError::Response(format!(
                "下载生成图失败：HTTP {}",
                status.as_u16()
            )));
        }
        let bytes = response
            .bytes()
            .await
            .map_err(|error| ComfyError::Request(error.to_string()))?;
        std::fs::create_dir_all(output_dir)
            .map_err(|error| ComfyError::Io(format!("{}: {error}", output_dir.display())))?;
        let target = output_dir.join(&image.filename);
        std::fs::write(&target, &bytes)
            .map_err(|error| ComfyError::Io(format!("{}: {error}", target.display())))?;
        Ok(target)
    }
}

fn summarize(text: &str) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() > 300 {
        collapsed.chars().take(300).collect::<String>() + "…"
    } else {
        collapsed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarize_collapses_and_truncates() {
        let text = "line one\n\nline two   with   spaces";
        assert_eq!(summarize(text), "line one line two with spaces");

        let long = "a".repeat(400);
        let summary = summarize(&long);
        assert_eq!(summary.chars().count(), 301);
        assert!(summary.ends_with('…'));
    }
}
