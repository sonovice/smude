//! Record creation on the feedback PocketBase, over NSH's platform HTTP stack.
//! The backend is reproducible from `feedback-backend/`.
use nsh_core::prelude::*;
use serde_json::Value;

const COLLECTIONS: &str = "https://smude.notavium.de/feedback/api/collections";
/// Keeps drive-by posts out; the collections' create rule checks it. Not a
/// secret: anyone with the binary has it, and nothing can be read with it.
const TOKEN: &str = "c415095c1254bc2ebea4fbff1e9629dd";

#[derive(Clone, Debug, PartialEq)]
pub enum Field {
    Text(String),
    Json(Value),
    File {
        name: String,
        mime: &'static str,
        bytes: Vec<u8>,
    },
}

/// `multipart/form-data`, as PocketBase parses it: JSON fields as text parts.
pub fn multipart(fields: &[(&str, Field)], boundary: &str) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, field) in fields {
        body.extend_from_slice(format!("--{boundary}\r\n").as_bytes());
        match field {
            Field::Text(text) => part(&mut body, name, None, text.as_bytes()),
            Field::Json(value) => part(&mut body, name, None, value.to_string().as_bytes()),
            Field::File { name: file, mime, bytes } => {
                part(&mut body, name, Some((file, mime)), bytes)
            }
        }
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    body
}
fn part(body: &mut Vec<u8>, name: &str, file: Option<(&str, &str)>, bytes: &[u8]) {
    let name = name.replace('"', "");
    match file {
        Some((file, mime)) => body.extend_from_slice(
            format!(
                "Content-Disposition: form-data; name=\"{name}\"; filename=\"{}\"\r\nContent-Type: {mime}\r\n\r\n",
                file.replace('"', "")
            )
            .as_bytes(),
        ),
        None => body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes(),
        ),
    }
    body.extend_from_slice(bytes);
    body.extend_from_slice(b"\r\n");
}

pub async fn create_multipart(collection: &str, fields: &[(&str, Field)]) -> Result<(), String> {
    let boundary = format!("smude-{:x}", crate::model::id());
    let request = HttpRequest::post(records(collection), multipart(fields, &boundary)).header(
        "Content-Type",
        format!("multipart/form-data; boundary={boundary}"),
    );
    send(request).await
}

pub async fn create_json(collection: &str, record: &Value) -> Result<(), String> {
    let request = HttpRequest::post(records(collection), record.to_string())
        .header("Content-Type", "application/json");
    send(request).await
}

fn records(collection: &str) -> String {
    format!("{COLLECTIONS}/{collection}/records")
}

async fn send(request: HttpRequest) -> Result<(), String> {
    let response = fetch(request.header("X-Smude-Token", TOKEN)).await?;
    if response.ok() {
        return Ok(());
    }
    let mut detail = response.text();
    detail.truncate(200);
    Err(format!("HTTP {}: {detail}", response.status()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multipart_keeps_binary_files_intact_and_closes_the_body() {
        let body = multipart(
            &[
                ("install", Field::Text("abc".into())),
                ("page", Field::Json(serde_json::json!({"id": 1}))),
                (
                    "original",
                    Field::File {
                        name: "1.jpg".into(),
                        mime: "image/jpeg",
                        bytes: vec![0xff, 0xd8, 0x00, 0xff],
                    },
                ),
            ],
            "b",
        );
        let text = String::from_utf8_lossy(&body);
        assert!(text.starts_with("--b\r\nContent-Disposition: form-data; name=\"install\"\r\n\r\nabc\r\n"));
        assert!(text.contains("name=\"page\"\r\n\r\n{\"id\":1}\r\n"));
        assert!(text.contains("name=\"original\"; filename=\"1.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n"));
        assert!(body.windows(4).any(|w| w == [0xff, 0xd8, 0x00, 0xff]));
        assert!(text.ends_with("\r\n--b--\r\n"));
    }
}
