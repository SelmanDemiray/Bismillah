use axum::{
    extract::{Path, State, Multipart, Query, WebSocketUpgrade, ws::{WebSocket, Message}},
    http::{StatusCode, Method, header},
    response::{Json, sse::{Event, Sse}, IntoResponse, Response},
    routing::{get, post},
    Router,
    middleware,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    convert::Infallible,
};
use tokio::net::TcpListener;
use tokio_stream::{wrappers::UnboundedReceiverStream, StreamExt};
use futures::stream::Stream;
use uuid::Uuid;
use tower_http::cors::{CorsLayer, Any};
use tower_http::trace::TraceLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};
use sqlx::{SqlitePool, Row};
use std::str::FromStr;
use chrono::{DateTime, Utc};
use jsonwebtoken::{encode, decode, Header, Algorithm, Validation, EncodingKey, DecodingKey};
use bcrypt::{hash, verify, DEFAULT_COST};

// ============================================================================
// API LAYER
// ============================================================================

mod api {
    use super::nn::{
        self,
        models::{GraphModel, AiModel},
        loss::{create_loss, LossType},
        optimizers::{create_optimizer, OptimizerConfig},
    };
    use super::*;

    #[derive(Clone)]
    pub struct AppState {
        pub models: Arc<Mutex<HashMap<String, Box<dyn AiModel>>>>,
        pub db: SqlitePool,
        pub tx: tokio::sync::broadcast::Sender<chat::ChatMessage>,
    }

    #[derive(Deserialize, Debug)]
    pub struct CreateModelRequest {
        pub graph: Value,
    }

    #[derive(Deserialize, Debug)]
    pub struct TrainRequest {
        pub x_train: Value,
        pub y_train: Value,
        pub optimizer: OptimizerConfig,
        pub loss: LossType,
        pub epochs: usize,
        pub batch_size: usize,
        pub validation_split: Option<f64>,
    }

    #[derive(Deserialize)]
    pub struct PredictRequest {
        pub x: Value,
    }

    #[derive(Serialize)]
    pub struct ApiResponse {
        pub status: String,
        pub message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub data: Option<Value>,
    }

    #[derive(Serialize, Deserialize)]
    pub struct SerializableTensor {
        shape: Vec<usize>,
        data: Vec<f64>,
    }

    #[derive(Serialize, Deserialize)]
    pub struct ExportedModel {
        architecture: Value,
        parameters: HashMap<String, SerializableTensor>,
    }


    pub async fn run() {
        tracing_subscriber::registry()
            .with(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| "backend=debug,tower_http=debug,axum=debug".into()),
            )
            .with(tracing_subscriber::fmt::layer())
            .init();

        // Initialize database
        let database_url = std::env::var("DATABASE_URL").unwrap_or_else(|_| "sqlite:/usr/src/app/data/chat.db".to_string());
        
        // Create directory if it doesn't exist (for SQLite file)
        if let Some(db_path) = database_url.strip_prefix("sqlite:") {
            if let Some(dir) = std::path::Path::new(db_path).parent() {
                tokio::fs::create_dir_all(dir).await.expect("Failed to create database directory");
            }
        }
        
        let db_pool = SqlitePool::connect_with(
            sqlx::sqlite::SqliteConnectOptions::from_str(&database_url)
                .expect("Invalid database URL")
                .create_if_missing(true)
        ).await.expect("Failed to connect to database");
        
        // Run migrations
        chat::init_database(&db_pool).await.expect("Failed to initialize database");
        auth::init_auth_tables(&db_pool).await.expect("Failed to initialize auth tables");

        let models = Arc::new(Mutex::new(HashMap::new()));
        let (tx, _rx) = tokio::sync::broadcast::channel(100);

        let state = AppState {
            models,
            db: db_pool,
            tx,
        };

        let cors = CorsLayer::new()
            .allow_origin(Any)
            .allow_methods([Method::GET, Method::POST, Method::DELETE, Method::OPTIONS])
            .allow_headers(Any);

        let app = Router::new()
            // Existing neural network routes
            .route("/models", post(create_model).get(list_models).delete(clear_all_models))
            .route("/models/import", post(import_model))
            .route("/models/:model_id", get(get_model_details).delete(delete_model))
            .route("/models/:model_id/metadata", get(get_model_metadata))
            .route("/models/:model_id/train", post(train_model))
            .route("/models/:model_id/train-stream", post(train_model_stream))
            .route("/models/:model_id/predict", post(predict))
            .route("/models/:model_id/export", get(export_model))
            .route("/health", get(health_check))
            // Authentication routes
            .route("/auth/register", post(auth::register))
            .route("/auth/login", post(auth::login))
            .route("/auth/user", get(auth::get_current_user))
            .route("/auth/profile", get(auth::get_profile).put(auth::update_profile))
            // Chat system routes
            .route("/chat/conversations", post(chat::create_conversation).get(chat::list_conversations))
            .route("/chat/conversations/:conversation_id/messages", get(chat::get_conversation_messages).post(chat::send_message))
            .route("/chat/ws", get(chat::websocket_handler))
            .route("/chat/upload", post(chat::upload_file))
            .route("/files/:file_id", get(chat::serve_file))
            .layer(cors)
            .layer(TraceLayer::new_for_http())
            .layer(middleware::from_fn(security_headers))
            .with_state(state);

        let addr: std::net::SocketAddr = "0.0.0.0:55320".parse().unwrap();
        tracing::info!("🚀 Upgraded Neural Network Backend listening on http://{}", addr);
        let listener = TcpListener::bind(addr).await.unwrap();
        axum::serve(listener, app).await.unwrap();
    }

    async fn health_check() -> Json<ApiResponse> {
        Json(ApiResponse {
            status: "success".into(),
            message: "Backend is healthy".into(),
            data: None,
        })
    }

    async fn security_headers(
        request: axum::extract::Request,
        next: axum::middleware::Next,
    ) -> axum::response::Response {
        let mut response = next.run(request).await;
        
        let headers = response.headers_mut();
        
        // Security headers
        headers.insert("X-Content-Type-Options", "nosniff".parse().unwrap());
        headers.insert("X-Frame-Options", "DENY".parse().unwrap());
        headers.insert("X-XSS-Protection", "1; mode=block".parse().unwrap());
        headers.insert("Referrer-Policy", "strict-origin-when-cross-origin".parse().unwrap());
        headers.insert("Permissions-Policy", "geolocation=(), microphone=(), camera=()".parse().unwrap());
        
        // Content Security Policy
        let csp = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; connect-src 'self' ws: wss:; font-src 'self' data:;";
        headers.insert("Content-Security-Policy", csp.parse().unwrap());
        
        response
    }

    // Input validation and sanitization
    fn sanitize_string(input: &str) -> String {
        input
            .chars()
            .filter(|c| !c.is_control() || *c == '\n' || *c == '\r' || *c == '\t')
            .collect::<String>()
            .trim()
            .to_string()
    }

    pub fn validate_username(username: &str) -> Result<String, String> {
        let sanitized = sanitize_string(username);
        
        if sanitized.len() < 3 {
            return Err("Username must be at least 3 characters long".to_string());
        }
        
        if sanitized.len() > 50 {
            return Err("Username must be less than 50 characters long".to_string());
        }
        
        if !sanitized.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '-') {
            return Err("Username can only contain alphanumeric characters, underscores, and hyphens".to_string());
        }
        
        Ok(sanitized)
    }

    pub fn validate_email(email: &str) -> Result<String, String> {
        let sanitized = sanitize_string(email);
        
        if sanitized.len() > 254 {
            return Err("Email must be less than 254 characters long".to_string());
        }
        
        if !sanitized.contains('@') || sanitized.matches('@').count() != 1 {
            return Err("Invalid email format".to_string());
        }
        
        let parts: Vec<&str> = sanitized.split('@').collect();
        if parts[0].is_empty() || parts[1].is_empty() {
            return Err("Invalid email format".to_string());
        }
        
        if !parts[1].contains('.') {
            return Err("Invalid email format".to_string());
        }
        
        Ok(sanitized)
    }

    pub fn validate_password(password: &str) -> Result<String, String> {
        if password.len() < 8 {
            return Err("Password must be at least 8 characters long".to_string());
        }
        
        if password.len() > 128 {
            return Err("Password must be less than 128 characters long".to_string());
        }
        
        // Check for at least one uppercase, one lowercase, and one digit
        let has_upper = password.chars().any(|c| c.is_uppercase());
        let has_lower = password.chars().any(|c| c.is_lowercase());
        let has_digit = password.chars().any(|c| c.is_ascii_digit());
        
        if !has_upper || !has_lower || !has_digit {
            return Err("Password must contain at least one uppercase letter, one lowercase letter, and one digit".to_string());
        }
        
        Ok(password.to_string())
    }

    async fn create_model(
        State(state): State<AppState>,
        Json(payload): Json<CreateModelRequest>,
    ) -> (StatusCode, Json<ApiResponse>) {
        tracing::info!("📥 Received model creation request");
        tracing::debug!("Config: {:?}", payload.graph);

        match GraphModel::from_config(&payload.graph) {
            Ok(model) => {
                let model_id = Uuid::new_v4().to_string();
                let model_details = model.to_value();
                state
                    .models
                    .lock()
                    .unwrap()
                    .insert(model_id.clone(), Box::new(model));
                tracing::info!("✅ Model created: {}", model_id);
                (
                    StatusCode::CREATED,
                    Json(ApiResponse {
                        status: "success".into(),
                        message: format!("Model {} created successfully", &model_id[..8]),
                        data: Some(serde_json::json!({
                            "model_id": model_id,
                            "architecture": model_details
                        })),
                    }),
                )
            }
            Err(e) => {
                tracing::error!("❌ Model creation failed: {}", e);
                (
                    StatusCode::BAD_REQUEST,
                    Json(ApiResponse {
                        status: "error".into(),
                        message: format!("Model creation failed: {}", e),
                        data: None,
                    }),
                )
            }
        }
    }

    fn json_to_tensor(json_val: &Value) -> Result<nn::core::tensor::Tensor, String> {
        let mut data = Vec::<f64>::new();
        let mut shape = Vec::<usize>::new();

        fn get_shape_and_check(
            val: &Value,
            shape: &mut Vec<usize>,
            level: usize,
        ) -> Result<(), String> {
            let arr = val
                .as_array()
                .ok_or_else(|| format!("Invalid JSON: not an array at level {}", level))?;
            if arr.is_empty() {
                return Ok(());
            }
            if shape.len() == level {
                shape.push(arr.len());
            } else if shape[level] != arr.len() {
                return Err("Inconsistent array dimensions.".to_string());
            }
            if !arr.is_empty() && arr[0].is_array() {
                for item in arr {
                    get_shape_and_check(item, shape, level + 1)?;
                }
            }
            Ok(())
        }

        get_shape_and_check(json_val, &mut shape, 0)?;

        fn flatten_json(v: &Value, data: &mut Vec<f64>) {
            if let Some(arr) = v.as_array() {
                for item in arr {
                    flatten_json(item, data);
                }
            } else if let Some(n) = v.as_f64() {
                data.push(n);
            }
        }

        flatten_json(json_val, &mut data);
        Ok(nn::core::tensor::Tensor::from_data(data, &shape))
    }

    fn tensor_to_json(tensor: &nn::core::tensor::Tensor) -> Value {
        let tensor_data = tensor.data.lock().unwrap();

        fn build_json(data_slice: &[f64], shape: &[usize]) -> Value {
            if shape.is_empty() || shape.iter().product::<usize>() == 0 {
                return serde_json::to_value(data_slice.get(0).unwrap_or(&0.0)).unwrap();
            }
            if shape.len() == 1 {
                return serde_json::to_value(data_slice).unwrap();
            }
            let stride = shape[1..].iter().product();
            let chunks: Vec<Value> = data_slice
                .chunks(stride)
                .map(|chunk| build_json(chunk, &shape[1..]))
                .collect();
            serde_json::to_value(chunks).unwrap()
        }

        build_json(&tensor_data, &tensor.shape)
    }

    async fn list_models(State(state): State<AppState>) -> (StatusCode, Json<ApiResponse>) {
        let model_ids: Vec<String> = state.models.lock().unwrap().keys().cloned().collect();
        tracing::info!("📋 Listed {} models", model_ids.len());
        (
            StatusCode::OK,
            Json(ApiResponse {
                status: "success".into(),
                message: format!("Found {} models", model_ids.len()),
                data: Some(serde_json::json!({ "models": model_ids })),
            }),
        )
    }

    async fn get_model_details(
        State(state): State<AppState>,
        Path(model_id): Path<String>,
    ) -> Result<Json<ApiResponse>, (StatusCode, Json<ApiResponse>)> {
        let models = state.models.lock().unwrap();
        let model = models.get(&model_id).ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ApiResponse {
                    status: "error".into(),
                    message: "Model not found".into(),
                    data: None,
                }),
            )
        })?;
        Ok(Json(ApiResponse {
            status: "success".into(),
            message: "Model details retrieved".into(),
            data: Some(serde_json::json!({ "architecture": model.to_value() })),
        }))
    }

    async fn get_model_metadata(
        State(state): State<AppState>,
        Path(model_id): Path<String>,
    ) -> Result<Json<ApiResponse>, (StatusCode, Json<ApiResponse>)> {
        let models = state.models.lock().unwrap();
        let model = models.get(&model_id).ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ApiResponse {
                    status: "error".into(),
                    message: "Model not found".into(),
                    data: None,
                }),
            )
        })?;

        let architecture = model.to_value();
        let empty_map = serde_json::Map::new();
        let nodes = architecture.get("nodes").and_then(|n| n.as_object()).unwrap_or(&empty_map);
        
        // Calculate model statistics
        let total_layers = nodes.len();
        let layer_types: std::collections::HashMap<String, usize> = nodes
            .values()
            .filter_map(|node| node.get("op").and_then(|op| op.as_str()))
            .fold(std::collections::HashMap::new(), |mut acc, op_type| {
                *acc.entry(op_type.to_string()).or_insert(0) += 1;
                acc
            });

        let has_linear = layer_types.contains_key("Linear");
        let has_activation = layer_types.iter().any(|(k, _)| 
            ["ReLU", "Gelu", "Sigmoid", "Tanh", "Softmax"].contains(&k.as_str())
        );
        let has_normalization = layer_types.contains_key("LayerNorm");

        let model_type = if has_linear && has_activation {
            "Neural Network"
        } else if has_linear {
            "Linear Model"
        } else if has_activation {
            "Activation Model"
        } else {
            "Unknown"
        };

        let total_params = model.get_named_params().values()
            .map(|tensor| tensor.size())
            .sum::<usize>();

        let metadata = serde_json::json!({
            "model_id": model_id,
            "model_type": model_type,
            "total_layers": total_layers,
            "total_parameters": total_params,
            "layer_types": layer_types,
            "capabilities": {
                "has_linear": has_linear,
                "has_activation": has_activation,
                "has_normalization": has_normalization,
                "is_classifier": has_linear && has_activation,
                "is_regressor": has_linear && !has_activation
            },
            "architecture_summary": {
                "input_nodes": architecture.get("inputs").and_then(|i| i.as_array()).map(|a| a.len()).unwrap_or(0),
                "output_nodes": architecture.get("output_node").map(|_| 1).unwrap_or(0),
                "hidden_layers": total_layers.saturating_sub(2)
            }
        });

        Ok(Json(ApiResponse {
            status: "success".into(),
            message: "Model metadata retrieved".into(),
            data: Some(metadata),
        }))
    }

    async fn train_model(
        State(state): State<AppState>,
        Path(model_id): Path<String>,
        Json(payload): Json<TrainRequest>,
    ) -> Result<Json<ApiResponse>, (StatusCode, Json<ApiResponse>)> {
        tracing::info!("🏋️ Starting training for model: {}", &model_id[..8]);

        let mut models = state.models.lock().unwrap();
        let model = models.get_mut(&model_id).ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ApiResponse {
                    status: "error".into(),
                    message: "Model not found".into(),
                    data: None,
                }),
            )
        })?;

        let x_data = json_to_tensor(&payload.x_train).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse {
                    status: "error".into(),
                    message: format!("Invalid x_train data: {}", e),
                    data: None,
                }),
            )
        })?;

        let y_data = json_to_tensor(&payload.y_train).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse {
                    status: "error".into(),
                    message: format!("Invalid y_train data: {}", e),
                    data: None,
                }),
            )
        })?;

        let mut optimizer = create_optimizer(payload.optimizer);
        let loss_fn = create_loss(payload.loss);

        let training_progress = nn::training::train(
            model.as_mut(),
            x_data,
            y_data,
            payload.epochs,
            payload.batch_size,
            &mut *optimizer,
            &*loss_fn,
        );

        tracing::info!("✅ Training complete");

        Ok(Json(ApiResponse {
            status: "success".to_string(),
            message: "Training completed successfully".to_string(),
            data: Some(serde_json::json!({
                "training_progress": training_progress,
                "final_metrics": training_progress.metrics.last(),
                "loss_history": training_progress.metrics.iter().map(|m| m.loss).collect::<Vec<f64>>(),
                "accuracy_history": training_progress.metrics.iter().map(|m| m.accuracy).collect::<Vec<f64>>(),
            })),
        }))
    }

    async fn train_model_stream(
        State(state): State<AppState>,
        Path(model_id): Path<String>,
        Json(payload): Json<TrainRequest>,
    ) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, Json<ApiResponse>)> {
        tracing::info!("🏋️ Starting streaming training for model: {}", &model_id[..8]);

        let models = state.models.lock().unwrap();
        let model_exists = models.contains_key(&model_id);
        drop(models);

        if !model_exists {
            return Err((
                StatusCode::NOT_FOUND,
                Json(ApiResponse {
                    status: "error".into(),
                    message: "Model not found".into(),
                    data: None,
                }),
            ));
        }

        let x_data = match json_to_tensor(&payload.x_train) {
            Ok(t) => t,
            Err(e) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ApiResponse {
                        status: "error".into(),
                        message: format!("Invalid x_train data: {}", e),
                        data: None,
                    }),
                ));
            }
        };

        let y_data = match json_to_tensor(&payload.y_train) {
            Ok(t) => t,
            Err(e) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(ApiResponse {
                        status: "error".into(),
                        message: format!("Invalid y_train data: {}", e),
                        data: None,
                    }),
                ));
            }
        };

        let optimizer_config = payload.optimizer;
        let loss_type = payload.loss;
        let epochs = payload.epochs;
        let batch_size = payload.batch_size;

        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        
        // Send initial "started" event
        let _ = tx.send(serde_json::json!({
            "type": "started",
            "data": {
                "epochs": epochs,
                "batch_size": batch_size
            }
        }));
        
        tracing::info!("Sent initial started event");
        
        // Spawn training in a separate blocking task
        let state_clone = state.clone();
        let model_id_clone = model_id.clone();
        tokio::task::spawn_blocking(move || {
            tracing::info!("🏋️ Training task started for model {}", &model_id_clone[..8]);
            
            // Send a progress indicator
            let _ = tx.send(serde_json::json!({
                "type": "info",
                "data": { "message": "Training task initialized" }
            }));
            
            let mut optimizer = create_optimizer(optimizer_config);
            let loss_fn = create_loss(loss_type);

            tracing::info!("Optimizer and loss function created");
            let _ = tx.send(serde_json::json!({
                "type": "info",
                "data": { "message": "Optimizer configured" }
            }));

            let mut models = state_clone.models.lock().unwrap();
            let model = models.get_mut(&model_id_clone).unwrap();

            tracing::info!("Starting training loop with {} epochs", epochs);
            let _ = tx.send(serde_json::json!({
                "type": "info",
                "data": { "message": format!("Starting {} epochs", epochs) }
            }));

            let training_result = nn::training::train_with_callback(
                model.as_mut(),
                x_data,
                y_data,
                epochs,
                batch_size,
                &mut *optimizer,
                &*loss_fn,
                |metric| {
                    let event_data = serde_json::json!({
                        "type": "progress",
                        "data": metric
                    });
                    tracing::info!("📊 Epoch {}/{}: Loss={:.6}", metric.epoch, epochs, metric.loss);
                    if let Err(e) = tx.send(event_data) {
                        tracing::error!("Failed to send progress event: {}", e);
                    }
                    None
                }
            );

            tracing::info!("✅ Training loop completed");

            // Send completion event
            let completion_data = serde_json::json!({
                "type": "complete",
                "data": {
                    "training_progress": training_result,
                    "final_metrics": training_result.metrics.last(),
                }
            });
            if let Err(e) = tx.send(completion_data) {
                tracing::error!("Failed to send completion event: {}", e);
            } else {
                tracing::info!("Completion event sent successfully");
            }
        });

        tracing::info!("Setting up SSE stream");
        
        let stream = UnboundedReceiverStream::new(rx).map(|data| {
            tracing::debug!("Streaming event to client");
            Ok(Event::default().json_data(data).unwrap())
        });

        Ok(Sse::new(stream).keep_alive(
            axum::response::sse::KeepAlive::new()
                .interval(std::time::Duration::from_secs(1))
                .text("keep-alive"),
        ))
    }

    async fn predict(
        State(state): State<AppState>,
        Path(model_id): Path<String>,
        Json(payload): Json<PredictRequest>,
    ) -> Result<Json<ApiResponse>, (StatusCode, Json<ApiResponse>)> {
        let mut models = state.models.lock().unwrap();
        let model = models.get_mut(&model_id).ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ApiResponse {
                    status: "error".into(),
                    message: "Model not found".into(),
                    data: None,
                }),
            )
        })?;

        let x_data = json_to_tensor(&payload.x).map_err(|e| {
            (
                StatusCode::BAD_REQUEST,
                Json(ApiResponse {
                    status: "error".into(),
                    message: format!("Invalid input data: {}", e),
                    data: None,
                }),
            )
        })?;

        let predictions = model.forward(&x_data);

        Ok(Json(ApiResponse {
            status: "success".to_string(),
            message: "Prediction successful".to_string(),
            data: Some(serde_json::json!({ "predictions": tensor_to_json(&predictions) })),
        }))
    }

    async fn export_model(
        State(state): State<AppState>,
        Path(model_id): Path<String>,
    ) -> Result<Json<ExportedModel>, (StatusCode, Json<ApiResponse>)> {
        let models = state.models.lock().unwrap();
        let model = models.get(&model_id).ok_or_else(|| {
            (
                StatusCode::NOT_FOUND,
                Json(ApiResponse {
                    status: "error".into(),
                    message: "Model not found".into(),
                    data: None,
                }),
            )
        })?;

        let architecture = model.to_value();
        let mut parameters = HashMap::new();
        for (name, tensor) in model.get_named_params() {
            let s_tensor = SerializableTensor {
                shape: tensor.shape.clone(),
                data: tensor.data.lock().unwrap().clone(),
            };
            parameters.insert(name.clone(), s_tensor);
        }

        tracing::info!("📦 Exported model {}", &model_id[..8]);
        Ok(Json(ExportedModel {
            architecture,
            parameters,
        }))
    }

    async fn import_model(
        State(state): State<AppState>,
        Json(payload): Json<ExportedModel>,
    ) -> (StatusCode, Json<ApiResponse>) {
        tracing::info!("📥 Received model import request");
        match GraphModel::from_config(&payload.architecture) {
            Ok(mut model) => {
                for (name, s_tensor) in payload.parameters {
                    let new_tensor = nn::core::tensor::Tensor::from_data(s_tensor.data, &s_tensor.shape);
                    model.set_param(&name, new_tensor);
                }

                let model_id = Uuid::new_v4().to_string();
                state
                    .models
                    .lock()
                    .unwrap()
                    .insert(model_id.clone(), Box::new(model));

                tracing::info!("✅ Model imported successfully: {}", &model_id[..8]);
                (
                    StatusCode::CREATED,
                    Json(ApiResponse {
                        status: "success".into(),
                        message: "Model imported successfully".into(),
                        data: Some(serde_json::json!({ "model_id": model_id })),
                    }),
                )
            }
            Err(e) => {
                tracing::error!("❌ Model import failed: {}", e);
                (
                    StatusCode::BAD_REQUEST,
                    Json(ApiResponse {
                        status: "error".into(),
                        message: format!("Model import failed: {}", e),
                        data: None,
                    }),
                )
            }
        }
    }

    async fn clear_all_models(State(state): State<AppState>) -> (StatusCode, Json<ApiResponse>) {
        let mut models = state.models.lock().unwrap();
        let count = models.len();
        models.clear();
        
        tracing::info!("🗑️ Cleared {} models from memory", count);
        (
            StatusCode::OK,
            Json(ApiResponse {
                status: "success".into(),
                message: format!("Cleared {} models from memory", count),
                data: Some(serde_json::json!({ "cleared_count": count })),
            }),
        )
    }

    async fn delete_model(
        State(state): State<AppState>,
        Path(model_id): Path<String>,
    ) -> (StatusCode, Json<ApiResponse>) {
        let mut models = state.models.lock().unwrap();
        let removed = models.remove(&model_id);
        
        if removed.is_some() {
            tracing::info!("🗑️ Deleted model: {}", &model_id[..8]);
            (
                StatusCode::OK,
                Json(ApiResponse {
                    status: "success".into(),
                    message: "Model deleted successfully".into(),
                    data: Some(serde_json::json!({ "deleted_model_id": model_id })),
                }),
            )
        } else {
            tracing::warn!("❌ Model not found for deletion: {}", &model_id[..8]);
            (
                StatusCode::NOT_FOUND,
                Json(ApiResponse {
                    status: "error".into(),
                    message: "Model not found".into(),
                    data: None,
                }),
            )
        }
    }
}

// ============================================================================
// CHAT SYSTEM
// ============================================================================

mod chat {
    use super::*;
    use futures::{SinkExt, StreamExt};

    pub type ChatState = api::AppState;

    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct Conversation {
        pub id: String,
        pub title: String,
        pub created_at: DateTime<Utc>,
        pub updated_at: DateTime<Utc>,
        pub user_id: Option<String>,
        pub metadata: Option<Value>,
    }

    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct ChatMessage {
        pub id: String,
        pub conversation_id: String,
        pub role: MessageRole,
        pub content: MessageContent,
        pub created_at: DateTime<Utc>,
        pub metadata: Option<Value>,
    }

    #[derive(Serialize, Deserialize, Debug, Clone)]
    #[serde(rename_all = "lowercase")]
    pub enum MessageRole {
        User,
        Assistant,
        System,
    }

    #[derive(Serialize, Deserialize, Debug, Clone)]
    #[serde(tag = "type", content = "data")]
    pub enum MessageContent {
        Text(String),
        Image { url: String, caption: Option<String> },
        Video { url: String, caption: Option<String> },
        Audio { url: String, duration: Option<f64> },
        Document { url: String, filename: String, mime_type: String },
        Mixed(Vec<MessageContent>),
    }

    #[derive(Serialize, Deserialize, Debug)]
    pub struct CreateConversationRequest {
        pub title: Option<String>,
        pub metadata: Option<Value>,
    }

    #[derive(Serialize, Deserialize, Debug)]
    pub struct SendMessageRequest {
        pub content: MessageContent,
        pub model_id: Option<String>,
        pub stream: Option<bool>,
    }

    #[derive(Serialize, Deserialize, Debug)]
    pub struct ConversationListQuery {
        pub limit: Option<i64>,
        pub offset: Option<i64>,
        pub search: Option<String>,
    }

    pub async fn init_database(pool: &SqlitePool) -> Result<(), sqlx::Error> {
        // Create conversations table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                user_id TEXT,
                metadata TEXT
            )
            "#,
        )
        .execute(pool)
        .await?;

        // Create messages table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            )
            "#,
        )
        .execute(pool)
        .await?;

        // Create files table
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                mime_type TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT
            )
            "#,
        )
        .execute(pool)
        .await?;

        Ok(())
    }

    pub async fn create_conversation(
        State(state): State<ChatState>,
        Json(payload): Json<CreateConversationRequest>,
    ) -> Result<Json<Conversation>, (StatusCode, Json<Value>)> {
        let conversation_id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let title = payload.title.unwrap_or_else(|| "New Conversation".to_string());

        let conversation = Conversation {
            id: conversation_id.clone(),
            title: title.clone(),
            created_at: now,
            updated_at: now,
            user_id: None, // TODO: Get from auth
            metadata: payload.metadata,
        };

        let metadata_json = conversation.metadata
            .as_ref()
            .map(|m| serde_json::to_string(m).unwrap_or_default())
            .unwrap_or_default();

        sqlx::query(
            r#"
            INSERT INTO conversations (id, title, created_at, updated_at, user_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&conversation_id)
        .bind(&title)
        .bind(now.to_rfc3339())
        .bind(now.to_rfc3339())
        .bind(&conversation.user_id)
        .bind(&metadata_json)
        .execute(&state.db)
        .await
        .map_err(|e| {
            tracing::error!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "Failed to create conversation" })),
            )
        })?;

        tracing::info!("Created conversation: {}", &conversation_id[..8]);
        Ok(Json(conversation))
    }

    pub async fn list_conversations(
        State(state): State<ChatState>,
        Query(query): Query<ConversationListQuery>,
    ) -> Result<Json<Vec<Conversation>>, (StatusCode, Json<Value>)> {
        let limit = query.limit.unwrap_or(50).min(100);
        let offset = query.offset.unwrap_or(0);

        let mut sql = "SELECT id, title, created_at, updated_at, user_id, metadata FROM conversations".to_string();
        let mut params = Vec::new();

        if let Some(search) = &query.search {
            sql.push_str(" WHERE title LIKE ?");
            params.push(format!("%{}%", search));
        }

        sql.push_str(" ORDER BY updated_at DESC LIMIT ? OFFSET ?");

        let mut query_builder = sqlx::query(&sql);
        for param in params {
            query_builder = query_builder.bind(param);
        }
        query_builder = query_builder.bind(limit).bind(offset);

        let rows = query_builder.fetch_all(&state.db).await.map_err(|e| {
            tracing::error!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "Failed to fetch conversations" })),
            )
        })?;

        let conversations: Vec<Conversation> = rows
            .into_iter()
            .map(|row| {
                let metadata_str: String = row.get("metadata");
                let metadata = if metadata_str.is_empty() {
                    None
                } else {
                    serde_json::from_str(&metadata_str).ok()
                };

                Conversation {
                    id: row.get("id"),
                    title: row.get("title"),
                    created_at: DateTime::parse_from_rfc3339(&row.get::<String, _>("created_at"))
                        .unwrap()
                        .with_timezone(&Utc),
                    updated_at: DateTime::parse_from_rfc3339(&row.get::<String, _>("updated_at"))
                        .unwrap()
                        .with_timezone(&Utc),
                    user_id: row.get("user_id"),
                    metadata,
                }
            })
            .collect();

        Ok(Json(conversations))
    }

    pub async fn get_conversation_messages(
        State(state): State<ChatState>,
        Path(conversation_id): Path<String>,
        Query(query): Query<ConversationListQuery>,
    ) -> Result<Json<Vec<ChatMessage>>, (StatusCode, Json<Value>)> {
        let limit = query.limit.unwrap_or(100).min(500);
        let offset = query.offset.unwrap_or(0);

        let rows = sqlx::query(
            r#"
            SELECT id, conversation_id, role, content, created_at, metadata 
            FROM messages 
            WHERE conversation_id = ? 
            ORDER BY created_at ASC 
            LIMIT ? OFFSET ?
            "#,
        )
        .bind(&conversation_id)
        .bind(limit)
        .bind(offset)
        .fetch_all(&state.db)
        .await
        .map_err(|e| {
            tracing::error!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "Failed to fetch messages" })),
            )
        })?;

        let messages: Vec<ChatMessage> = rows
            .into_iter()
            .map(|row| {
                let content_str: String = row.get("content");
                let content: MessageContent = serde_json::from_str(&content_str)
                    .unwrap_or(MessageContent::Text("Error parsing message".to_string()));

                let metadata_str: String = row.get("metadata");
                let metadata = if metadata_str.is_empty() {
                    None
                } else {
                    serde_json::from_str(&metadata_str).ok()
                };

                ChatMessage {
                    id: row.get("id"),
                    conversation_id: row.get("conversation_id"),
                    role: match row.get::<String, _>("role").as_str() {
                        "user" => MessageRole::User,
                        "assistant" => MessageRole::Assistant,
                        "system" => MessageRole::System,
                        _ => MessageRole::User,
                    },
                    content,
                    created_at: DateTime::parse_from_rfc3339(&row.get::<String, _>("created_at"))
                        .unwrap()
                        .with_timezone(&Utc),
                    metadata,
                }
            })
            .collect();

        Ok(Json(messages))
    }

    pub async fn send_message(
        State(state): State<ChatState>,
        Path(conversation_id): Path<String>,
        Json(payload): Json<SendMessageRequest>,
    ) -> Result<Json<ChatMessage>, (StatusCode, Json<Value>)> {
        // Verify conversation exists
        let conversation_exists = sqlx::query("SELECT 1 FROM conversations WHERE id = ?")
            .bind(&conversation_id)
            .fetch_optional(&state.db)
            .await
            .map_err(|e| {
                tracing::error!("Database error: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": "Database error" })),
                )
            })?
            .is_some();

        if !conversation_exists {
            return Err((
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": "Conversation not found" })),
            ));
        }

        // Save user message
        let user_message_id = Uuid::new_v4().to_string();
        let now = Utc::now();

        let user_message = ChatMessage {
            id: user_message_id.clone(),
            conversation_id: conversation_id.clone(),
            role: MessageRole::User,
            content: payload.content.clone(),
            created_at: now,
            metadata: None,
        };

        let content_json = serde_json::to_string(&payload.content).unwrap();

        sqlx::query(
            r#"
            INSERT INTO messages (id, conversation_id, role, content, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&user_message_id)
        .bind(&conversation_id)
        .bind("user")
        .bind(&content_json)
        .bind(now.to_rfc3339())
        .bind("")
        .execute(&state.db)
        .await
        .map_err(|e| {
            tracing::error!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "Failed to save message" })),
            )
        })?;

        // Update conversation timestamp
        sqlx::query("UPDATE conversations SET updated_at = ? WHERE id = ?")
            .bind(now.to_rfc3339())
            .bind(&conversation_id)
            .execute(&state.db)
            .await
            .map_err(|e| {
                tracing::error!("Database error: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": "Failed to update conversation" })),
                )
            })?;

        // Broadcast message to WebSocket clients
        let _ = state.tx.send(user_message.clone());

        // Generate AI response
        tokio::spawn(generate_ai_response(
            state.clone(),
            conversation_id,
            payload.content,
            payload.model_id,
        ));

        Ok(Json(user_message))
    }

    async fn generate_ai_response(
        state: ChatState,
        conversation_id: String,
        user_content: MessageContent,
        model_id: Option<String>,
    ) {
        let response_content = match route_to_model(&state, &user_content, model_id) {
            Ok(content) => content,
            Err(e) => {
                tracing::error!("Model routing error: {}", e);
                MessageContent::Text("I'm sorry, I encountered an error processing your request.".to_string())
            }
        };

        // Save AI response
        let ai_message_id = Uuid::new_v4().to_string();
        let now = Utc::now();

        let ai_message = ChatMessage {
            id: ai_message_id.clone(),
            conversation_id: conversation_id.clone(),
            role: MessageRole::Assistant,
            content: response_content,
            created_at: now,
            metadata: None,
        };

        let content_json = serde_json::to_string(&ai_message.content).unwrap();

        if let Err(e) = sqlx::query(
            r#"
            INSERT INTO messages (id, conversation_id, role, content, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            "#,
        )
        .bind(&ai_message_id)
        .bind(&conversation_id)
        .bind("assistant")
        .bind(&content_json)
        .bind(now.to_rfc3339())
        .bind("")
        .execute(&state.db)
        .await
        {
            tracing::error!("Failed to save AI response: {}", e);
            return;
        }

        // Update conversation timestamp
        let _ = sqlx::query("UPDATE conversations SET updated_at = ? WHERE id = ?")
            .bind(now.to_rfc3339())
            .bind(&conversation_id)
            .execute(&state.db)
            .await;

        // Broadcast AI response
        let _ = state.tx.send(ai_message);
    }

    fn route_to_model(
        state: &ChatState,
        content: &MessageContent,
        model_id: Option<String>,
    ) -> Result<MessageContent, String> {
        match content {
            MessageContent::Text(text) => {
                // Route to text model
                if let Some(model_id) = model_id {
                    let models = state.models.lock().unwrap();
                    if let Some(_model) = models.get(&model_id) {
                        // For now, return a simple response
                        // TODO: Implement actual model inference
                        Ok(MessageContent::Text(format!("Echo from model {}: {}", &model_id[..8], text)))
                    } else {
                        Err("Model not found".to_string())
                    }
                } else {
                    // Default text response
                    Ok(MessageContent::Text(format!("I received your message: {}", text)))
                }
            }
            MessageContent::Image { url: _, caption } => {
                // Route to vision model
                Ok(MessageContent::Text(format!(
                    "I see an image{}. Image analysis would be processed here.",
                    caption.as_ref().map(|c| format!(" with caption: {}", c)).unwrap_or_default()
                )))
            }
            MessageContent::Mixed(contents) => {
                // Handle multiple inputs
                let mut responses = Vec::new();
                for content in contents {
                    if let Ok(response) = route_to_model(state, content, model_id.clone()) {
                        responses.push(response);
                    }
                }
                Ok(MessageContent::Mixed(responses))
            }
            _ => Ok(MessageContent::Text("I can process this type of content, but the handler isn't implemented yet.".to_string())),
        }
    }

    pub async fn websocket_handler(
        ws: WebSocketUpgrade,
        State(state): State<ChatState>,
    ) -> Response {
        ws.on_upgrade(|socket| websocket_connection(socket, state))
    }

    async fn websocket_connection(socket: WebSocket, state: ChatState) {
        let mut rx = state.tx.subscribe();
        let (mut sender, mut receiver) = socket.split();

        let send_task = tokio::spawn(async move {
            while let Ok(msg) = rx.recv().await {
                let json_msg = serde_json::to_string(&msg).unwrap();
                if sender.send(Message::Text(json_msg)).await.is_err() {
                    break;
                }
            }
        });

        let recv_task = tokio::spawn(async move {
            while let Some(msg) = receiver.next().await {
                if let Ok(msg) = msg {
                    if let Message::Text(text) = msg {
                        tracing::info!("Received WebSocket message: {}", text);
                    }
                }
            }
        });

        tokio::select! {
            _ = send_task => {},
            _ = recv_task => {},
        }
    }

    pub async fn upload_file(
        State(state): State<ChatState>,
        mut multipart: Multipart,
    ) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
        let mut files = Vec::new();

        while let Some(field) = multipart.next_field().await.map_err(|e| {
            tracing::error!("Multipart error: {}", e);
            (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "Invalid multipart data" })),
            )
        })? {
            if let Some(filename) = field.file_name().map(|s| s.to_string()) {
                let content_type = field.content_type().unwrap_or("application/octet-stream").to_string();
                let data = field.bytes().await.map_err(|e| {
                    tracing::error!("File read error: {}", e);
                    (
                        StatusCode::BAD_REQUEST,
                        Json(serde_json::json!({ "error": "Failed to read file" })),
                    )
                })?;

                let file_id = Uuid::new_v4().to_string();
                let file_path = format!("uploads/{}", file_id);

                // Create uploads directory if it doesn't exist
                tokio::fs::create_dir_all("uploads").await.map_err(|e| {
                    tracing::error!("Directory creation error: {}", e);
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({ "error": "Failed to create upload directory" })),
                    )
                })?;

                // Save file to disk
                tokio::fs::write(&file_path, &data).await.map_err(|e| {
                    tracing::error!("File write error: {}", e);
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({ "error": "Failed to save file" })),
                    )
                })?;

                // Save file metadata to database
                let now = Utc::now();
                sqlx::query(
                    r#"
                    INSERT INTO files (id, filename, original_filename, mime_type, size_bytes, path, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    "#,
                )
                .bind(&file_id)
                .bind(&file_id)
                .bind(&filename)
                .bind(&content_type)
                .bind(data.len() as i64)
                .bind(&file_path)
                .bind(now.to_rfc3339())
                .bind("")
                .execute(&state.db)
                .await
                .map_err(|e| {
                    tracing::error!("Database error: {}", e);
                    (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        Json(serde_json::json!({ "error": "Failed to save file metadata" })),
                    )
                })?;

                files.push(serde_json::json!({
                    "id": file_id,
                    "filename": filename,
                    "mime_type": content_type,
                    "size": data.len(),
                    "url": format!("/files/{}", file_id)
                }));

                tracing::info!("Uploaded file: {} ({} bytes)", filename, data.len());
            }
        }

        Ok(Json(serde_json::json!({ "files": files })))
    }

    pub async fn serve_file(
        State(state): State<ChatState>,
        Path(file_id): Path<String>,
    ) -> Result<Response, (StatusCode, Json<Value>)> {
        let row = sqlx::query("SELECT path, mime_type, original_filename FROM files WHERE id = ?")
            .bind(&file_id)
            .fetch_optional(&state.db)
            .await
            .map_err(|e| {
                tracing::error!("Database error: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": "Database error" })),
                )
            })?;

        let row = row.ok_or((
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": "File not found" })),
        ))?;

        let file_path: String = row.get("path");
        let mime_type: String = row.get("mime_type");
        let original_filename: String = row.get("original_filename");

        let file_data = tokio::fs::read(&file_path).await.map_err(|e| {
            tracing::error!("File read error: {}", e);
            (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": "File not found on disk" })),
            )
        })?;

        let headers = [
            (header::CONTENT_TYPE, mime_type.as_str()),
            (header::CONTENT_DISPOSITION, &format!("inline; filename=\"{}\"", original_filename)),
        ];

        Ok((headers, file_data).into_response())
    }
}

// ============================================================================
// AUTHENTICATION SYSTEM
// ============================================================================

mod auth {
    use super::*;
    use axum::http::HeaderMap;

    const JWT_SECRET: &str = "your-secret-key-change-in-production"; // TODO: Use environment variable
    
    pub fn get_jwt_secret() -> String {
        std::env::var("JWT_SECRET").unwrap_or_else(|_| JWT_SECRET.to_string())
    }

    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct User {
        pub id: String,
        pub username: String,
        pub email: String,
        pub created_at: DateTime<Utc>,
    }

    #[derive(Serialize, Deserialize, Debug)]
    pub struct Claims {
        pub user_id: String,
        pub username: String,
        pub exp: usize,
    }

    #[derive(Deserialize, Debug)]
    pub struct RegisterRequest {
        pub username: String,
        pub email: String,
        pub password: String,
    }

    #[derive(Deserialize, Debug)]
    pub struct LoginRequest {
        pub username: String,
        pub password: String,
    }

    #[derive(Serialize, Debug)]
    pub struct AuthResponse {
        pub token: String,
        pub user: User,
    }

    pub async fn init_auth_tables(pool: &SqlitePool) -> Result<(), sqlx::Error> {
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            "#,
        )
        .execute(pool)
        .await?;

        Ok(())
    }

    pub async fn register(
        State(state): State<api::AppState>,
        Json(payload): Json<RegisterRequest>,
    ) -> Result<Json<AuthResponse>, (StatusCode, Json<Value>)> {
        // Validate and sanitize input
        let username = match api::validate_username(&payload.username) {
            Ok(u) => u,
            Err(e) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": e })),
                ));
            }
        };

        let email = match api::validate_email(&payload.email) {
            Ok(e) => e,
            Err(e) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": e })),
                ));
            }
        };

        let _password = match api::validate_password(&payload.password) {
            Ok(p) => p,
            Err(e) => {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": e })),
                ));
            }
        };

        // Check if user already exists
        let existing_user = sqlx::query("SELECT id FROM users WHERE username = ? OR email = ?")
            .bind(&username)
            .bind(&email)
            .fetch_optional(&state.db)
            .await
            .map_err(|e| {
                tracing::error!("Database error: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": "Database error" })),
                )
            })?;

        if existing_user.is_some() {
            return Err((
                StatusCode::CONFLICT,
                Json(serde_json::json!({ "error": "Username or email already exists" })),
            ));
        }

        // Hash password
        let password_hash = hash(&payload.password, DEFAULT_COST).map_err(|e| {
            tracing::error!("Password hashing error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "Failed to process password" })),
            )
        })?;

        // Create user
        let user_id = Uuid::new_v4().to_string();
        let now = Utc::now();

        sqlx::query(
            r#"
            INSERT INTO users (id, username, email, password_hash, created_at)
            VALUES (?, ?, ?, ?, ?)
            "#,
        )
        .bind(&user_id)
        .bind(&username)
        .bind(&email)
        .bind(&password_hash)
        .bind(now.to_rfc3339())
        .execute(&state.db)
        .await
        .map_err(|e| {
            tracing::error!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "Failed to create user" })),
            )
        })?;

        let user = User {
            id: user_id.clone(),
            username: payload.username,
            email: payload.email,
            created_at: now,
        };

        // Generate JWT token
        let token = generate_token(&user)?;

        tracing::info!("User registered: {}", user.username);

        Ok(Json(AuthResponse { token, user }))
    }

    pub async fn login(
        State(state): State<api::AppState>,
        Json(payload): Json<LoginRequest>,
    ) -> Result<Json<AuthResponse>, (StatusCode, Json<Value>)> {
        // Find user
        let row = sqlx::query(
            "SELECT id, username, email, password_hash, created_at FROM users WHERE username = ?",
        )
        .bind(&payload.username)
        .fetch_optional(&state.db)
        .await
        .map_err(|e| {
            tracing::error!("Database error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "Database error" })),
            )
        })?;

        let row = row.ok_or((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "Invalid username or password" })),
        ))?;

        // Verify password
        let password_hash: String = row.get("password_hash");
        let is_valid = verify(&payload.password, &password_hash).map_err(|e| {
            tracing::error!("Password verification error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "Authentication error" })),
            )
        })?;

        if !is_valid {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "error": "Invalid username or password" })),
            ));
        }

        let user = User {
            id: row.get("id"),
            username: row.get("username"),
            email: row.get("email"),
            created_at: DateTime::parse_from_rfc3339(&row.get::<String, _>("created_at"))
                .unwrap()
                .with_timezone(&Utc),
        };

        // Generate JWT token
        let token = generate_token(&user)?;

        tracing::info!("User logged in: {}", user.username);

        Ok(Json(AuthResponse { token, user }))
    }

    pub fn generate_token(user: &User) -> Result<String, (StatusCode, Json<Value>)> {
        let expiration = Utc::now()
            .checked_add_signed(chrono::Duration::hours(24))
            .expect("valid timestamp")
            .timestamp() as usize;

        let claims = Claims {
            user_id: user.id.clone(),
            username: user.username.clone(),
            exp: expiration,
        };

        let token = encode(
            &Header::default(),
            &claims,
            &EncodingKey::from_secret(get_jwt_secret().as_ref()),
        )
        .map_err(|e| {
            tracing::error!("JWT encoding error: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "Token generation failed" })),
            )
        })?;

        Ok(token)
    }

    pub fn verify_token(token: &str) -> Result<Claims, (StatusCode, Json<Value>)> {
        let token_data = decode::<Claims>(
            token,
            &DecodingKey::from_secret(get_jwt_secret().as_ref()),
            &Validation::new(Algorithm::HS256),
        )
        .map_err(|e| {
            tracing::debug!("JWT verification error: {}", e);
            (
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "error": "Invalid token" })),
            )
        })?;

        Ok(token_data.claims)
    }

    pub fn extract_user_from_headers(headers: &HeaderMap) -> Result<Claims, (StatusCode, Json<Value>)> {
        let auth_header = headers
            .get("authorization")
            .ok_or((
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "error": "Missing authorization header" })),
            ))?
            .to_str()
            .map_err(|_| {
                (
                    StatusCode::UNAUTHORIZED,
                    Json(serde_json::json!({ "error": "Invalid authorization header" })),
                )
            })?;

        if !auth_header.starts_with("Bearer ") {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(serde_json::json!({ "error": "Invalid authorization format" })),
            ));
        }

        let token = &auth_header[7..];
        verify_token(token)
    }

    pub async fn get_current_user(
        State(state): State<api::AppState>,
        headers: HeaderMap,
    ) -> Result<Json<User>, (StatusCode, Json<Value>)> {
        let claims = extract_user_from_headers(&headers)?;

        let row = sqlx::query("SELECT id, username, email, created_at FROM users WHERE id = ?")
            .bind(&claims.user_id)
            .fetch_optional(&state.db)
            .await
            .map_err(|e| {
                tracing::error!("Database error: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": "Database error" })),
                )
            })?;

        let row = row.ok_or((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "User not found" })),
        ))?;

        let user = User {
            id: row.get("id"),
            username: row.get("username"),
            email: row.get("email"),
            created_at: DateTime::parse_from_rfc3339(&row.get::<String, _>("created_at"))
                .unwrap()
                .with_timezone(&Utc),
        };

        Ok(Json(user))
    }

    pub async fn get_profile(
        State(state): State<api::AppState>,
        headers: HeaderMap,
    ) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
        let claims = extract_user_from_headers(&headers)?;

        let row = sqlx::query("SELECT id, username, email, created_at, preferences FROM users WHERE id = ?")
            .bind(&claims.user_id)
            .fetch_optional(&state.db)
            .await
            .map_err(|e| {
                tracing::error!("Database error: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": "Database error" })),
                )
            })?;

        let row = row.ok_or((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "User not found" })),
        ))?;

        let preferences: Option<String> = row.get("preferences");
        let user_profile = serde_json::json!({
            "id": row.get::<String, _>("id"),
            "username": row.get::<String, _>("username"),
            "email": row.get::<String, _>("email"),
            "created_at": row.get::<String, _>("created_at"),
            "preferences": preferences.and_then(|p| serde_json::from_str(&p).ok()).unwrap_or(serde_json::json!({}))
        });

        Ok(Json(user_profile))
    }

    pub async fn update_profile(
        State(state): State<api::AppState>,
        headers: HeaderMap,
        Json(update_data): Json<Value>,
    ) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
        let claims = extract_user_from_headers(&headers)?;

        let mut update_fields = Vec::new();
        let mut query_builder = sqlx::QueryBuilder::new("UPDATE users SET ");

        if let Some(username) = update_data.get("username").and_then(|v| v.as_str()) {
            if username.len() < 3 || username.len() > 50 {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": "Username must be between 3 and 50 characters" })),
                ));
            }
            if !update_fields.is_empty() {
                query_builder.push(", ");
            }
            query_builder.push("username = ");
            query_builder.push_bind(username);
            update_fields.push("username");
        }

        if let Some(email) = update_data.get("email").and_then(|v| v.as_str()) {
            if !email.contains('@') {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "error": "Invalid email format" })),
                ));
            }
            if !update_fields.is_empty() {
                query_builder.push(", ");
            }
            query_builder.push("email = ");
            query_builder.push_bind(email);
            update_fields.push("email");
        }

        if let Some(preferences) = update_data.get("preferences") {
            if !update_fields.is_empty() {
                query_builder.push(", ");
            }
            query_builder.push("preferences = ");
            query_builder.push_bind(serde_json::to_string(preferences).unwrap_or_else(|_| "{}".to_string()));
            update_fields.push("preferences");
        }

        if update_fields.is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "No valid fields to update" })),
            ));
        }

        query_builder.push(" WHERE id = ");
        query_builder.push_bind(&claims.user_id);

        let result = query_builder.build()
            .execute(&state.db)
            .await
            .map_err(|e| {
                tracing::error!("Database error: {}", e);
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": "Database error" })),
                )
            })?;

        if result.rows_affected() == 0 {
            return Err((
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "error": "User not found" })),
            ));
        }

        Ok(Json(serde_json::json!({ "message": "Profile updated successfully" })))
    }
}

// ============================================================================
// NEURAL NETWORK ENGINE
// ============================================================================

pub mod nn {
    use self::core::tensor::Tensor;
    use self::models::AiModel;

    pub mod core {
        pub mod tensor {
            use super::autograd::Context;
            use std::collections::HashSet;
            use std::ops::{Add, Mul, Sub};
            use std::sync::atomic::{AtomicUsize, Ordering};
            use std::sync::{Arc, Mutex};

            static TENSOR_COUNTER: AtomicUsize = AtomicUsize::new(0);

            #[derive(Debug, Clone)]
            pub struct Tensor {
                pub id: usize,
                pub data: Arc<Mutex<Vec<f64>>>,
                pub shape: Vec<usize>,
                pub grad: Arc<Mutex<Option<Box<Tensor>>>>,
                pub ctx: Option<Arc<Context>>,
            }

            impl Tensor {
                pub fn new(data: Vec<f64>, shape: &[usize], ctx: Option<Arc<Context>>) -> Self {
                    if !shape.is_empty() {
                        assert_eq!(
                            data.len(),
                            shape.iter().product::<usize>(),
                            "Data size does not match shape"
                        );
                    }
                    let id = TENSOR_COUNTER.fetch_add(1, Ordering::Relaxed);
                    Self {
                        id,
                        data: Arc::new(Mutex::new(data)),
                        shape: shape.to_vec(),
                        grad: Arc::new(Mutex::new(None)),
                        ctx,
                    }
                }

                pub fn from_data(data: Vec<f64>, shape: &[usize]) -> Self {
                    Self::new(data, shape, None)
                }

                pub fn backward(&self) {
                    let mut visited = HashSet::new();
                    let mut tape = Vec::new();

                    fn build_tape(node: &Tensor, visited: &mut HashSet<usize>, tape: &mut Vec<Tensor>) {
                        if visited.contains(&node.id) {
                            return;
                        }
                        visited.insert(node.id);
                        if let Some(ctx) = &node.ctx {
                            for parent in &ctx.parents {
                                build_tape(parent, visited, tape);
                            }
                        }
                        tape.push(node.clone());
                    }

                    build_tape(self, &mut visited, &mut tape);

                    *self.grad.lock().unwrap() = Some(Box::new(Tensor::from_data(
                        vec![1.0; self.size().max(1)],
                        &self.shape,
                    )));

                    for node in tape.iter().rev() {
                        if let Some(ctx) = &node.ctx {
                            if let Some(grad) = node.grad.lock().unwrap().clone() {
                                let parent_grads = ctx.op.backward(ctx, &grad);
                                for (i, parent) in ctx.parents.iter().enumerate() {
                                    let mut parent_grad = parent.grad.lock().unwrap();
                                    if let Some(pg) = parent_grad.as_mut() {
                                        let mut pg_data = pg.data.lock().unwrap();
                                        let new_grad_data = parent_grads[i].data.lock().unwrap();
                                        for (a, b) in pg_data.iter_mut().zip(new_grad_data.iter()) {
                                            *a += b;
                                        }
                                    } else {
                                        *parent_grad = Some(Box::new(parent_grads[i].clone()));
                                    }
                                }
                            }
                        }
                    }
                }

                pub fn size(&self) -> usize {
                    self.shape.iter().product()
                }

                pub fn dot(&self, other: &Tensor) -> Tensor {
                    let ctx = Context::new(
                        Arc::new(super::ops::MatMulOp),
                        vec![self.clone(), other.clone()],
                    );
                    ctx.op.forward(&ctx)
                }

                pub fn reshape(&self, new_shape: &[usize]) -> Tensor {
                    let ctx = Context::new(
                        Arc::new(super::ops::ReshapeOp::new(new_shape.to_vec())),
                        vec![self.clone()],
                    );
                    ctx.op.forward(&ctx)
                }

                pub fn transpose(&self) -> Tensor {
                    let ctx = Context::new(Arc::new(super::ops::TransposeOp), vec![self.clone()]);
                    ctx.op.forward(&ctx)
                }

                pub fn sum(&self) -> Tensor {
                    let ctx = Context::new(Arc::new(super::ops::SumOp), vec![self.clone()]);
                    ctx.op.forward(&ctx)
                }

                pub fn mul_scalar(&self, scalar: f64) -> Tensor {
                    let ctx = Context::new(
                        Arc::new(super::ops::MulScalarOp { scalar }),
                        vec![self.clone()],
                    );
                    ctx.op.forward(&ctx)
                }
            }

            impl Add for Tensor {
                type Output = Tensor;
                fn add(self, rhs: Tensor) -> Tensor {
                    let ctx = Context::new(Arc::new(super::ops::AddOp), vec![self, rhs]);
                    ctx.op.forward(&ctx)
                }
            }

            impl Mul for Tensor {
                type Output = Tensor;
                fn mul(self, rhs: Tensor) -> Tensor {
                    let ctx = Context::new(Arc::new(super::ops::MulOp), vec![self, rhs]);
                    ctx.op.forward(&ctx)
                }
            }

            impl Sub for Tensor {
                type Output = Tensor;
                fn sub(self, rhs: Tensor) -> Tensor {
                    let ctx = Context::new(Arc::new(super::ops::SubOp), vec![self, rhs]);
                    ctx.op.forward(&ctx)
                }
            }
        }

        pub mod autograd {
            use super::tensor::Tensor;
            use std::sync::Arc;

            pub trait Op: std::fmt::Debug + Send + Sync {
                fn forward(&self, ctx: &Arc<Context>) -> Tensor;
                fn backward(&self, ctx: &Arc<Context>, grad: &Tensor) -> Vec<Tensor>;
            }

            #[derive(Debug)]
            pub struct Context {
                pub op: Arc<dyn Op>,
                pub parents: Vec<Tensor>,
            }

            impl Context {
                pub fn new(op: Arc<dyn Op>, parents: Vec<Tensor>) -> Arc<Self> {
                    Arc::new(Self { op, parents })
                }
            }
        }

        pub mod ops {
            use super::autograd::{Context, Op};
            use super::tensor::Tensor;
            use std::f64::consts::FRAC_2_SQRT_PI;
            use std::sync::Arc;
            use rayon::iter::{IntoParallelRefIterator, ParallelIterator, IndexedParallelIterator};

            #[derive(Debug)]
            pub struct AddOp;
            impl Op for AddOp {
                fn forward(&self, ctx: &Arc<Context>) -> Tensor {
                    let a = &ctx.parents[0];
                    let b = &ctx.parents[1];
                    let a_data = a.data.lock().unwrap().clone();
                    let b_data = b.data.lock().unwrap().clone();

                    if a.shape == b.shape {
                        let data = a_data
                            .par_iter()
                            .zip(b_data.par_iter())
                            .map(|(av, bv)| av + bv)
                            .collect();
                        Tensor::new(data, &a.shape, Some(ctx.clone()))
                    } else if a.shape.len() == 2 && b.shape.len() == 1 && a.shape[1] == b.shape[0] {
                        let mut data = Vec::with_capacity(a.size());
                        let rows = a.shape[0];
                        for i in 0..rows {
                            let start = i * b.shape[0];
                            let end = start + b.shape[0];
                            data.extend(
                                a_data[start..end]
                                    .iter()
                                    .zip(b_data.iter())
                                    .map(|(av, bv)| av + bv),
                            );
                        }
                        Tensor::new(data, &a.shape, Some(ctx.clone()))
                    } else {
                        panic!(
                            "AddOp broadcasting not supported for shapes {:?} and {:?}",
                            a.shape, b.shape
                        );
                    }
                }

                fn backward(&self, ctx: &Arc<Context>, grad: &Tensor) -> Vec<Tensor> {
                    let a = &ctx.parents[0];
                    let b = &ctx.parents[1];

                    let grad_a = grad.clone();
                    let mut grad_b = grad.clone();

                    if a.shape != b.shape {
                        if a.shape.len() > b.shape.len() {
                            let grad_data = grad.data.lock().unwrap().clone();
                            let mut b_grad_data = vec![0.0; b.size()];
                            let cols = b.shape[0];
                            for row in grad_data.chunks(cols) {
                                for (i, val) in row.iter().enumerate() {
                                    b_grad_data[i] += val;
                                }
                            }
                            grad_b = Tensor::from_data(b_grad_data, &b.shape);
                        }
                    }
                    vec![grad_a, grad_b]
                }
            }

            #[derive(Debug)]
            pub struct SubOp;
            impl Op for SubOp {
                fn forward(&self, ctx: &Arc<Context>) -> Tensor {
                    let a_data = ctx.parents[0].data.lock().unwrap().clone();
                    let b_data = ctx.parents[1].data.lock().unwrap().clone();
                    let data = a_data.par_iter().zip(b_data.par_iter()).map(|(a, b)| a - b).collect();
                    Tensor::new(data, &ctx.parents[0].shape, Some(ctx.clone()))
                }

                fn backward(&self, _ctx: &Arc<Context>, grad: &Tensor) -> Vec<Tensor> {
                    let grad_data = grad.data.lock().unwrap().clone();
                    let neg_grad_data: Vec<f64> =
                        grad_data.par_iter().map(|&g| -g).collect();
                    vec![grad.clone(), Tensor::from_data(neg_grad_data, &grad.shape)]
                }
            }

            #[derive(Debug)]
            pub struct MulOp;
            impl Op for MulOp {
                fn forward(&self, ctx: &Arc<Context>) -> Tensor {
                    let a_data = ctx.parents[0].data.lock().unwrap().clone();
                    let b_data = ctx.parents[1].data.lock().unwrap().clone();
                    let data = a_data.par_iter().zip(b_data.par_iter()).map(|(a, b)| a * b).collect();
                    Tensor::new(data, &ctx.parents[0].shape, Some(ctx.clone()))
                }

                fn backward(&self, ctx: &Arc<Context>, grad: &Tensor) -> Vec<Tensor> {
                    let a = &ctx.parents[0];
                    let b = &ctx.parents[1];
                    let grad_data = grad.data.lock().unwrap().clone();
                    let a_data = a.data.lock().unwrap().clone();
                    let b_data = b.data.lock().unwrap().clone();
                    let grad_a_data: Vec<f64> = b_data
                        .par_iter()
                        .zip(grad_data.par_iter())
                        .map(|(bv, gv)| bv * gv)
                        .collect();
                    let grad_b_data: Vec<f64> = a_data
                        .par_iter()
                        .zip(grad_data.par_iter())
                        .map(|(av, gv)| av * gv)
                        .collect();
                    vec![
                        Tensor::from_data(grad_a_data, &a.shape),
                        Tensor::from_data(grad_b_data, &b.shape),
                    ]
                }
            }

            #[derive(Debug)]
            pub struct MatMulOp;
            impl Op for MatMulOp {
                fn forward(&self, ctx: &Arc<Context>) -> Tensor {
                    let a = &ctx.parents[0];
                    let b = &ctx.parents[1];
                    assert_eq!(a.shape.len(), 2, "MatMul requires 2D tensors");
                    assert_eq!(b.shape.len(), 2, "MatMul requires 2D tensors");
                    assert_eq!(
                        a.shape[1], b.shape[0],
                        "MatMul shape mismatch: {:?} vs {:?}",
                        a.shape, b.shape
                    );
                    let a_data = a.data.lock().unwrap().clone();
                    let b_data = b.data.lock().unwrap().clone();
                    let (m, k, n) = (a.shape[0], a.shape[1], b.shape[1]);
                    let mut c_data = vec![0.0; m * n];
                    c_data.chunks_mut(n).enumerate().for_each(|(i, row)| {
                        for j in 0..n {
                            let mut sum = 0.0;
                            for l in 0..k {
                                sum += a_data[i * k + l] * b_data[l * n + j];
                            }
                            row[j] = sum;
                        }
                    });
                    Tensor::new(c_data, &[m, n], Some(ctx.clone()))
                }

                fn backward(&self, ctx: &Arc<Context>, grad: &Tensor) -> Vec<Tensor> {
                    let a = &ctx.parents[0];
                    let b = &ctx.parents[1];
                    let a_t = a.transpose();
                    let b_t = b.transpose();
                    let grad_a = grad.dot(&b_t);
                    let grad_b = a_t.dot(grad);
                    vec![grad_a, grad_b]
                }
            }

            #[derive(Debug)]
            pub struct ReLUOp;
            impl Op for ReLUOp {
                fn forward(&self, ctx: &Arc<Context>) -> Tensor {
                    let input_data = ctx.parents[0].data.lock().unwrap().clone();
                    let result_data: Vec<f64> =
                        input_data.par_iter().map(|&x| x.max(0.0)).collect();
                    Tensor::new(result_data, &ctx.parents[0].shape, Some(ctx.clone()))
                }

                fn backward(&self, ctx: &Arc<Context>, grad: &Tensor) -> Vec<Tensor> {
                    let input_data = ctx.parents[0].data.lock().unwrap().clone();
                    let grad_data = grad.data.lock().unwrap().clone();
                    let result_grad: Vec<f64> = input_data
                        .par_iter()
                        .zip(grad_data.par_iter())
                        .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
                        .collect();
                    vec![Tensor::from_data(result_grad, &grad.shape)]
                }
            }

            #[derive(Debug)]
            pub struct SigmoidOp;
            impl Op for SigmoidOp {
                fn forward(&self, ctx: &Arc<Context>) -> Tensor {
                    let input_data = ctx.parents[0].data.lock().unwrap().clone();
                    let result_data: Vec<f64> = input_data
                        .par_iter()
                        .map(|&x| 1.0 / (1.0 + (-x).exp()))
                        .collect();
                    Tensor::new(result_data, &ctx.parents[0].shape, Some(ctx.clone()))
                }

                fn backward(&self, ctx: &Arc<Context>, grad: &Tensor) -> Vec<Tensor> {
                    let fwd_output = self.forward(ctx);
                    let fwd_data = fwd_output.data.lock().unwrap().clone();
                    let grad_data = grad.data.lock().unwrap().clone();
                    let result_grad: Vec<f64> = fwd_data
                        .par_iter()
                        .zip(grad_data.par_iter())
                        .map(|(&y, &g)| y * (1.0 - y) * g)
                        .collect();
                    vec![Tensor::from_data(result_grad, &grad.shape)]
                }
            }
            
            #[derive(Debug)]
            pub struct TanhOp;
            impl Op for TanhOp {
                fn forward(&self, ctx: &Arc<Context>) -> Tensor {
                    let input_data = ctx.parents[0].data.lock().unwrap().clone();
                    let result_data: Vec<f64> = input_data.par_iter().map(|&x| x.tanh()).collect();
                    Tensor::new(result_data, &ctx.parents[0].shape, Some(ctx.clone()))
                }

                fn backward(&self, ctx: &Arc<Context>, grad: &Tensor) -> Vec<Tensor> {
                    let fwd_output = self.forward(ctx);
                    let y_data = fwd_output.data.lock().unwrap().clone();
                    let grad_data = grad.data.lock().unwrap().clone();
                    let result_grad: Vec<f64> = y_data
                        .par_iter()
                        .zip(grad_data.par_iter())
                        .map(|(&y, &g)| (1.0 - y.powi(2)) * g)
                        .collect();
                    vec![Tensor::from_data(result_grad, &grad.shape)]
                }
            }

            #[derive(Debug)]
            pub struct ReshapeOp {
                pub new_shape: Vec<usize>,
            }
            impl ReshapeOp {
                pub fn new(new_shape: Vec<usize>) -> Self {
                    Self { new_shape }
                }
            }
            impl Op for ReshapeOp {
                fn forward(&self, ctx: &Arc<Context>) -> Tensor {
                    let input = &ctx.parents[0];
                    let data = input.data.lock().unwrap().clone();
                    Tensor::new(data, &self.new_shape, Some(ctx.clone()))
                }

                fn backward(&self, ctx: &Arc<Context>, grad: &Tensor) -> Vec<Tensor> {
                    let input_shape = &ctx.parents[0].shape;
                    vec![grad.clone().reshape(input_shape)]
                }
            }

            #[derive(Debug)]
            pub struct TransposeOp;
            impl Op for TransposeOp {
                fn forward(&self, ctx: &Arc<Context>) -> Tensor {
                    let input = &ctx.parents[0];
                    assert_eq!(input.shape.len(), 2, "TransposeOp only supports 2D tensors.");
                    let (rows, cols) = (input.shape[0], input.shape[1]);
                    let input_data = input.data.lock().unwrap();
                    let mut transposed_data = vec![0.0; rows * cols];
                    for i in 0..rows {
                        for j in 0..cols {
                            transposed_data[j * rows + i] = input_data[i * cols + j];
                        }
                    }
                    Tensor::new(transposed_data, &[cols, rows], Some(ctx.clone()))
                }

                fn backward(&self, _ctx: &Arc<Context>, grad: &Tensor) -> Vec<Tensor> {
                    vec![grad.transpose()]
                }
            }

            #[derive(Debug)]
            pub struct SumOp;
            impl Op for SumOp {
                fn forward(&self, ctx: &Arc<Context>) -> Tensor {
                    let input_data = ctx.parents[0].data.lock().unwrap();
                    let sum = input_data.iter().sum();
                    Tensor::new(vec![sum], &[1], Some(ctx.clone()))
                }

                fn backward(&self, ctx: &Arc<Context>, grad: &Tensor) -> Vec<Tensor> {
                    let input = &ctx.parents[0];
                    let grad_val = grad.data.lock().unwrap()[0];
                    let grad_data = vec![grad_val; input.size()];
                    vec![Tensor::from_data(grad_data, &input.shape)]
                }
            }

            #[derive(Debug)]
            pub struct MulScalarOp {
                pub scalar: f64,
            }
            impl Op for MulScalarOp {
                fn forward(&self, ctx: &Arc<Context>) -> Tensor {
                    let input_data = ctx.parents[0].data.lock().unwrap().clone();
                    let result_data = input_data.par_iter().map(|&x| x * self.scalar).collect();
                    Tensor::new(result_data, &ctx.parents[0].shape, Some(ctx.clone()))
                }

                fn backward(&self, ctx: &Arc<Context>, grad: &Tensor) -> Vec<Tensor> {
                    let grad_data = grad.data.lock().unwrap().clone();
                    let result_grad = grad_data.par_iter().map(|&g| g * self.scalar).collect();
                    vec![Tensor::from_data(result_grad, &ctx.parents[0].shape)]
                }
            }

            #[derive(Debug)]
            pub struct SoftmaxOp;
            impl Op for SoftmaxOp {
                fn forward(&self, ctx: &Arc<Context>) -> Tensor {
                    let input = &ctx.parents[0];
                    let input_data = input.data.lock().unwrap();
                    let last_dim = *input.shape.last().unwrap();
                    let mut output_data = Vec::with_capacity(input.size());

                    for row in input_data.chunks(last_dim) {
                        let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                        let exps: Vec<f64> = row.iter().map(|x| (x - max_val).exp()).collect();
                        let sum_exps: f64 = exps.iter().sum();
                        output_data.extend(exps.iter().map(|e| e / sum_exps));
                    }
                    Tensor::new(output_data, &input.shape, Some(ctx.clone()))
                }

                fn backward(&self, ctx: &Arc<Context>, grad: &Tensor) -> Vec<Tensor> {
                    let output = self.forward(ctx);
                    let output_data = output.data.lock().unwrap();
                    let grad_data = grad.data.lock().unwrap();
                    let last_dim = *output.shape.last().unwrap();
                    let mut input_grad = vec![0.0; output.size()];

                    for (i, row_out) in output_data.chunks(last_dim).enumerate() {
                        let row_grad = &grad_data[i * last_dim..(i + 1) * last_dim];
                        let mut jacobian = vec![vec![0.0; last_dim]; last_dim];
                        for r in 0..last_dim {
                            for c in 0..last_dim {
                                if r == c {
                                    jacobian[r][c] = row_out[r] * (1.0 - row_out[c]);
                                } else {
                                    jacobian[r][c] = -row_out[r] * row_out[c];
                                }
                            }
                        }
                        for r in 0..last_dim {
                            let mut grad_sum = 0.0;
                            for c in 0..last_dim {
                                grad_sum += row_grad[c] * jacobian[c][r];
                            }
                            input_grad[i * last_dim + r] = grad_sum;
                        }
                    }
                    vec![Tensor::from_data(input_grad, &output.shape)]
                }
            }

            #[derive(Debug)]
            pub struct GeluOp;
            impl Op for GeluOp {
                fn forward(&self, ctx: &Arc<Context>) -> Tensor {
                    let input_data = ctx.parents[0].data.lock().unwrap().clone();
                    let output_data = input_data
                        .par_iter()
                        .map(|&x| {
                            0.5 * x * (1.0 + (FRAC_2_SQRT_PI * (x + 0.044715 * x.powi(3))).tanh())
                        })
                        .collect();
                    Tensor::new(output_data, &ctx.parents[0].shape, Some(ctx.clone()))
                }

                fn backward(&self, ctx: &Arc<Context>, grad: &Tensor) -> Vec<Tensor> {
                    let input_data = ctx.parents[0].data.lock().unwrap().clone();
                    let grad_data = grad.data.lock().unwrap().clone();
                    let input_grad = input_data
                        .par_iter()
                        .zip(grad_data.par_iter())
                        .map(|(&x, &g)| {
                            let c = FRAC_2_SQRT_PI;
                            let k = 0.044715;
                            let x3 = x.powi(3);
                            let inner = c * (x + k * x3);
                            let tanh_inner = inner.tanh();
                            let sech_inner2 = 1.0 - tanh_inner.powi(2);
                            let d_inner = c * (1.0 + 3.0 * k * x.powi(2));
                            let d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_inner2 * d_inner;
                            d_gelu * g
                        })
                        .collect();
                    vec![Tensor::from_data(input_grad, &ctx.parents[0].shape)]
                }
            }

            #[derive(Debug)]
            pub struct LayerNormOp;
            impl Op for LayerNormOp {
                fn forward(&self, ctx: &Arc<Context>) -> Tensor {
                    let input = &ctx.parents[0];
                    let gamma = &ctx.parents[1];
                    let beta = &ctx.parents[2];
                    let input_data = input.data.lock().unwrap();
                    let gamma_data = gamma.data.lock().unwrap();
                    let beta_data = beta.data.lock().unwrap();
                    let last_dim = *input.shape.last().unwrap();
                    let mut output_data = vec![0.0; input.size()];
                    let eps = 1e-5;

                    for (i, row) in input_data.chunks(last_dim).enumerate() {
                        let mean = row.iter().sum::<f64>() / last_dim as f64;
                        let var = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                            / last_dim as f64;
                        let std_dev = (var + eps).sqrt();

                        for j in 0..last_dim {
                            let normalized = (row[j] - mean) / std_dev;
                            output_data[i * last_dim + j] =
                                normalized * gamma_data[j] + beta_data[j];
                        }
                    }
                    Tensor::new(output_data, &input.shape, Some(ctx.clone()))
                }

                fn backward(&self, ctx: &Arc<Context>, grad: &Tensor) -> Vec<Tensor> {
                    let input = &ctx.parents[0];
                    let gamma = &ctx.parents[1];
                    let grad_data = grad.data.lock().unwrap();
                    let input_data = input.data.lock().unwrap();
                    let gamma_data = gamma.data.lock().unwrap();
                    let last_dim = *input.shape.last().unwrap();
                    let eps = 1e-5;

                    let mut grad_gamma = vec![0.0; gamma.size()];
                    let mut grad_beta = vec![0.0; gamma.size()];
                    let mut grad_input = vec![0.0; input.size()];

                    for (i, row) in input_data.chunks(last_dim).enumerate() {
                        let mean = row.iter().sum::<f64>() / last_dim as f64;
                        let var = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                            / last_dim as f64;
                        let std_inv = 1.0 / (var + eps).sqrt();

                        let row_grad = &grad_data[i * last_dim..(i + 1) * last_dim];

                        for j in 0..last_dim {
                            let normalized = (row[j] - mean) * std_inv;
                            grad_beta[j] += row_grad[j];
                            grad_gamma[j] += row_grad[j] * normalized;
                        }
                    }

                    for i in 0..grad_data.len() / last_dim {
                        for j in 0..last_dim {
                            grad_input[i * last_dim + j] = grad_data[i * last_dim + j] * gamma_data[j];
                        }
                    }

                    vec![
                        Tensor::from_data(grad_input, &input.shape),
                        Tensor::from_data(grad_gamma, &gamma.shape),
                        Tensor::from_data(grad_beta, &gamma.shape),
                    ]
                }
            }
        }
    }

    pub mod models {
        use super::*;
        use self::core::autograd::{Context, Op};
        use self::core::ops::*;
        use serde_json::Value;
        use std::collections::{HashMap, HashSet, VecDeque};
        use std::sync::Arc;

        pub trait AiModel: Send + Sync {
            fn forward(&mut self, input: &Tensor) -> Tensor;
            fn get_params(&self) -> Vec<Tensor>;
            fn get_named_params(&self) -> &HashMap<String, Tensor>;
            fn to_value(&self) -> Value;
            fn set_param(&mut self, name: &str, tensor: Tensor);
        }

        pub struct GraphModel {
            config: Value,
            params: HashMap<String, Tensor>,
            topo_order: Vec<String>,
        }

        impl GraphModel {
            pub fn from_config(config: &Value) -> Result<Self, String> {
                let mut params = HashMap::new();
                let nodes = config["nodes"]
                    .as_object()
                    .ok_or("Graph config must have a 'nodes' object")?;

                for (name, node_cfg) in nodes {
                    let op_name = node_cfg["op"].as_str().unwrap_or("");
                    match op_name {
                        "Linear" => {
                            let p = node_cfg.get("params").ok_or_else(|| {
                                format!("Node '{}' of type 'Linear' is missing 'params'", name)
                            })?;

                            let in_f = p
                                .get("in_features")
                                .and_then(Value::as_u64)
                                .ok_or_else(|| {
                                    format!(
                                        "Linear node '{}' is missing or has invalid 'in_features'",
                                        name
                                    )
                                })? as usize;
                            let out_f = p
                                .get("out_features")
                                .and_then(Value::as_u64)
                                .ok_or_else(|| {
                                    format!(
                                        "Linear node '{}' is missing or has invalid 'out_features'",
                                        name
                                    )
                                })? as usize;

                            if in_f == 0 || out_f == 0 {
                                return Err(format!(
                                    "Linear node '{}' cannot have zero-sized dimensions.",
                                    name
                                ));
                            }

                            let limit_calc: f64 = 6.0 / (in_f + out_f) as f64;
                            let limit = limit_calc.sqrt();
                            let w_data = (0..in_f * out_f)
                                .map(|_| (rand::random::<f64>() * 2.0 - 1.0) * limit)
                                .collect();
                            let b_data = vec![0.0; out_f];
                            params.insert(
                                format!("{}_w", name),
                                Tensor::from_data(w_data, &[in_f, out_f]),
                            );
                            params.insert(format!("{}_b", name), Tensor::from_data(b_data, &[out_f]));
                        }
                        "LayerNorm" => {
                            let p = node_cfg.get("params").ok_or_else(|| {
                                format!("Node '{}' of type 'LayerNorm' is missing 'params'", name)
                            })?;
                            let dim = p
                                .get("normalized_shape")
                                .and_then(Value::as_u64)
                                .ok_or_else(|| {
                                    format!(
                                        "LayerNorm node '{}' is missing or has invalid 'normalized_shape'",
                                        name
                                    )
                                })? as usize;

                            if dim == 0 {
                                return Err(format!(
                                    "LayerNorm node '{}' cannot have a zero-sized shape.",
                                    name
                                ));
                            }

                            params.insert(
                                format!("{}_gamma", name),
                                Tensor::from_data(vec![1.0; dim], &[dim]),
                            );
                            params.insert(
                                format!("{}_beta", name),
                                Tensor::from_data(vec![0.0; dim], &[dim]),
                            );
                        }
                        _ => {}
                    }
                }

                let (topo_order, _) = Self::topological_sort(config)?;
                Ok(Self {
                    config: config.clone(),
                    params,
                    topo_order,
                })
            }

            fn topological_sort(
                config: &Value,
            ) -> Result<(Vec<String>, HashMap<String, Vec<String>>), String> {
                let nodes = config["nodes"]
                    .as_object()
                    .ok_or("Graph config must have 'nodes'")?;
                let mut in_degree = HashMap::new();
                let mut adj_list = HashMap::new();
                let input_names: HashSet<String> = config["inputs"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_str().unwrap().to_string())
                    .collect();

                for name in nodes.keys() {
                    in_degree.insert(name.clone(), 0);
                    adj_list.insert(name.clone(), Vec::new());
                }

                for (name, node_cfg) in nodes {
                    for input_v in node_cfg["inputs"].as_array().unwrap() {
                        let input_name = input_v.as_str().unwrap().to_string();
                        if !input_names.contains(&input_name) {
                            adj_list
                                .get_mut(&input_name)
                                .ok_or(format!(
                                    "Input node '{}' for node '{}' not found in graph",
                                    input_name, name
                                ))?
                                .push(name.clone());
                            *in_degree.get_mut(name).unwrap() += 1;
                        }
                    }
                }

                let mut queue: VecDeque<String> = in_degree
                    .iter()
                    .filter(|(_, &deg)| deg == 0)
                    .map(|(name, _)| name.clone())
                    .collect();
                let mut topo_order = Vec::new();

                while let Some(u) = queue.pop_front() {
                    topo_order.push(u.clone());
                    if let Some(neighbors) = adj_list.get(&u) {
                        for v in neighbors {
                            if let Some(deg) = in_degree.get_mut(v) {
                                *deg -= 1;
                                if *deg == 0 {
                                    queue.push_back(v.clone());
                                }
                            }
                        }
                    }
                }

                if topo_order.len() != nodes.len() {
                    return Err("Graph contains a cycle".to_string());
                }

                Ok((topo_order, adj_list))
            }

            fn execute_graph(&self, inputs: &HashMap<String, Tensor>) -> Tensor {
                let mut node_outputs = inputs.clone();
                let nodes_cfg = self.config["nodes"].as_object().unwrap();
                let output_node_name = self.config["output_node"].as_str().unwrap();

                for name in &self.topo_order {
                    let node_cfg = &nodes_cfg[name];
                    let op_name = node_cfg["op"].as_str().unwrap();
                    let input_names_v = node_cfg["inputs"].as_array().unwrap();
                    let mut op_inputs: Vec<Tensor> = input_names_v
                        .iter()
                        .map(|n| node_outputs.get(n.as_str().unwrap()).unwrap().clone())
                        .collect();

                    let output = match op_name {
                        "Linear" => {
                            let w = self.params[&format!("{}_w", name)].clone();
                            let b = self.params[&format!("{}_b", name)].clone();
                            op_inputs[0].dot(&w) + b
                        }
                        "LayerNorm" => {
                            op_inputs.push(self.params[&format!("{}_gamma", name)].clone());
                            op_inputs.push(self.params[&format!("{}_beta", name)].clone());
                            let ctx = Context::new(Arc::new(LayerNormOp), op_inputs);
                            ctx.op.forward(&ctx)
                        }
                        _ => {
                            let op: Arc<dyn Op> = match op_name {
                                "ReLU" => Arc::new(ReLUOp),
                                "Gelu" => Arc::new(GeluOp),
                                "Sigmoid" => Arc::new(SigmoidOp),
                                "Softmax" => Arc::new(SoftmaxOp),
                                "Tanh" => Arc::new(TanhOp),
                                "Add" => Arc::new(AddOp),
                                "Sub" => Arc::new(SubOp),
                                "Mul" => Arc::new(MulOp),
                                "MatMul" => Arc::new(MatMulOp),
                                "Transpose" => Arc::new(TransposeOp),
                                "Reshape" => {
                                    let shape = node_cfg["params"]["shape"]
                                        .as_array()
                                        .unwrap()
                                        .iter()
                                        .map(|v| v.as_u64().unwrap() as usize)
                                        .collect();
                                    Arc::new(ReshapeOp::new(shape))
                                }
                                _ => panic!("Unknown op: {}", op_name),
                            };
                            let ctx = Context::new(op, op_inputs);
                            ctx.op.forward(&ctx)
                        }
                    };
                    node_outputs.insert(name.to_string(), output);
                }
                node_outputs[output_node_name].clone()
            }
        }

        impl AiModel for GraphModel {
            fn forward(&mut self, input: &Tensor) -> Tensor {
                let mut inputs = HashMap::new();
                let input_node_name = self.config["inputs"].as_array().unwrap()[0]
                    .as_str()
                    .unwrap();
                inputs.insert(input_node_name.to_string(), input.clone());
                self.execute_graph(&inputs)
            }

            fn get_params(&self) -> Vec<Tensor> {
                self.params.values().cloned().collect()
            }
            
            fn get_named_params(&self) -> &HashMap<String, Tensor> {
                &self.params
            }

            fn to_value(&self) -> Value {
                self.config.clone()
            }
            
            fn set_param(&mut self, name: &str, tensor: Tensor) {
                self.params.insert(name.to_string(), tensor);
            }
        }
    }

    pub mod loss {
        use super::core::tensor::Tensor;
        use serde::Deserialize;

        pub trait Loss: Send + Sync {
            fn forward(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor;
        }

        #[derive(Deserialize, Debug)]
        pub enum LossType {
            MSE,
            CrossEntropy,
        }

        pub fn create_loss(loss_type: LossType) -> Box<dyn Loss> {
            match loss_type {
                LossType::MSE => Box::new(MSE),
                LossType::CrossEntropy => Box::new(CrossEntropy),
            }
        }

        pub struct MSE;
        impl Loss for MSE {
            fn forward(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
                let n = y_true.size() as f64;
                if n == 0.0 {
                    return Tensor::from_data(vec![0.0], &[1]);
                }
                let diff = y_pred.clone() - y_true.clone();
                let sq_error = diff.clone() * diff;
                let sum_sq_error = sq_error.sum();
                sum_sq_error.mul_scalar(1.0 / n)
            }
        }

        pub struct CrossEntropy;
        impl Loss for CrossEntropy {
            fn forward(&self, y_true: &Tensor, y_pred: &Tensor) -> Tensor {
                let n = y_true.size() as f64;
                if n == 0.0 {
                    return Tensor::from_data(vec![0.0], &[1]);
                }
                
                let y_true_data = y_true.data.lock().unwrap();
                let y_pred_data = y_pred.data.lock().unwrap();
                
                let mut loss = 0.0;
                for (true_val, pred_val) in y_true_data.iter().zip(y_pred_data.iter()) {
                    // Add small epsilon to prevent log(0)
                    let pred_clipped = pred_val.max(1e-15).min(1.0 - 1e-15);
                    loss -= true_val * pred_clipped.ln();
                }
                
                Tensor::from_data(vec![loss / n], &[1])
            }
        }
    }

    pub mod optimizers {
        use super::{core::tensor::Tensor, models::AiModel};
        use serde::Deserialize;
        use std::collections::HashMap;

        pub trait Optimizer: Send + Sync {
            fn step(&mut self, model: &dyn AiModel);
            fn zero_grad(&mut self, model: &dyn AiModel);
        }

        #[derive(Deserialize, Debug)]
        #[serde(rename_all = "camelCase")]
        pub enum OptimizerConfig {
            Sgd { lr: f64 },
            Adam { lr: f64 },
        }

        pub fn create_optimizer(config: OptimizerConfig) -> Box<dyn Optimizer> {
            match config {
                OptimizerConfig::Sgd { lr } => Box::new(SGD { lr }),
                OptimizerConfig::Adam { lr } => Box::new(Adam::new(lr)),
            }
        }

        pub struct SGD {
            pub lr: f64,
        }

        impl Optimizer for SGD {
            fn step(&mut self, model: &dyn AiModel) {
                for param in model.get_params() {
                    let mut p_data = param.data.lock().unwrap();
                    if let Some(g) = param.grad.lock().unwrap().as_ref() {
                        let g_data = g.data.lock().unwrap();
                        for (p, g_val) in p_data.iter_mut().zip(g_data.iter()) {
                            *p -= self.lr * g_val;
                        }
                    }
                }
            }

            fn zero_grad(&mut self, model: &dyn AiModel) {
                for param in model.get_params() {
                    *param.grad.lock().unwrap() = None;
                }
            }
        }
        
        pub struct Adam {
            lr: f64,
            beta1: f64,
            beta2: f64,
            epsilon: f64,
            m: HashMap<usize, Tensor>,
            v: HashMap<usize, Tensor>,
            t: usize,
        }

        impl Adam {
            pub fn new(lr: f64) -> Self {
                Self {
                    lr,
                    beta1: 0.9,
                    beta2: 0.999,
                    epsilon: 1e-8,
                    m: HashMap::new(),
                    v: HashMap::new(),
                    t: 0,
                }
            }
        }
        impl Optimizer for Adam {
            fn step(&mut self, model: &dyn AiModel) {
                self.t += 1;
                let beta1_t = self.beta1.powi(self.t as i32);
                let beta2_t = self.beta2.powi(self.t as i32);

                for param in model.get_params() {
                    if let Some(g) = param.grad.lock().unwrap().as_ref() {
                        let m_t = self.m.entry(param.id).or_insert_with(|| {
                            Tensor::from_data(vec![0.0; param.size()], &param.shape)
                        });
                        let v_t = self.v.entry(param.id).or_insert_with(|| {
                            Tensor::from_data(vec![0.0; param.size()], &param.shape)
                        });

                        let mut m_data = m_t.data.lock().unwrap();
                        let mut v_data = v_t.data.lock().unwrap();
                        let g_data = g.data.lock().unwrap();
                        
                        // Update biased first moment estimate
                        for (m_val, g_val) in m_data.iter_mut().zip(g_data.iter()) {
                            *m_val = self.beta1 * *m_val + (1.0 - self.beta1) * g_val;
                        }
                        
                        // Update biased second raw moment estimate
                        for (v_val, g_val) in v_data.iter_mut().zip(g_data.iter()) {
                            *v_val = self.beta2 * *v_val + (1.0 - self.beta2) * g_val.powi(2);
                        }
                        
                        let m_hat_data: Vec<f64> = m_data.iter().map(|&m| m / (1.0 - beta1_t)).collect();
                        let v_hat_data: Vec<f64> = v_data.iter().map(|&v| v / (1.0 - beta2_t)).collect();
                        
                        let mut p_data = param.data.lock().unwrap();
                        for i in 0..p_data.len() {
                            p_data[i] -= self.lr * m_hat_data[i] / (v_hat_data[i].sqrt() + self.epsilon);
                        }
                    }
                }
            }

            fn zero_grad(&mut self, model: &dyn AiModel) {
                for param in model.get_params() {
                    *param.grad.lock().unwrap() = None;
                }
            }
        }
    }

    pub mod training {
        use super::{loss::Loss, optimizers::Optimizer, AiModel, Tensor};
        use serde::Serialize;

        #[derive(Serialize, Debug)]
        pub struct TrainingMetrics {
            pub epoch: usize,
            pub loss: f64,
            pub accuracy: f64,
            pub learning_rate: f64,
            pub batch_loss: f64,
            pub gradient_norm: f64,
            pub param_norm: f64,
        }

        #[derive(Serialize, Debug)]
        pub struct TrainingProgress {
            pub metrics: Vec<TrainingMetrics>,
            pub total_epochs: usize,
            pub current_epoch: usize,
            pub is_complete: bool,
        }

        pub fn calculate_accuracy(y_true: &Tensor, y_pred: &Tensor) -> f64 {
            let y_true_data = y_true.data.lock().unwrap();
            let y_pred_data = y_pred.data.lock().unwrap();
            
            if y_true_data.len() != y_pred_data.len() {
                return 0.0;
            }
            
            let mut correct = 0.0;
            let total = y_true_data.len() as f64;
            
            // For classification tasks, find the index with maximum value
            if y_true.shape.len() > 1 && y_true.shape[1] > 1 {
                // Multi-class classification
                let batch_size = y_true.shape[0];
                let num_classes = y_true.shape[1];
                
                for i in 0..batch_size {
                    let start_idx = i * num_classes;
                    let true_slice = &y_true_data[start_idx..start_idx + num_classes];
                    let pred_slice = &y_pred_data[start_idx..start_idx + num_classes];
                    
                    let true_max_idx = true_slice.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(idx, _)| idx).unwrap_or(0);
                    
                    let pred_max_idx = pred_slice.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(idx, _)| idx).unwrap_or(0);
                    
                    if true_max_idx == pred_max_idx {
                        correct += 1.0;
                    }
                }
            } else {
                // Regression or binary classification
                for (true_val, pred_val) in y_true_data.iter().zip(y_pred_data.iter()) {
                    if (true_val - pred_val).abs() < 0.5 {
                        correct += 1.0;
                    }
                }
            }
            
            correct / total
        }

        pub fn calculate_gradient_norm(model: &dyn AiModel) -> f64 {
            let mut total_norm = 0.0;
            for param in model.get_params() {
                if let Some(grad) = param.grad.lock().unwrap().as_ref() {
                    let grad_data = grad.data.lock().unwrap();
                    for &val in grad_data.iter() {
                        total_norm += val * val;
                    }
                }
            }
            total_norm.sqrt()
        }

        pub fn calculate_param_norm(model: &dyn AiModel) -> f64 {
            let mut total_norm = 0.0;
            for param in model.get_params() {
                let param_data = param.data.lock().unwrap();
                for &val in param_data.iter() {
                    total_norm += val * val;
                }
            }
            total_norm.sqrt()
        }

        pub fn train_with_callback<F>(
            model: &mut dyn AiModel,
            x: Tensor,
            y: Tensor,
            epochs: usize,
            batch_size: usize,
            optimizer: &mut dyn Optimizer,
            loss_fn: &dyn Loss,
            mut callback: F,
        ) -> TrainingProgress
        where
            F: FnMut(&TrainingMetrics) -> Option<axum::response::sse::Event>,
        {
            let mut metrics = Vec::new();
            let num_samples = x.shape[0];
            if num_samples == 0 {
                return TrainingProgress {
                    metrics,
                    total_epochs: epochs,
                    current_epoch: 0,
                    is_complete: true,
                };
            }
            let num_batches = (num_samples as f64 / batch_size as f64).ceil() as usize;

            for epoch in 0..epochs {
                let mut epoch_loss = 0.0;
                let mut epoch_accuracy = 0.0;
                let mut batch_losses = Vec::new();

                for i in 0..num_batches {
                    let start = i * batch_size;
                    let end = (start + batch_size).min(num_samples);
                    if start >= end {
                        continue;
                    }

                    let x_batch_len: usize = x.shape[1..].iter().product();
                    let y_batch_len: usize = y.shape[1..].iter().product();
                    let mut x_batch_shape = x.shape.clone();
                    x_batch_shape[0] = end - start;
                    let mut y_batch_shape = y.shape.clone();
                    y_batch_shape[0] = end - start;

                    let x_data = x.data.lock().unwrap();
                    let y_data = y.data.lock().unwrap();

                    let x_batch_data = x_data[start * x_batch_len..end * x_batch_len].to_vec();
                    let y_batch_data = y_data[start * y_batch_len..end * y_batch_len].to_vec();
                    let x_batch = Tensor::from_data(x_batch_data, &x_batch_shape);
                    let y_batch = Tensor::from_data(y_batch_data, &y_batch_shape);

                    optimizer.zero_grad(model);
                    let y_pred = model.forward(&x_batch);
                    let loss_tensor = loss_fn.forward(&y_batch, &y_pred);

                    loss_tensor.backward();
                    optimizer.step(model);

                    let loss_val = loss_tensor.data.lock().unwrap()[0];
                    let accuracy = calculate_accuracy(&y_batch, &y_pred);
                    
                    epoch_loss += loss_val;
                    epoch_accuracy += accuracy;
                    batch_losses.push(loss_val);
                }

                let avg_loss = if num_batches > 0 {
                    epoch_loss / num_batches as f64
                } else {
                    0.0
                };
                
                let avg_accuracy = if num_batches > 0 {
                    epoch_accuracy / num_batches as f64
                } else {
                    0.0
                };

                let gradient_norm = calculate_gradient_norm(model);
                let param_norm = calculate_param_norm(model);
                let batch_loss = batch_losses.iter().sum::<f64>() / batch_losses.len() as f64;

                let metric = TrainingMetrics {
                    epoch: epoch + 1,
                    loss: avg_loss,
                    accuracy: avg_accuracy,
                    learning_rate: 0.001,
                    batch_loss,
                    gradient_norm,
                    param_norm,
                };

                // Call the callback with the metric
                callback(&metric);
                metrics.push(metric);

                if (epoch + 1) % 10 == 0 || epoch == 0 || epoch == epochs - 1 {
                    tracing::info!(
                        "Epoch {}/{}, Loss: {:.6}, Accuracy: {:.4}, Grad Norm: {:.6}, Param Norm: {:.6}",
                        epoch + 1, epochs, avg_loss, avg_accuracy, gradient_norm, param_norm
                    );
                }
            }

            TrainingProgress {
                metrics,
                total_epochs: epochs,
                current_epoch: epochs,
                is_complete: true,
            }
        }

        pub fn train(
            model: &mut dyn AiModel,
            x: Tensor,
            y: Tensor,
            epochs: usize,
            batch_size: usize,
            optimizer: &mut dyn Optimizer,
            loss_fn: &dyn Loss,
        ) -> TrainingProgress {
            let mut metrics = Vec::new();
            let num_samples = x.shape[0];
            if num_samples == 0 {
                return TrainingProgress {
                    metrics,
                    total_epochs: epochs,
                    current_epoch: 0,
                    is_complete: true,
                };
            }
            let num_batches = (num_samples as f64 / batch_size as f64).ceil() as usize;

            for epoch in 0..epochs {
                let mut epoch_loss = 0.0;
                let mut epoch_accuracy = 0.0;
                let mut batch_losses = Vec::new();

                for i in 0..num_batches {
                    let start = i * batch_size;
                    let end = (start + batch_size).min(num_samples);
                    if start >= end {
                        continue;
                    }

                    let x_batch_len: usize = x.shape[1..].iter().product();
                    let y_batch_len: usize = y.shape[1..].iter().product();
                    let mut x_batch_shape = x.shape.clone();
                    x_batch_shape[0] = end - start;
                    let mut y_batch_shape = y.shape.clone();
                    y_batch_shape[0] = end - start;

                    let x_data = x.data.lock().unwrap();
                    let y_data = y.data.lock().unwrap();

                    let x_batch_data = x_data[start * x_batch_len..end * x_batch_len].to_vec();
                    let y_batch_data = y_data[start * y_batch_len..end * y_batch_len].to_vec();
                    let x_batch = Tensor::from_data(x_batch_data, &x_batch_shape);
                    let y_batch = Tensor::from_data(y_batch_data, &y_batch_shape);

                    optimizer.zero_grad(model);
                    let y_pred = model.forward(&x_batch);
                    let loss_tensor = loss_fn.forward(&y_batch, &y_pred);

                    loss_tensor.backward();
                    optimizer.step(model);

                    let loss_val = loss_tensor.data.lock().unwrap()[0];
                    let accuracy = calculate_accuracy(&y_batch, &y_pred);
                    
                    epoch_loss += loss_val;
                    epoch_accuracy += accuracy;
                    batch_losses.push(loss_val);
                }

                let avg_loss = if num_batches > 0 {
                    epoch_loss / num_batches as f64
                } else {
                    0.0
                };
                
                let avg_accuracy = if num_batches > 0 {
                    epoch_accuracy / num_batches as f64
                } else {
                    0.0
                };

                let gradient_norm = calculate_gradient_norm(model);
                let param_norm = calculate_param_norm(model);
                let batch_loss = batch_losses.iter().sum::<f64>() / batch_losses.len() as f64;

                let metric = TrainingMetrics {
                    epoch: epoch + 1,
                    loss: avg_loss,
                    accuracy: avg_accuracy,
                    learning_rate: 0.001, // This should be extracted from optimizer
                    batch_loss,
                    gradient_norm,
                    param_norm,
                };

                metrics.push(metric);

                if (epoch + 1) % 10 == 0 || epoch == 0 || epoch == epochs - 1 {
                    tracing::info!(
                        "Epoch {}/{}, Loss: {:.6}, Accuracy: {:.4}, Grad Norm: {:.6}, Param Norm: {:.6}",
                        epoch + 1, epochs, avg_loss, avg_accuracy, gradient_norm, param_norm
                    );
                }
            }

            TrainingProgress {
                metrics,
                total_epochs: epochs,
                current_epoch: epochs,
                is_complete: true,
            }
        }
    }
}

#[tokio::main]
async fn main() {
    api::run().await;
}