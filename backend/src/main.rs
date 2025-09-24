use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use actix_cors::Cors;
use actix_multipart::Multipart;
use futures_util::StreamExt;
use reqwest::Client;
use serde::Serialize;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use sysinfo::System;

mod ai;

#[derive(Serialize)]
struct HardwareInfo {
    cpu_usage: Vec<f32>,
    total_memory: u64,
    used_memory: u64,
    total_swap: u64,
    used_swap: u64,
    uptime: u64,
    energy_consumption_wh: f64, // Simulated energy
}

struct AppState {
    sys: Mutex<System>,
    start_time: Mutex<std::time::Instant>,
}

#[get("/api/hardware")]
async fn get_hardware_info(data: web::Data<AppState>) -> impl Responder {
    let mut sys = data.sys.lock().unwrap();
    sys.refresh_all();

    // Simulate energy consumption for demonstration.
    // In a real scenario, this would require specific hardware interfaces.
    // Assuming an average desktop power consumption of 150W.
    let elapsed_seconds = data.start_time.lock().unwrap().elapsed().as_secs_f64();
    let power_watts = 150.0;
    let energy_joules = power_watts * elapsed_seconds;
    let energy_consumption_wh = energy_joules / 3600.0;

    let info = HardwareInfo {
        cpu_usage: sys.cpus().iter().map(|cpu| cpu.cpu_usage()).collect(),
        total_memory: sys.total_memory(),
        used_memory: sys.used_memory(),
        total_swap: sys.total_swap(),
        used_swap: sys.used_swap(),
        uptime: System::uptime(),
        energy_consumption_wh,
    };

    HttpResponse::Ok().json(info)
}

// Placeholder for starting a training process
#[get("/api/train")]
async fn start_training() -> impl Responder {
    // In the real app, this would trigger the neural network training loop.
    // For now, it returns a success message.
    HttpResponse::Ok().json({
        serde_json::json!({
            "status": "success",
            "message": "Training process initiated successfully."
        })
    })
}

#[post("/api/process_dataset")]
async fn process_dataset(mut payload: Multipart) -> impl Responder {
    // Save uploaded file
    while let Some(item) = payload.next().await {
        let mut field = item.unwrap();

        // Get filename before starting mutable borrow
        let filename = field.content_disposition()
            .get_filename()
            .map(|s| s.to_string())
            .unwrap_or("dataset.csv".to_string());
        let filepath = format!("/tmp/{}", filename);

        let mut f = File::create(&filepath).unwrap();
        while let Some(chunk) = field.next().await {
            let data = chunk.unwrap();
            f.write_all(&data).unwrap();
        }

        // Now use filename (owned String, no borrow conflict)
        if filename == "personal.json" {
            // TODO: Add your personal data processing logic here
            // For now, just simulate success
            return HttpResponse::Ok().json({
                serde_json::json!({
                    "status": "success",
                    "message": "Personal data processed and ready."
                })
            });
        }

        // Simulate success for now
        return HttpResponse::Ok().json({
            serde_json::json!({
                "status": "success",
                "message": "Dataset processed and ready."
            })
        });
    }
    HttpResponse::BadRequest().json({
        serde_json::json!({
            "status": "error",
            "message": "No dataset uploaded."
        })
    })
}

#[post("/api/download_dataset")]
async fn download_dataset(info: web::Json<serde_json::Value>) -> impl Responder {
    let dataset = info["dataset"].as_str().unwrap_or("");
    let url = match dataset {
        "mnist" => "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
        "cifar10" => "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
        _ => return HttpResponse::BadRequest().json({
            serde_json::json!({"status": "error", "message": "Unknown dataset"})
        }),
    };

    let filename = format!("/tmp/{}", dataset);
    if !Path::new(&filename).exists() {
        let client = Client::new();
        let resp = client.get(url).send().await;
        match resp {
            Ok(mut response) => {
                let bytes = response.bytes().await.unwrap();
                std::fs::write(&filename, &bytes).unwrap();
            }
            Err(_) => {
                return HttpResponse::InternalServerError().json({
                    serde_json::json!({"status": "error", "message": "Download failed"})
                });
            }
        }
    }
    HttpResponse::Ok().json({
        serde_json::json!({"status": "success", "message": "Dataset downloaded"})
    })
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize system info
    let app_state = web::Data::new(AppState {
        sys: Mutex::new(System::new_all()),
        start_time: Mutex::new(std::time::Instant::now()),
    });

    println!("🚀 Server starting at http://0.0.0.0:8080");

    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allow_any_method()
            .allow_any_header()
            .max_age(3600);

        App::new()
            .wrap(cors)
            .app_data(app_state.clone())
            .service(get_hardware_info)
            .service(start_training)
            .service(process_dataset)
            .service(download_dataset)
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}

