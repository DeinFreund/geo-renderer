use std::convert::TryInto;
use std::fs::create_dir_all;
use std::path::PathBuf;

use anyhow::{ensure, Result};
use clap::Parser;
use image::DynamicImage;
use itertools::Itertools;
use log::{debug, info};
use nalgebra::Vector3;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelIterator;
use serde::{Deserialize, Serialize};

use geo_renderer::camera::Intrinsics;
use geo_renderer::config::StorageConfig;
use geo_renderer::renderer::{RenderRequest, Renderer, RequestPose};
use geo_renderer::Coords;

#[derive(Parser)]
struct Flags {
    /// Path to csv with camera poses to render
    #[clap(long)]
    camera_pose_csv_path: PathBuf,
    /// Minimum view distance to render in m, at most 100km
    #[clap(long)]
    view_range_m: f32,
    /// Folder where the data will be saved
    #[clap(long)]
    output_dir: PathBuf,
    /// Paths to the swisstopo data
    #[clap(flatten)]
    storage_config: StorageConfig,
    /// Verbose printing
    #[clap(long)]
    debug: bool,
}

impl Flags {
    pub fn validate(&mut self) -> Result<()> {
        ensure!(self.camera_pose_csv_path.exists());

        self.storage_config.validate()
    }
}

#[derive(Parser, Serialize)]
struct LV95Coords {
    /// North coordinate to render in LV95
    #[clap(long)]
    easting_m: f32,
    /// East coordinate to render in LV95
    #[clap(long)]
    northing_m: f32,
    /// Altitude above ground level to render, in meters
    #[clap(long)]
    altitude_m: f32,
}

impl From<LV95Coords> for Coords {
    fn from(lv95: LV95Coords) -> Coords {
        Coords::new(lv95.easting_m, lv95.northing_m, lv95.altitude_m)
    }
}

impl From<Coords> for LV95Coords {
    fn from(coords: Coords) -> LV95Coords {
        LV95Coords {
            easting_m: coords.x,
            northing_m: coords.y,
            altitude_m: coords.z,
        }
    }
}

#[derive(Serialize)]
struct Image {
    rgb_image_path: PathBuf,
    depth_image_path: PathBuf,
    camera_pos_lv95: LV95Coords,
    camera_forward: [f32; 3],
    camera_up: [f32; 3],
}

#[derive(Serialize)]
struct RenderedDataset {
    images: Vec<Image>,
    intrinsics: Intrinsics,
}

#[derive(Deserialize)]
struct PoseCsvRecord {
    cam_pos_lv95_e: f32,
    cam_pos_lv95_n: f32,
    cam_pos_lv95_u: f32,
    cam_fwd_lv95_e: f32,
    cam_fwd_lv95_n: f32,
    cam_fwd_lv95_u: f32,
    cam_up_lv95_e: f32,
    cam_up_lv95_n: f32,
    cam_up_lv95_u: f32,
}

async fn run(mut args: Flags) -> Result<()> {
    args.validate()?;
    let intrinsics = Intrinsics::load("camera_params.toml")?;
    create_dir_all(&args.output_dir)?;
    let image_json_path = args.output_dir.join("images.json");
    if image_json_path.exists() {
        info!("Found existing images.json, skipping chunk");
        return Ok(());
    }
    let mut state = Renderer::new(intrinsics.clone()).await;

    let csv_records: Vec<PoseCsvRecord> = csv::Reader::from_path(&args.camera_pose_csv_path)?
        .deserialize()
        .into_iter()
        .collect::<Result<Vec<PoseCsvRecord>, _>>()?;
    let render_requests = csv_records
        .into_iter()
        .enumerate()
        .map(|(id, record)| RenderRequest {
            camera_pose: RequestPose::FacingAsl {
                camera_pos_asl: Coords::new(
                    record.cam_pos_lv95_e,
                    record.cam_pos_lv95_n,
                    record.cam_pos_lv95_u,
                ),
                camera_fwd: Vector3::<f32>::new(
                    record.cam_fwd_lv95_e,
                    record.cam_fwd_lv95_n,
                    record.cam_fwd_lv95_u,
                ),
                camera_up: Vector3::<f32>::new(
                    record.cam_up_lv95_e,
                    record.cam_up_lv95_n,
                    record.cam_up_lv95_u,
                ),
            },
            request_id: id as u32,
        });
    let mut images: Vec<Image> = Vec::new();

    // Render in chunks to prevent running out of memory
    for render_chunk in render_requests.chunks(2000).into_iter() {
        let rendered_requests = state
            .render_images(
                render_chunk.collect_vec(),
                args.view_range_m,
                &args.storage_config,
            )
            .await?;

        info!("Storing {} images", rendered_requests.len());
        images.extend(
            rendered_requests
                .into_par_iter()
                .map(|request| {
                    let filename = args
                        .output_dir
                        .join(format!("image_{}", request.request_id));
                    let rgb_image_path = filename.with_extension("png");
                    let depth_image_path = filename.with_extension("bin");

                    let image_rgba = DynamicImage::ImageRgba8(request.image_rgba);
                    image_rgba.save(&rgb_image_path).unwrap();

                    let depth_bin: &[u8] = bytemuck::cast_slice(&request.image_depth);
                    std::fs::write(&depth_image_path, depth_bin).unwrap();

                    Image {
                        rgb_image_path: PathBuf::from(rgb_image_path.file_name().expect("")),
                        depth_image_path: PathBuf::from(depth_image_path.file_name().expect("")),
                        camera_pos_lv95: request.camera_pos_lv95.into(),
                        camera_forward: request.camera_forward.as_slice().try_into().unwrap(),
                        camera_up: request.camera_up.as_slice().try_into().unwrap(),
                    }
                })
                .collect::<Vec<_>>(),
        );
    }
    let dataset = RenderedDataset { images, intrinsics };
    std::fs::write(image_json_path, serde_json::to_string_pretty(&dataset)?)?;
    Ok(())
}

fn main() {
    let args = Flags::parse();
    let level = if args.debug {
        log::LevelFilter::Debug
    } else {
        log::LevelFilter::Info
    };
    let colors = fern::colors::ColoredLevelConfig::new()
        .debug(fern::colors::Color::Blue)
        .info(fern::colors::Color::Green)
        .error(fern::colors::Color::Red)
        .warn(fern::colors::Color::Yellow);
    debug!("Running in debug mode");
    fern::Dispatch::new()
        .format(move |out, message, record| {
            out.finish(format_args!(
                "{} {} [{}] {}",
                chrono::Local::now().format("[%Y-%m-%d %H:%M:%S:%f]"),
                colors.color(record.level()),
                record.target(),
                message,
            ))
        })
        .level(level)
        .level_for("wgpu_core", log::LevelFilter::Warn)
        .level_for("wgpu_hal", log::LevelFilter::Warn)
        .chain(std::io::stdout())
        .apply()
        .unwrap();
    if let Err(err) = pollster::block_on(run(args)) {
        println!("{}", err);
    }
}
