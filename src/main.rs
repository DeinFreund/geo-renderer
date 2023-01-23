use std::convert::TryInto;
use std::fs::create_dir_all;
use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser;
use image::DynamicImage;
use log::{debug, info};
use nalgebra::Point3;
use rayon::iter::ParallelIterator;
use rayon::prelude::IntoParallelIterator;
use serde::Serialize;

use geo_renderer::camera::Intrinsics;
use geo_renderer::config::StorageConfig;
use geo_renderer::gridsquare::GridCoords;
use geo_renderer::renderer::{RenderRequest, Renderer, RequestPose};
use geo_renderer::Coords;

#[derive(Parser)]
struct Flags {
    /// Leftmost chunk to render in LV95 in km
    #[clap(long)]
    min_easting: i32,
    /// Rightmost chunk to render in LV95 in km
    #[clap(long)]
    max_easting: i32,
    /// Bottom chunk to render in LV95 in km
    #[clap(long)]
    min_northing: i32,
    /// Top chunk to render in LV95 in km
    #[clap(long)]
    max_northing: i32,
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
        (self.min_northing, self.max_northing) = (
            self.min_northing.min(self.max_northing),
            self.min_northing.max(self.max_northing),
        );
        (self.min_easting, self.max_easting) = (
            self.min_easting.min(self.max_easting),
            self.min_easting.max(self.max_easting),
        );

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

async fn render_chunk(
    chunk_coords: GridCoords,
    view_range_m: f32,
    storage_config: &StorageConfig,
    output_dir: &Path,
) -> Result<()> {
    let intrinsics = Intrinsics::load("camera_params.toml")?;
    let output_dir = output_dir.join(format!("render_{}_{}", chunk_coords.0.x, chunk_coords.0.y));
    create_dir_all(&output_dir)?;
    let image_json_path = output_dir.join("images.json");
    if image_json_path.exists() {
        info!("Found existing images.json, skipping chunk");
        return Ok(());
    }
    let mut state = Renderer::new(intrinsics.clone()).await;

    let camera_pos: Coords = chunk_coords.into();
    let mut camera_positions: Vec<Coords> = Vec::new();
    for agl_m in [/*100,  200, */ 300, 550, 800, 1200, 2000] {
        let resolution = 1500 / agl_m + 1;
        let step_m = 1000.0 / resolution as f32;
        let offset_m = step_m / 2.0;
        for x_step in 0..resolution {
            for y_step in 0..resolution {
                camera_positions.push(Point3::<f32>::new(
                    camera_pos.x + step_m * x_step as f32 + offset_m,
                    camera_pos.y + step_m * y_step as f32 + offset_m,
                    agl_m as f32,
                ));
            }
        }
    }
    let render_requests: Vec<RenderRequest> = camera_positions
        .into_iter()
        .enumerate()
        .map(|(id, pos)| RenderRequest {
            camera_pose: RequestPose::PositionAgl {
                camera_pos_agl: pos,
            },
            request_id: id as u32,
        })
        .collect();
    let rendered_requests = state
        .render_images(render_requests, view_range_m, storage_config)
        .await?;

    info!("Storing {} images", rendered_requests.len());
    let images = rendered_requests
        .into_par_iter()
        .map(|request| {
            let filename = output_dir.join(&format!("image_{}", request.request_id));
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
        .collect();
    let dataset = RenderedDataset { images, intrinsics };
    std::fs::write(image_json_path, serde_json::to_string_pretty(&dataset)?)?;
    Ok(())
}

async fn run(mut args: Flags) -> Result<()> {
    args.validate()?;
    for x in args.min_easting..=args.max_easting {
        for y in args.min_northing..=args.max_northing {
            let chunk_coords = GridCoords::new(x, y);
            render_chunk(
                chunk_coords,
                args.view_range_m,
                &args.storage_config,
                &args.output_dir,
            )
            .await?;
        }
    }
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
