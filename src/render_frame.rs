use std::convert::TryInto;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use image::DynamicImage;
use nalgebra::Point3;

use serde::Serialize;
use geo_renderer::camera::Intrinsics;
use geo_renderer::config::StorageConfig;
use geo_renderer::renderer::{RenderRequest, Renderer, RequestPose};
use geo_renderer::Coords;

#[derive(Parser)]
struct Flags {
    /// Coordinate to render in LV95
    #[clap(flatten)]
    camera_pos: LV95Coords,
    /// Minimum view distance to render in m, at most 100km
    #[clap(long)]
    view_range_m: f32,
    /// Path to store the image
    #[clap(long)]
    output: PathBuf,
    /// Paths to the swisstopo data
    #[clap(flatten)]
    storage_config: StorageConfig,
    /// Verbose printing
    #[clap(long)]
    debug: bool,
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

async fn run(args: Flags) -> Result<()> {
    let intrinsics = Intrinsics::load("camera_params.toml")?;
    let mut state = Renderer::new(intrinsics.clone()).await;

    let camera_pos = Point3::<f32>::new(
        args.camera_pos.easting_m,
        args.camera_pos.northing_m,
        args.camera_pos.altitude_m,
    );
    let render_requests: Vec<RenderRequest> = vec![RenderRequest {
        camera_pose: RequestPose::PositionAgl {
            camera_pos_agl: camera_pos,
        },
        request_id: 0,
    }];
    let rendered_requests = state
        .render_images(render_requests, args.view_range_m, &args.storage_config)
        .await?;

    let images = rendered_requests
        .into_iter()
        .map(|request| {
            let rgb_image_path = args.output.with_extension("png");
            let depth_image_path = args.output.with_extension("bin");

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
    std::fs::write(
        args.output.with_extension("json"),
        serde_json::to_string_pretty(&dataset)?,
    )?;
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
