use std::convert::TryInto;
use std::fs::File;
use std::num::NonZeroU32;

use anyhow::{bail, Context, Result};
use image::{imageops::FilterType, ImageBuffer};
use log::debug;
use nalgebra::{Point2, Vector3};
use tiff::decoder::DecodingResult;
use wgpu::util::DeviceExt;

use crate::config::StorageConfig;
use crate::model::{Material, Mesh, Model, ModelVertex};
use crate::texture::Texture;
use crate::Coords;

const IMAGE_SIZE_M: f32 = 1000.0;
const ORTHOIMAGE_RESOLUTION_PX: u32 = 10_000;

const ELEVATION_MAX_LOD: usize = 2;
const ORTHOIMAGE_MAX_LOD: usize = 5;
const MESH_MAX_RESOLUTION: u32 = 4000;
const MESH_MIN_RESOLUTION: u32 = 2; //60;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub struct GridCoords(pub Point2<i32>);

impl From<Coords> for GridCoords {
    fn from(point: Coords) -> Self {
        Self(Point2::new(
            (point.x / IMAGE_SIZE_M).floor() as i32,
            (point.y / IMAGE_SIZE_M).floor() as i32,
        ))
    }
}

impl From<GridCoords> for Coords {
    fn from(coords: GridCoords) -> Self {
        Coords::new(
            coords.0.x as f32 * IMAGE_SIZE_M,
            coords.0.y as f32 * IMAGE_SIZE_M,
            0.0,
        )
    }
}

impl GridCoords {
    pub fn new(x: i32, y: i32) -> Self {
        Self(Point2::new(x, y))
    }

    pub fn circle_m(&self, radius: f32) -> Vec<GridCoords> {
        self.circle((radius / IMAGE_SIZE_M).floor() as i32 + 1)
    }

    pub fn circle(&self, radius: i32) -> Vec<GridCoords> {
        let mut circle_coords: Vec<GridCoords> = Vec::new();
        for x in self.0.x - radius..=self.0.x + radius {
            for y in self.0.y - radius..=self.0.y + radius {
                let pt = Point2::<i32>::new(x, y);
                let diff = pt.coords - self.0.coords;
                if diff.x * diff.x + diff.y * diff.y <= radius * radius {
                    circle_coords.push(GridCoords(pt));
                }
            }
        }
        circle_coords
    }

    pub fn below(&self) -> Self {
        GridCoords(Point2::new(self.0.x, self.0.y + 1))
    }
    pub fn right(&self) -> Self {
        GridCoords(Point2::new(self.0.x + 1, self.0.y))
    }
    pub fn above(&self) -> Self {
        GridCoords(Point2::new(self.0.x, self.0.y - 1))
    }
    pub fn left(&self) -> Self {
        GridCoords(Point2::new(self.0.x - 1, self.0.y))
    }

    /// Distance of closest two corners, i.e. 0 if direct 8-neighbours
    pub fn min_dist_m(&self, other: &GridCoords) -> f32 {
        let dx = ((other.0.x - self.0.x).abs() - 1).max(0) as f32;
        let dy = ((other.0.y - self.0.y).abs() - 1).max(0) as f32;
        (dx * dx + dy * dy).sqrt() * IMAGE_SIZE_M
    }
}

/// Given image resolution, calculate minimum power of 2 LOD that satisfies the target resolution
/// E.g. with image_resolution = 1024 and target_resolution = 256, calc_lod returns 2.0
fn calc_lod(image_resolution: u32, target_resolution: u32) -> usize {
    ((image_resolution / target_resolution) as f32)
        .log2()
        .floor()
        .max(0.0) as usize
}

/// A terrain tile of 1x1 km in a given resolution
#[derive(Debug)]
pub struct GridSquare {
    /// Target resolution for the square, at least 2 (2x2=4 vertices)
    pub resolution: u32,
    /// Origin coordinates of the square
    pub coords: GridCoords,
    /// Grid with elevation data for each vertex
    pub elevation: ndarray::Array2<f32>,
    /// Paths for swisstopo data
    pub storage_config: StorageConfig,
}

impl GridSquare {
    /// Create a grid of size resolution x resolution with altitude in meters.
    pub fn new(
        coords: GridCoords,
        resolution_m: f32,
        storage_config: StorageConfig,
    ) -> Result<GridSquare> {
        let resolution = ((IMAGE_SIZE_M / resolution_m).ceil() as u32).max(2);
        let mut path = storage_config
            .surface_dir
            .join(format!("{}-{}.tif", coords.0.x, coords.0.y));
        if !path.exists() {
            path = storage_config
                .alti_dir
                .join(format!("{}-{}.tif", coords.0.x, coords.0.y))
        }
        let image = File::open(path)?;
        let mut decoder = tiff::decoder::Decoder::new(image)?;

        debug!(
            "Loading tile {:?} at resolution {}m -> {}",
            coords, resolution_m, resolution
        );
        let mesh_resolution = resolution.min(MESH_MAX_RESOLUTION).max(MESH_MIN_RESOLUTION) as usize;

        let lod = calc_lod(decoder.dimensions()?.0, mesh_resolution as u32);
        // Make sure meshes are similar resolutions to allow good matching with neighboring squares
        let mesh_resolution = (decoder.dimensions()?.0 / (1 << lod)).next_power_of_two() as usize;

        debug!("Mesh is {}x{}", mesh_resolution, mesh_resolution);

        decoder.seek_to_image(lod.min(ELEVATION_MAX_LOD))?;
        let (width, height) = decoder.dimensions()?;
        let pixels = if let DecodingResult::F32(pixels) = decoder.read_image()? {
            pixels.into_iter().map(|x| x * 1e-6).collect()
        } else {
            bail!("Elevation Data not F32");
        };
        debug!(
            "Loading Elevation for {:?} at LOD {} ({}x{})",
            coords, lod, width, height
        );
        let mut image: ImageBuffer<image::Luma<f32>, Vec<_>> =
            ImageBuffer::from_vec(width, height, pixels)
                .context("Unable to parse elevation data")?;

        let mut elevation: ndarray::Array2<f32> =
            ndarray::Array2::from_elem(((mesh_resolution + 1), (mesh_resolution + 1)), 0f32);
        if mesh_resolution as u32 != image.dimensions().0 {
            image = image::imageops::resize(
                &image,
                mesh_resolution as u32,
                mesh_resolution as u32,
                image::imageops::FilterType::Gaussian,
            );
            debug!("Resized image to {}x{}", mesh_resolution, mesh_resolution);
        }
        for (y, r) in image.rows().enumerate() {
            for (x, p) in r.enumerate() {
                let y = mesh_resolution - y - 1;
                elevation[[x, y]] = p.0[0] * 1e6;
            }
        }

        // Fill bottom row
        for x in 0..mesh_resolution + 1 {
            elevation[[x, mesh_resolution]] =
                elevation[[x.min(mesh_resolution - 1), mesh_resolution - 1]];
        }
        // Fill rightmost row
        for y in 0..mesh_resolution + 1 {
            elevation[[mesh_resolution, y]] =
                elevation[[mesh_resolution - 1, y.min(mesh_resolution - 1)]];
        }

        Ok(GridSquare {
            resolution,
            coords,
            elevation,
            storage_config,
        })
    }

    /// Fill in the border of the elevation grid to match with neighboring cells
    pub fn cleanup_borders(
        &mut self,
        bottom_neighbor: Option<&GridSquare>,
        right_neighbor: Option<&GridSquare>,
        top_neighbor: Option<&GridSquare>,
        left_neighbor: Option<&GridSquare>,
    ) {
        let mesh_resolution = self.elevation.dim().0 - 1;
        let origin: Coords = self.coords.into();
        // Fill bottom row
        if let Some(bottom_neighbor) = bottom_neighbor {
            for x in 0..mesh_resolution + 1 {
                let coords: Coords = origin
                    + Vector3::new(
                        x as f32 / mesh_resolution as f32 * IMAGE_SIZE_M,
                        IMAGE_SIZE_M,
                        0f32,
                    );
                self.elevation[[x, mesh_resolution]] = bottom_neighbor.sample_altitude(coords);
            }
        }
        // Fill rightmost row
        if let Some(right_neighbor) = right_neighbor {
            for y in 0..mesh_resolution + 1 {
                let coords: Coords = origin
                    + Vector3::new(
                        IMAGE_SIZE_M,
                        y as f32 / mesh_resolution as f32 * IMAGE_SIZE_M,
                        0f32,
                    );
                self.elevation[[mesh_resolution, y]] = right_neighbor.sample_altitude(coords);
            }
        }
        // Fill top row
        if let Some(top_neighbor) = top_neighbor {
            for x in 0..mesh_resolution + 1 {
                let coords: Coords = origin
                    + Vector3::new(x as f32 / mesh_resolution as f32 * IMAGE_SIZE_M, 0f32, 0f32);
                self.elevation[[x, 0]] = top_neighbor.sample_altitude(coords);
            }
        }
        // Fill leftmost row
        if let Some(left_neighbor) = left_neighbor {
            for y in 0..mesh_resolution + 1 {
                let coords: Coords = origin
                    + Vector3::new(0f32, y as f32 / mesh_resolution as f32 * IMAGE_SIZE_M, 0f32);
                self.elevation[[0, y]] = left_neighbor.sample_altitude(coords);
            }
        }
    }

    /// Bilinearly interpolated sampling of the altitude mesh
    pub fn sample_altitude(&self, coords: Coords) -> f32 {
        let origin: Coords = self.coords.into();
        let idx = (self.elevation.dim().0 - 1) as f32 * (coords - origin) / IMAGE_SIZE_M;
        let left = idx.x.floor() as usize;
        let right = left + 1;
        let left_fac = right as f32 - idx.x;
        let right_fac = idx.x - left as f32;
        let bottom = idx.y.floor() as usize;
        let top = bottom + 1;
        let bottom_fac = top as f32 - idx.y;
        let top_fac = idx.y - bottom as f32;
        let left = left.min(self.elevation.dim().0 - 1);
        let right = right.min(self.elevation.dim().0 - 1);
        let bottom = bottom.min(self.elevation.dim().0 - 1);
        let top = top.min(self.elevation.dim().0 - 1);
        let left_val =
            self.elevation[[left, top]] * top_fac + self.elevation[[left, bottom]] * bottom_fac;
        let right_val =
            self.elevation[[right, top]] * top_fac + self.elevation[[right, bottom]] * bottom_fac;
        left_val * left_fac + right_val * right_fac
    }

    pub fn mesh(&self, device: &wgpu::Device) -> Mesh {
        let mut vertices: Vec<ModelVertex> = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        let resolution = self.elevation.len_of(ndarray::Axis(0));
        let inv_resolution = 1f32 / (resolution - 1) as f32;
        let grid_size_m = IMAGE_SIZE_M * inv_resolution;
        let origin: Coords = self.coords.into();
        for x in 0..resolution - 1 {
            for y in 0..resolution - 1 {
                let x0 = x as f32 * grid_size_m + origin.x;
                let x1 = (x + 1) as f32 * grid_size_m + origin.x;
                let y0 = y as f32 * grid_size_m + origin.y;
                let y1 = (y + 1) as f32 * grid_size_m + origin.y;
                let u0 = x as f32 * inv_resolution;
                let v0 = 1.0 - y as f32 * inv_resolution;
                let u1 = (x + 1) as f32 * inv_resolution;
                let v1 = 1.0 - (y + 1) as f32 * inv_resolution;
                let v = vertices.len() as u32;
                indices.append(&mut vec![v + 2, v + 1, v, v + 1, v + 2, v + 3]);
                vertices.push(ModelVertex {
                    position: [x0, y0, self.elevation[[x, y]]],
                    tex_coords: [u0, v0],
                });
                vertices.push(ModelVertex {
                    position: [x0, y1, self.elevation[[x, y + 1]]],
                    tex_coords: [u0, v1],
                });
                vertices.push(ModelVertex {
                    position: [x1, y0, self.elevation[[x + 1, y]]],
                    tex_coords: [u1, v0],
                });
                vertices.push(ModelVertex {
                    position: [x1, y1, self.elevation[[x + 1, y + 1]]],
                    tex_coords: [u1, v1],
                });
            }
        }

        let model_name = "dummy";
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?} Vertex Buffer", model_name)),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{:?} Index Buffer", model_name)),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        Mesh {
            name: model_name.to_string(),
            vertex_buffer,
            index_buffer,
            num_elements: indices.len() as u32,
            material: 0,
        }
    }

    pub fn model(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<Model> {
        let resolution = self
            .resolution
            .min(ORTHOIMAGE_RESOLUTION_PX / (1 << self.storage_config.image_max_lod));

        let lod = calc_lod(ORTHOIMAGE_RESOLUTION_PX, resolution)
            .max(self.storage_config.image_max_lod)
            .min(ORTHOIMAGE_MAX_LOD);
        let path = self.storage_config.image_dir.join(format!(
            "{}-{}_lod{}.jpg",
            self.coords.0.x, self.coords.0.y, lod
        ));
        let data = std::fs::read(&path)?;
        let label = path.file_name().unwrap().to_str().unwrap();
        let mut img = image::load_from_memory(&data)?;
        let max_lod = NonZeroU32::new(
            (img.width() as f32 / resolution as f32)
                .log2()
                .max(0.0)
                .floor() as u32
                + 1,
        )
        .unwrap();
        if img.width() > ORTHOIMAGE_RESOLUTION_PX / (1 << self.storage_config.image_max_lod) {
            let width = img.width();
            img = img.resize(
                ORTHOIMAGE_RESOLUTION_PX / (1 << self.storage_config.image_max_lod),
                ORTHOIMAGE_RESOLUTION_PX / (1 << self.storage_config.image_max_lod),
                FilterType::Lanczos3,
            );
            img = img.resize(width, width, FilterType::Lanczos3);
        }
        if max_lod == 1.try_into().unwrap() && resolution < ORTHOIMAGE_RESOLUTION_PX {
            // If there's only one LOD, downscale the image to the required resolution
            img = img.resize(resolution, resolution, FilterType::Lanczos3);
        }
        let diffuse_texture = Texture::from_image(device, queue, &img, Some(label), max_lod)?;
        debug!(
            "Loading texture for {:?} at LOD {} ({}x{}) target {} max LOD {}",
            self.coords,
            lod,
            diffuse_texture.size.width,
            diffuse_texture.size.height,
            resolution,
            max_lod
        );
        Ok(Model {
            meshes: vec![self.mesh(device)],
            materials: vec![Material::new(
                device,
                "dummy mat",
                diffuse_texture,
                texture_bind_group_layout,
            )],
        })
    }
}
