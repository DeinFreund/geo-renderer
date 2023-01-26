use std::collections::HashMap;

use log::{info, warn};
use nalgebra::{distance, Point2, Point3};
use rayon::iter::ParallelIterator;
use rayon::prelude::*;

use crate::camera::Camera;
use crate::config::StorageConfig;
use crate::gridsquare::{GridCoords, GridSquare};
use crate::model::Model;

pub struct TerrainGrid {
    tiles: Vec<GridSquare>,
}

impl TerrainGrid {
    /// Loads terrain in a circular grid around center_coords
    ///
    /// # Arguments
    ///
    /// * `center_coords` - GridSquare coordinates of the central tile
    /// * `agl_m` - Altitude of the viewpoint above the terrain
    /// * `camera` - The camera used for the observation
    /// * `view_range_m` - The radius within which to load terrain, all tiles that are within this radius from any part of the central tile are loaded.
    pub fn new(
        center_coords: GridCoords,
        agl_m: f32,
        camera: &Camera,
        view_range_m: f32,
        storage_config: &StorageConfig,
    ) -> Self {
        let mut circle = center_coords.circle_m(view_range_m);
        circle.sort_by(|x, y| (y.0.x, y.0.y).cmp(&(x.0.x, x.0.y)));
        info!("Loading {} terrain tiles", circle.len());
        let mut tiles: HashMap<GridCoords, GridSquare> = circle
            .par_iter()
            .filter_map(|coords| {
                let pt1_m = Point3::new(0.0, coords.min_dist_m(&center_coords), agl_m);
                let pt1_px = camera.project(pt1_m);
                let pt2_m = camera
                    .unproject(Point2::new(pt1_px.x, pt1_px.y + 1.0), agl_m)
                    .unwrap();
                let pt3_m = camera
                    .unproject(Point2::new(pt1_px.x + 1.0, pt1_px.y), agl_m)
                    .unwrap();
                let resolution_m = 0.5 * (distance(&pt1_m, &pt2_m) + distance(&pt1_m, &pt3_m));
                match GridSquare::new(*coords, 1f32 * resolution_m, storage_config.clone()) {
                    Ok(square) => Some((*coords, square)),
                    Err(e) => {
                        warn!("Unable to load square at {:?}: {}", &coords, e);
                        None
                    }
                }
            })
            .collect();

        for coords in &circle {
            let mut tile = tiles.remove(coords).unwrap();
            tile.cleanup_borders(
                tiles.get(&coords.below()),
                tiles.get(&coords.right()),
                None,
                None,
            );
            tiles.insert(*coords, tile);
        }
        for coords in &circle {
            let mut tile = tiles.remove(coords).unwrap();
            tile.cleanup_borders(
                None,
                None,
                tiles.get(&coords.above()),
                tiles.get(&coords.left()),
            );
            tiles.insert(*coords, tile);
        }
        Self {
            tiles: tiles.into_values().into_iter().collect(),
        }
    }

    pub fn models(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Vec<Model> {
        self.tiles
            .par_iter()
            .filter_map(
                |square| match square.model(device, queue, texture_bind_group_layout) {
                    Ok(square) => Some(square),
                    Err(e) => {
                        warn!(
                            "Unable to load square texture at {:?}: {}",
                            square.coords, e
                        );
                        None
                    }
                },
            )
            .collect()
    }
}
