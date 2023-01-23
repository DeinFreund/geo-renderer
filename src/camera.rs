use std::path::Path;

use anyhow::{bail, Result};
use nalgebra::{Matrix4, Point2, Point3, Vector3};
use serde::{Deserialize, Serialize};

use crate::Coords;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
/// Byte representation of camera parameters for use in the shader
pub struct CameraUniform {
    view: [[f32; 4]; 4],
    xi: f32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    /// 16 byte padding
    dummy: [f32; 3],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view: Matrix4::identity().into(),
            xi: 0.0,
            fx: 0.0,
            fy: 0.0,
            cx: 0.0,
            cy: 0.0,
            dummy: [0.0, 0.0, 0.0],
        }
    }

    /// Recalculate camera parameters from a given config
    pub fn update(&mut self, camera: &Camera) {
        self.view = (camera.calc_matrix()).into();
        let intrinsics = &camera.intrinsics;
        self.xi = camera.intrinsics.xi;
        // Change parameters from [0, w] x [0, h] to [-1, 1] x [-1, 1] camera coordinates
        self.fx = 2.0 * intrinsics.focal_length_x_px / intrinsics.image_width_px as f32;
        self.fy = 2.0 * intrinsics.focal_length_y_px / intrinsics.image_height_px as f32;
        self.cx = 2.0 * intrinsics.optical_center_x_px / intrinsics.image_width_px as f32 - 1.0;
        self.cy = 2.0 * intrinsics.optical_center_y_px / intrinsics.image_height_px as f32 - 1.0;
    }
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Intrinsics {
    /// Xi parameter for fisheye model
    pub xi: f32,
    /// Focal length for x and y axis in pixels
    pub focal_length_x_px: f32,
    pub focal_length_y_px: f32,
    /// Optical center coordinates in pixels, normally in the center of the image
    pub optical_center_x_px: f32,
    pub optical_center_y_px: f32,
    /// Image resolution (width and height)
    pub image_width_px: u32,
    pub image_height_px: u32,
}

impl Intrinsics {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        Ok(toml::from_str(&std::fs::read_to_string(path)?)?)
    }
}

#[derive(Debug)]
pub struct Camera {
    pub position: Coords,
    pub forward: Vector3<f32>,
    pub up: Vector3<f32>,
    pub intrinsics: Intrinsics,
}

impl Camera {
    pub fn new<V: Into<Coords>>(position: V, intrinsics: Intrinsics) -> Self {
        Self {
            position: position.into(),
            forward: Vector3::new(0.0, 0.0, -1.0),
            up: Vector3::new(0.0, -1.0, 0.0),
            intrinsics,
        }
    }

    pub fn calc_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(&self.position, &(self.position + self.forward), &self.up)
    }

    /// Project world (camera frame, positive z) into pixel (screen) coordinates
    pub fn project(&self, point_m: Coords) -> Point2<f32> {
        let norm: f32 = point_m.z + self.intrinsics.xi * point_m.coords.norm();
        Point2::new(
            self.intrinsics.focal_length_x_px * point_m.x / norm
                + self.intrinsics.optical_center_x_px,
            self.intrinsics.focal_length_y_px * point_m.y / norm
                + self.intrinsics.optical_center_y_px,
        )
    }

    /// Project  pixel (screen) into world (camera frame, positive z) coordinates
    pub fn unproject(&self, mut point_px: Point2<f32>, depth_m: f32) -> Result<Coords> {
        point_px.x =
            (point_px.x - self.intrinsics.optical_center_x_px) / self.intrinsics.focal_length_x_px;
        point_px.y =
            (point_px.y - self.intrinsics.optical_center_y_px) / self.intrinsics.focal_length_y_px;

        let norm2 = point_px.coords.norm_squared();
        let xi2 = self.intrinsics.xi * self.intrinsics.xi;
        let normxi2 = norm2 * xi2;

        let arg = 1.0 + norm2 - normxi2;
        if arg <= 0.0 {
            bail!("Point not in FOV")
        }
        let a = self.intrinsics.xi + arg.sqrt();
        let s = a / (a - self.intrinsics.xi * (norm2 + 1.0));
        Ok(depth_m * Point3::new(s * point_px.x, s * point_px.y, 1.0))
    }
}
