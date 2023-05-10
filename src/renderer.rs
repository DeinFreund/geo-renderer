use std::num::NonZeroU32;

use anyhow::Result;
use image::{ImageBuffer, Rgba};
use itertools::Itertools;
use log::info;
use nalgebra::Vector3;
use wgpu::util::DeviceExt;

use crate::camera::{Camera, CameraUniform, Intrinsics};
use crate::config::StorageConfig;
use crate::gridsquare::{GridCoords, GridSquare};
use crate::model::{DrawModel, Model, Vertex};
use crate::terraingrid::TerrainGrid;
use crate::{model, texture, Coords};

#[derive(Debug, Copy, Clone)]
pub enum RequestPose {
    PositionAgl {
        camera_pos_agl: Coords,
    },
    PositionAsl {
        camera_pos_asl: Coords,
    },
    FacingAsl {
        camera_pos_asl: Coords,
        camera_fwd: Vector3<f32>,
        camera_up: Vector3<f32>,
    },
}

impl From<RequestPose> for GridCoords {
    fn from(pose: RequestPose) -> Self {
        match pose {
            RequestPose::PositionAgl { camera_pos_agl } => camera_pos_agl,
            RequestPose::PositionAsl { camera_pos_asl } => camera_pos_asl,
            RequestPose::FacingAsl { camera_pos_asl, .. } => camera_pos_asl,
        }
        .into()
    }
}

#[derive(Debug)]
pub struct RenderRequest {
    pub camera_pose: RequestPose,
    pub request_id: u32,
}

struct NormalizedRenderRequest {
    camera_pos_agl: Coords,
    camera_pos_asl: Coords,
    camera_fwd: Vector3<f32>,
    camera_up: Vector3<f32>,
    request_id: u32,
}

impl RenderRequest {
    fn normalize(self, grid_square: &GridSquare) -> NormalizedRenderRequest {
        match self.camera_pose {
            RequestPose::PositionAgl { camera_pos_agl } => NormalizedRenderRequest {
                camera_pos_agl,
                camera_pos_asl: Coords::new(
                    camera_pos_agl.x,
                    camera_pos_agl.y,
                    camera_pos_agl.z + grid_square.sample_altitude(camera_pos_agl),
                ),
                camera_fwd: Vector3::new(0.0, 0.0, -1.0),
                camera_up: Vector3::new(0.0, -1.0, 0.0),
                request_id: self.request_id,
            },
            RequestPose::PositionAsl { camera_pos_asl } => NormalizedRenderRequest {
                camera_pos_agl: Coords::new(
                    camera_pos_asl.x,
                    camera_pos_asl.y,
                    camera_pos_asl.z - grid_square.sample_altitude(camera_pos_asl),
                ),
                camera_pos_asl,
                camera_fwd: Vector3::new(0.0, 0.0, -1.0),
                camera_up: Vector3::new(0.0, -1.0, 0.0),
                request_id: self.request_id,
            },
            RequestPose::FacingAsl {
                camera_pos_asl,
                camera_fwd,
                camera_up,
            } => NormalizedRenderRequest {
                camera_pos_agl: Coords::new(
                    camera_pos_asl.x,
                    camera_pos_asl.y,
                    camera_pos_asl.z - grid_square.sample_altitude(camera_pos_asl),
                ),
                camera_pos_asl,
                camera_fwd,
                camera_up,
                request_id: self.request_id,
            },
        }
    }
}

#[derive(Debug, Default)]
pub struct RenderedRequest {
    pub camera_pos_agl: Coords,
    pub camera_pos_lv95: Coords,
    pub camera_forward: Vector3<f32>,
    pub camera_up: Vector3<f32>,
    pub request_id: u32,
    pub image_rgba: ImageBuffer<Rgba<u8>, Vec<u8>>,
    pub image_depth: Vec<f32>,
}

pub struct Renderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    render_pipeline: wgpu::RenderPipeline,
    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    output_buffer: wgpu::Buffer,
    depth_output_buffer: wgpu::Buffer,
    render_texture_view: wgpu::TextureView,
    render_texture_size: wgpu::Extent3d,
    render_texture: wgpu::Texture,
    depth_texture: texture::Texture,
}

impl Renderer {
    pub async fn new(intrinsics: Intrinsics) -> Self {
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();
        let limits = wgpu::Limits {
            max_texture_dimension_1d: 16384,
            max_texture_dimension_2d: 16384,
            max_buffer_size: 1 << 32,
            ..Default::default()
        };
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::POLYGON_MODE_LINE,
                    limits,
                },
                None,
            )
            .await
            .unwrap();

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        // This should match the filterable field of the
                        // corresponding Texture entry above.
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        let render_texture_desc = wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: intrinsics.image_width_px,
                height: intrinsics.image_height_px,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("RenderTexture"),
        };
        let render_texture = device.create_texture(&render_texture_desc);
        let render_texture_view = render_texture.create_view(&Default::default());

        let u32_size = std::mem::size_of::<u32>() as u32;

        let output_buffer_size =
            (u32_size * render_texture_desc.size.width * render_texture_desc.size.height)
                as wgpu::BufferAddress;
        let output_buffer_desc = wgpu::BufferDescriptor {
            size: output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            label: None,
            mapped_at_creation: false,
        };
        let output_buffer = device.create_buffer(&output_buffer_desc);

        // Camera
        let camera = Camera::new(Coords::new(0.0, 0.0, 0.0), intrinsics);
        let camera_uniform = CameraUniform::new();

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let depth_texture = texture::Texture::create_depth_texture(
            &device,
            render_texture_desc.size,
            "depth_texture",
        );

        let f32_size = std::mem::size_of::<f32>() as u32;
        let depth_output_buffer_size =
            (f32_size * render_texture_desc.size.width * render_texture_desc.size.height)
                as wgpu::BufferAddress;
        let depth_output_buffer_desc = wgpu::BufferDescriptor {
            size: depth_output_buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            label: None,
            mapped_at_creation: false,
        };
        let depth_output_buffer = device.create_buffer(&depth_output_buffer_desc);

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&texture_bind_group_layout, &camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Normal Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            };
            Self::create_render_pipeline(
                &device,
                &render_pipeline_layout,
                render_texture_desc.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                shader,
            )
        };

        Self {
            device,
            queue,
            render_pipeline,
            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            texture_bind_group_layout,
            render_texture_view,
            render_texture_size: render_texture_desc.size,
            render_texture,
            output_buffer,
            depth_texture,
            depth_output_buffer,
        }
    }

    fn create_render_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        vertex_layouts: &[wgpu::VertexBufferLayout],
        shader: wgpu::ShaderModuleDescriptor,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(shader);

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("{:?}", shader)),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: vertex_layouts,
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState {
                        alpha: wgpu::BlendComponent::REPLACE,
                        color: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill, // Set this to Line for mesh rendering
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        })
    }

    /// Fills in all optional fields in the render request
    pub async fn render_images(
        &mut self,
        render_requests: Vec<RenderRequest>,
        view_range_m: f32,
        storage_config: &StorageConfig,
    ) -> Result<Vec<RenderedRequest>> {
        let camera_positions = render_requests
            .into_iter()
            .map(|req| -> (GridCoords, RenderRequest) { (req.camera_pose.into(), req) })
            .into_group_map();
        let mut rendered_requests: Vec<RenderedRequest> = Vec::new();
        for (grid_coords, chunk_requests) in camera_positions {
            let grid_square = GridSquare::new(grid_coords, 10.0, storage_config.clone())?;
            let mut chunk_requests: Vec<NormalizedRenderRequest> = chunk_requests
                .into_iter()
                .map(|r| r.normalize(&grid_square))
                .collect();
            chunk_requests.sort_by(|p1, p2| p1.camera_pos_agl.z.total_cmp(&p2.camera_pos_agl.z));
            let mut models = Vec::new();
            let mut agl_m = -1000.0;
            for render_request in chunk_requests {
                if render_request.camera_pos_agl.z > 1.5 * agl_m {
                    agl_m = render_request.camera_pos_agl.z;
                    models = TerrainGrid::new(
                        grid_coords,
                        agl_m,
                        &self.camera,
                        view_range_m,
                        storage_config,
                    )
                    .models(
                        &self.device,
                        &self.queue,
                        &self.texture_bind_group_layout,
                    );
                }
                info!(
                    "Rendering image {} at {:?} agl: {}/{}m",
                    &render_request.request_id,
                    &render_request.camera_pos_asl,
                    render_request.camera_pos_agl.z,
                    agl_m
                );
                rendered_requests.push(
                    self.render_image(
                        render_request.camera_pos_agl,
                        render_request.camera_pos_asl,
                        render_request.camera_fwd,
                        render_request.camera_up,
                        render_request.request_id,
                        &models,
                    )
                    .await?,
                );
            }
        }
        rendered_requests.sort_by_key(|r| r.request_id);
        Ok(rendered_requests)
    }

    pub async fn render_image(
        &mut self,
        camera_pos_agl: Coords,
        camera_pos_asl: Coords,
        camera_fwd_lv95: Vector3<f32>,
        camera_up_lv95: Vector3<f32>,
        request_id: u32,
        models: &Vec<Model>,
    ) -> Result<RenderedRequest> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        self.camera.position = camera_pos_asl;
        self.camera.forward = camera_fwd_lv95;
        self.camera.up = camera_up_lv95;
        self.camera_uniform.update(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        let render_pass_desc = wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.render_texture_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.2,
                        b: 0.3,
                        a: 1.0,
                    }),
                    store: true,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth_texture.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: true,
                }),
                stencil_ops: None,
            }),
        };

        {
            // Scope for render_pass
            let mut render_pass = encoder.begin_render_pass(&render_pass_desc);
            render_pass.set_pipeline(&self.render_pipeline);
            for model in models {
                render_pass.draw_model(model, &self.camera_bind_group);
            }
        }

        let u32_size = std::mem::size_of::<u32>() as u32;
        let f32_size = std::mem::size_of::<f32>() as u32;
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.render_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(u32_size * self.render_texture_size.width),
                    rows_per_image: NonZeroU32::new(self.render_texture_size.height),
                },
            },
            self.render_texture_size,
        );

        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                aspect: wgpu::TextureAspect::All,
                texture: &self.depth_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.depth_output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: NonZeroU32::new(f32_size * self.render_texture_size.width),
                    rows_per_image: NonZeroU32::new(self.render_texture_size.height),
                },
            },
            self.render_texture_size,
        );

        self.queue.submit(Some(encoder.finish()));

        let rendered_request;

        {
            let buffer_slice = self.output_buffer.slice(..);
            let depth_buffer_slice = self.depth_output_buffer.slice(..);

            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            self.device.poll(wgpu::Maintain::Wait);
            rx.receive().await.unwrap().unwrap();

            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
            depth_buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
            self.device.poll(wgpu::Maintain::Wait);
            rx.receive().await.unwrap().unwrap();

            let data = (*buffer_slice.get_mapped_range()).to_vec();
            let depth_data = (*depth_buffer_slice.get_mapped_range()).to_vec();

            let image_rgba = ImageBuffer::<Rgba<u8>, _>::from_raw(
                self.render_texture_size.width,
                self.render_texture_size.height,
                data,
            )
            .unwrap();

            rendered_request = RenderedRequest {
                camera_pos_agl,
                camera_pos_lv95: camera_pos_asl,
                camera_forward: self.camera.forward,
                camera_up: self.camera.up,
                request_id,
                image_rgba,
                image_depth: bytemuck::cast_slice(&depth_data).to_vec(),
            };
        }
        self.output_buffer.unmap();
        self.depth_output_buffer.unmap();
        Ok(rendered_request)
    }
}
