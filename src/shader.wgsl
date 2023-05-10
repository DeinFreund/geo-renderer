// Vertex shader

struct Camera {
    view: mat4x4<f32>,
    xi: f32,
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
}

@group(1) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    let world_position = vec4<f32>(model.position, 1.0);

    var out: VertexOutput;
    let localPos: vec4<f32> = camera.view * world_position;
    let dist : f32 = sqrt((localPos[0]*localPos[0] + localPos[1]*localPos[1] + localPos[2]*localPos[2])/(localPos[3]*localPos[3]));
    let norm : f32 = -localPos[2] + camera.xi * dist;
    out.clip_position[0] = camera.fx * localPos[0] + camera.cx * norm;
    out.clip_position[1] = camera.fy * localPos[1] + camera.cy * norm;
    out.clip_position[2] = dist*norm *0.0001; // Simple distance, for the proper culling use (-localPos[2] * camera.xi + dist)*norm*0.0001
    out.clip_position[3] = norm;

    out.tex_coords = model.tex_coords;
    return out;
}


// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0)@binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let object_color: vec4<f32> = textureSample(t_diffuse, s_diffuse, in.tex_coords);

    return object_color;
}