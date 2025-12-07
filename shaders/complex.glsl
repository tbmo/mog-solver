#version 460
layout(local_size_x = 8, local_size_y = 8) in;

layout(location = 0) uniform ivec3 tensorDimensions;

layout(std430, binding = 0) readonly buffer MassBuffer { uint mass_grid[]; };

layout(rg32f, binding = 1) writeonly uniform image2D complexTexture;

void main() {
  ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
  if (any(greaterThanEqual(pos, tensorDimensions.xy)))
    return;

  int idx = pos.y * tensorDimensions.x + pos.x;
  float mass = float(mass_grid[idx]) / 1000.0; // Convert back from uint

  imageStore(complexTexture, pos, vec4(mass, 0.0, 0.0, 0.0));
}