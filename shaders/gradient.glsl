#version 460
layout(local_size_x = 8, local_size_y = 8) in;

layout(location = 0) uniform ivec3 tensorDimensions;
layout(location = 1) uniform float voxelSize;

layout(rg32f, binding = 0) readonly uniform image2D inputTexture;
layout(rgba32f, binding = 1) writeonly uniform image2D gradientTexture;

ivec2 wrapCoords(ivec2 pos) {
  return (pos % tensorDimensions.xy + tensorDimensions.xy) %
         tensorDimensions.xy;
}

float getFieldValue(ivec2 pos) {
  pos = wrapCoords(pos);
  return imageLoad(inputTexture, pos).x;
}

void main() {
  ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
  if (any(greaterThanEqual(pos, tensorDimensions.xy)))
    return;

  float dx =
      (getFieldValue(pos + ivec2(1, 0)) - getFieldValue(pos - ivec2(1, 0))) /
      (2.0 * voxelSize);
  float dy =
      (getFieldValue(pos + ivec2(0, 1)) - getFieldValue(pos - ivec2(0, 1))) /
      (2.0 * voxelSize);

  vec2 gradient = vec2(dx, dy);
  float magnitude = length(gradient);

  imageStore(gradientTexture, pos, vec4(gradient, 0.0, magnitude));
}