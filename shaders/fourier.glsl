layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(location = 0) uniform int gridSize;
layout(location = 1) uniform int nGrids;

layout(rg32f, binding = 0) uniform image3D spectrumTex;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);
  ivec3 texSize = imageSize(spectrumTex);

  if (any(greaterThanEqual(pos, texSize)))
    return;

  int localZ = pos.z % gridSize;
  if (pos.x == 0 && pos.y == 0 && localZ == 0) {
    imageStore(spectrumTex, pos, vec4(0.0, 0.0, 0.0, 0.0));
  }
}
