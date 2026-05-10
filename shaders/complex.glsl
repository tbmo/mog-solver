layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(std430, binding = 0) readonly buffer MassBuffer { uint mass_grid[]; };

layout(rg32f, binding = 1) writeonly uniform image3D complexTexture;

layout(location = 0) uniform int gridSize;
layout(location = 1) uniform int nGrids;

void main() {
  ivec3 pos = ivec3(gl_GlobalInvocationID);

  int gridIdx = pos.z / gridSize;
  int localZ = pos.z % gridSize;

  if (pos.x >= gridSize || pos.y >= gridSize || gridIdx >= nGrids)
    return;

  int cellsPerGrid = gridSize * gridSize * gridSize;
  int idx = gridIdx * cellsPerGrid + localZ * gridSize * gridSize +
            pos.y * gridSize + pos.x;

  float mass = float(mass_grid[idx]) / 1000.0;
  imageStore(complexTexture, pos, vec4(mass, 0.0, 0.0, 0.0));
}
