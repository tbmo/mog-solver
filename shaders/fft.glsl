#version 460
// Injected by Python: #define DIM 2 or #define DIM 3
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(location = 0) uniform ivec3 tensorDimensions;
layout(location = 1) uniform int stage;
layout(location = 2) uniform int direction;
layout(location = 3) uniform int axis; // 0=X, 1=Y, 2=Z

// --- AGNOSTIC TYPES ---
#if DIM == 3
#define IVEC_TYPE ivec3
layout(rg32f, binding = 0) readonly uniform image3D inputTexture;
layout(rg32f, binding = 1) writeonly uniform image3D outputTexture;
#else
#define IVEC_TYPE ivec2
layout(rg32f, binding = 0) readonly uniform image2D inputTexture;
layout(rg32f, binding = 1) writeonly uniform image2D outputTexture;
#endif
// ----------------------

const float PI = 3.14159265359;
const float PI2 = 2.0 * PI;

vec2 complex_mul(vec2 a, vec2 b) {
  return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec2 compute_twiddle(int k, int stageSize) {
  if (k == 0)
    return vec2(1.0, 0.0);
  if (k == stageSize / 2)
    return vec2(-1.0, 0.0);

  float phase = -float(direction) * PI2 * float(k) / float(stageSize);
  float c = cos(phase);
  float s = sin(phase);

  // Numerical stability
  if (abs(c) < 1e-6)
    c = 0.0;
  if (abs(s) < 1e-6)
    s = 0.0;
  return vec2(c, s);
}

// --- INDEXING HELPERS ---
int get_axis_index(IVEC_TYPE pos) {
  if (axis == 0)
    return pos.x;
  if (axis == 1)
    return pos.y;
#if DIM == 3
  if (axis == 2)
    return pos.z;
#endif
  return 0;
}

IVEC_TYPE set_axis_index(IVEC_TYPE pos, int idx) {
  if (axis == 0)
    pos.x = idx;
  else if (axis == 1)
    pos.y = idx;
#if DIM == 3
  else if (axis == 2)
    pos.z = idx;
#endif
  return pos;
}

int get_axis_size() {
  if (axis == 0)
    return tensorDimensions.x;
  if (axis == 1)
    return tensorDimensions.y;
#if DIM == 3
  if (axis == 2)
    return tensorDimensions.z;
#endif
  return tensorDimensions.x;
}
// ------------------------

uint bit_reverse(uint x, uint n) {
  uint result = 0;
  for (uint i = 0; i < n; ++i) {
    result = (result << 1) | (x & 1);
    x >>= 1;
  }
  return result;
}

void main() {
  // 1. Get agnostic coordinate
  IVEC_TYPE pos = IVEC_TYPE(gl_GlobalInvocationID);

// 2. Bounds check (agnostic)
#if DIM == 3
  if (any(greaterThanEqual(pos, tensorDimensions)))
    return;
#else
  if (any(greaterThanEqual(pos, tensorDimensions.xy)))
    return;
#endif

  int axisSize = get_axis_size();
  int axisIndex = get_axis_index(pos);

  if (axisIndex >= axisSize)
    return;

  uint n = uint(log2(float(axisSize)));

  // Stage 0: Bit reversal
  if (stage == 0) {
    uint revIndex = bit_reverse(uint(axisIndex), n);
    IVEC_TYPE revPos = set_axis_index(pos, int(revIndex));
    vec2 val = imageLoad(inputTexture, pos).xy;
    imageStore(outputTexture, revPos, vec4(val, 0.0, 0.0));
    return;
  }

  // Butterfly stages
  int stageSize = 1 << stage;
  int halfStageSize = stageSize >> 1;

  // [FIX 1] Eliminate Race Condition:
  // We only need N/2 threads. If index is in the second half, kill it.
  if (axisIndex >= axisSize / 2)
    return;

  // [FIX 2] Correct Cooley-Tukey Index Mapping:
  // Map the linear thread ID (0..N/2) to the correct butterfly indices.
  // "t" acts as the unique thread identifier.
  int t = axisIndex;
  int i1 = (t / halfStageSize) * stageSize + (t % halfStageSize);
  int i2 = i1 + halfStageSize;

  // The twiddle factor 'k' is the offset within the group
  int k = t % halfStageSize;

  // int group = axisIndex / stageSize;
  // int pairOffset = axisIndex % halfStageSize;
  // int k = pairOffset;

  // int i1 = group * stageSize + pairOffset;
  // int i2 = i1 + halfStageSize;

  if (i2 >= axisSize)
    return;

  IVEC_TYPE pos1 = set_axis_index(pos, i1);
  IVEC_TYPE pos2 = set_axis_index(pos, i2);

  vec2 p = imageLoad(inputTexture, pos1).xy;
  vec2 q = imageLoad(inputTexture, pos2).xy;

  vec2 w = compute_twiddle(k, stageSize);
  vec2 temp = complex_mul(q, w);

  if (abs(temp.x) < 1e-6)
    temp.x = 0.0;
  if (abs(temp.y) < 1e-6)
    temp.y = 0.0;

  imageStore(outputTexture, pos1, vec4(p + temp, 0.0, 0.0));
  imageStore(outputTexture, pos2, vec4(p - temp, 0.0, 0.0));
}