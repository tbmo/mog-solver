#version 460
layout(local_size_x = 8, local_size_y = 8) in;

layout(location = 0) uniform ivec3 tensorDimensions;
layout(location = 1) uniform int stage;
layout(location = 2) uniform int direction; // 1 forward, -1 inverse
layout(location = 3) uniform int axis;      // 0 for X, 1 for Y

layout(rg32f, binding = 0) readonly uniform image2D inputTexture;
layout(rg32f, binding = 1) writeonly uniform image2D outputTexture;

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

  float phase = -direction * PI2 * float(k) / float(stageSize);
  float c = cos(phase);
  float s = sin(phase);
  if (abs(c) < 1e-6)
    c = 0.0;
  if (abs(s) < 1e-6)
    s = 0.0;
  return vec2(c, s);
}

int get_axis_index(ivec2 pos) { return (axis == 0) ? pos.x : pos.y; }

ivec2 set_axis_index(ivec2 pos, int idx) {
  if (axis == 0)
    return ivec2(idx, pos.y);
  return ivec2(pos.x, idx);
}

int get_axis_size() {
  return (axis == 0) ? tensorDimensions.x : tensorDimensions.y;
}

uint bit_reverse(uint x, uint n) {
  uint result = 0;
  for (uint i = 0; i < n; ++i) {
    result = (result << 1) | (x & 1);
    x >>= 1;
  }
  return result;
}

void main() {
  ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
  if (any(greaterThanEqual(pos, tensorDimensions.xy)))
    return;

  int axisSize = get_axis_size();
  int axisIndex = get_axis_index(pos);
  if (axisIndex >= axisSize)
    return;

  uint n = uint(log2(float(axisSize)));

  // Stage 0: Bit reversal
  if (stage == 0) {
    uint revIndex = bit_reverse(uint(axisIndex), n);
    ivec2 revPos = set_axis_index(pos, int(revIndex));
    vec2 val = imageLoad(inputTexture, pos).xy;
    imageStore(outputTexture, revPos, vec4(val, 0.0, 0.0));
    return;
  }

  // Butterfly stages
  int stageSize = 1 << stage;
  int halfStageSize = stageSize >> 1;

  int group = axisIndex / stageSize;
  int pairOffset = axisIndex % halfStageSize;
  int k = pairOffset;

  int i1 = group * stageSize + pairOffset;
  int i2 = i1 + halfStageSize;

  if (i2 >= axisSize)
    return;

  ivec2 pos1 = set_axis_index(pos, i1);
  ivec2 pos2 = set_axis_index(pos, i2);

  vec2 p = imageLoad(inputTexture, pos1).xy;
  vec2 q = imageLoad(inputTexture, pos2).xy;

  vec2 w = compute_twiddle(k, stageSize);
  vec2 temp = complex_mul(q, w);

  if (abs(temp.x) < 1e-6)
    temp.x = 0.0;
  if (abs(temp.y) < 1e-6)
    temp.y = 0.0;

  vec2 P = p + temp;
  vec2 Q = p - temp;

  imageStore(outputTexture, pos1, vec4(P, 0.0, 0.0));
  imageStore(outputTexture, pos2, vec4(Q, 0.0, 0.0));
}