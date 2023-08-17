#version 460
#extension GL_EXT_scalar_block_layout : enable


layout(location=0) in uint model_index;
layout(location=1) in vec3 in_position;
//layout(scalar, location=0) in uint model_index;
//layout(scalar, location=1) in vec3 in_position;

struct VertexData {
    vec4 color;
};
layout(scalar, set=0, binding=0) readonly buffer VertexDataBuffer { VertexData vertex_data[]; };
layout(scalar, set=0, binding=1) readonly buffer ModelTransformBuffer { mat4 model_transform[]; };
layout(scalar, set=0, binding=2) readonly buffer MVP { mat4 mvp[]; };

layout(location=0) out vec4 frag_color;

void main() {
    gl_Position = mvp[0] * model_transform[model_index] * vec4(in_position, 1.0);
    frag_color = vertex_data[gl_VertexIndex].color;
}
