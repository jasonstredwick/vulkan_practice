#version 460

layout(location=0) in vec4 in_position;
layout(location=1) in vec4 in_color;

layout(location=0) out vec4 frag_color;

void main() {
    gl_Position = vec4(in_position);
    frag_color = in_color;
}
