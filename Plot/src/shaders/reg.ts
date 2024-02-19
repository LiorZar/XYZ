const shader_reg = {
    vert: `
in vec2 position;

void main()
{
    gl_Position = vec4(ModelProjectionPosition(position), 0.0, 1.0);
}
`,
    frag: `

uniform vec4 color;
out vec4 fragColor;

void main() 
{
    fragColor = color;
}
`
};