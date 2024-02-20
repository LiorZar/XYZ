const shader_derivative = {
    vert: `
in vec2 position;

uniform vec2 uModelScale;

void main()
{
    gl_Position = vec4(ModelProjectionPosition(position*uModelScale), 0.0, 1.0);
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