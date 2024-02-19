const shader_bk = {
    vert: `
in vec2 position;
in vec4 color;

out vec4 vColor;

void main()
{
    vColor = color;
    gl_Position = vec4(position, 0.0, 1.0);
}
`,
    frag: `
in vec4 vColor;
out vec4 fragColor;

void main() 
{
    fragColor = vColor;
}
`
};