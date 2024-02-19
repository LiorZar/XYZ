const shader_grid = {
    vert: `
in vec2 position;
void main()
{
    vec4 grid = uGrid*2.0;
    vec2 pos = vec2(mix(grid.x, grid.y, position.x), mix(grid.z, grid.w, position.y));
    gl_Position = vec4(ModelProjectionPosition(pos), 0.0, 1.0);
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