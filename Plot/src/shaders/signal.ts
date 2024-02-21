const shader_signal = {
    vert: `

in float sampleY;

uniform float count;
uniform vec2 uModelScale;

void main()
{
    float x = float(gl_VertexID) / (count-1.0) - 0.5;
    vec2 position = vec2(x, sampleY);
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