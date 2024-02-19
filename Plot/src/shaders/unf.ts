const unfBlock: string =
    `#version 300 es

precision highp float;

uniform vec2 uResolution;
uniform vec4 uGrid;
uniform float uScale;   
uniform vec2 uTranslate;

mat3 ProjectionMatrix()
{
    float rl = 1.0 / (uGrid.y - uGrid.x);
    float tb = 1.0 / (uGrid.w - uGrid.z);
    float tx = -(uGrid.y + uGrid.x) * rl;
    float ty = -(uGrid.w + uGrid.z) * tb;

    return mat3
    (
        2.0 * rl, 0.0, 0.0,
        0.0, 2.0 * tb, 0.0,
        tx, ty, 1.0
    );
}

mat3 ModelMatrix()
{
    float aspect = uResolution.x / uResolution.y;
    float sx = uScale;
    float sy = uScale;
    if (aspect > 1.f)
        sx /= aspect;
    else
        sy *= aspect;
    return mat3
    (
        sx, 0.f, 0.f,
        0.f, sy, 0.f,
        uTranslate.x, uTranslate.y, 1.f
    );
}
vec2 ModelPosition(vec2 position)
{
    return (ModelMatrix() * vec3(position, 1.f)).xy;
}
vec2 ModelProjectionPosition(vec2 position)
{
    vec3 pos = ProjectionMatrix() * ModelMatrix() * vec3(position, 1.f);
    return pos.xy / pos.z;
}

`;