const unfBlock: string =
    `#version 300 es

precision highp float;

uniform vec2 uResolution;
uniform vec4 uGrid;
uniform float uScale;   
uniform vec2 uTranslate;
// unfiorm mat3 uModelMatrix;
uniform mat3 uProjectionMatrix;

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
    vec3 pos = uProjectionMatrix * ModelMatrix() * vec3(position, 1.f);
    return pos.xy / pos.z;
}

`;