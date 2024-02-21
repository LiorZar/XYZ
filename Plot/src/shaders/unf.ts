const unfBlock: string =
    `#version 300 es

precision highp float;

uniform vec2 uResolution;
uniform vec4 uGrid;
uniform float uScale;   
uniform vec2 uTranslate;
uniform mat3 uModelMatrix;
uniform mat3 uProjectionMatrix;


vec2 ModelPosition(vec2 position)
{
    return (uModelMatrix * vec3(position, 1.f)).xy;
}
vec2 ModelProjectionPosition(vec2 position)
{
    vec3 pos = uProjectionMatrix * uModelMatrix * vec3(position, 1.f);
    return pos.xy / pos.z;
}

`;