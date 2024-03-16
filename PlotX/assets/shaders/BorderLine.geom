//--------------------------------------------------------------------------------------------------------------------//
#include "version430.glsl"
#include "render.h"
//--------------------------------------------------------------------------------------------------------------------//
layout(points) in;
layout(line_strip, max_vertices=5) out;
//--------------------------------------------------------------------------------------------------------------------//
flat out int vert;
//--------------------------------------------------------------------------------------------------------------------//
void drawBorder()
{
	const float Z = 0.005;
	vec4  rect = inp.params.rectangle;
	const vec2 T = vec2(inp.params.thickness*0.5);
	rect.xy += T;
	rect.zw -= T;
	
	vert = 1;
	vec2 pos = rect.xy;	
	gl_Position = gl_ModelViewProjectionMatrix * vec4(pos,Z,1);
	EmitVertex();
	
	vert = 0;
	pos = rect.zy;	
	gl_Position = gl_ModelViewProjectionMatrix * vec4(pos,Z,1);
	EmitVertex();
		
	vert = 1;
	pos = rect.zw;	
	gl_Position = gl_ModelViewProjectionMatrix * vec4(pos,Z,1);
	EmitVertex();
	
	vert = 0;
	pos = rect.xw;	
	gl_Position = gl_ModelViewProjectionMatrix * vec4(pos,Z,1);
	EmitVertex();
	
	vert = 1;
	pos = rect.xy;	
	gl_Position = gl_ModelViewProjectionMatrix * vec4(pos,Z,1);
	EmitVertex();
	
	EndPrimitive();	
}
//--------------------------------------------------------------------------------------------------------------------//
void main()
{		
   drawBorder();
} 
//--------------------------------------------------------------------------------------------------------------------//