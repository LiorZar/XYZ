/// <reference path="INode.ts" />
/// <reference path="Shaders.ts" />
/// <reference path="Buffer.ts" />

class RNode implements INode {
    public name: string;
    public shader: string;
    private buffer: GLBuffer;

    constructor(name: string, shader: string, vertexData: number[], numComponents: number[]) {
        this.name = name;
        this.shader = shader;
        const buffer = new GLBuffer();
        buffer.create(new Float32Array(vertexData), numComponents, gl.STATIC_DRAW);
        this.buffer = buffer;

    }

    public draw(): void {
        shaders.use(this.shader);
        this.buffer.Draw();
    }
}