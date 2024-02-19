/// <reference path="INode.ts" />
/// <reference path="../Shaders.ts" />
/// <reference path="../Buffer.ts" />

class Lines implements INode {
    public name: string;
    public shader: string;
    private buffer: GLBuffer;
    private color: number[];

    constructor(name: string, shader: string, vertexData: number[], color: number[] = [0.2, 1, 0.2, 1]) {
        this.name = name;
        this.shader = shader;
        this.color = color;
        const buffer = new GLBuffer();
        buffer.create(new Float32Array(vertexData), [2], gl.STATIC_DRAW);
        this.buffer = buffer;
    }
    public draw(prog: GLProgram): void {
        prog.bind({ color: this.color });
        this.buffer.Draw(gl.LINES);
    }
}