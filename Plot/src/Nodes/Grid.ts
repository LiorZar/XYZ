/// <reference path="INode.ts" />
/// <reference path="../Shaders.ts" />
/// <reference path="../Buffer.ts" />
/// <reference path="../Colors.ts" />


class Grid implements INode {
    public name: string;
    public shader: string;
    private buffer: GLBuffer;
    private color: number[];

    constructor(name: string, count: number, color: number[] = [0.2, 0.2, 0.2, 1]) {
        this.name = name;
        this.shader = 'grid';
        this.color = color;

        const vertexData = [];
        const space: number = 1.0 / count;
        vertexData.push(0, 0.5, 1, 0.5); // Vertical lines
        vertexData.push(0.5, 0, 0.5, 1); // Horizontal lines

        for (let i = 0; i <= 1; i += space) {
            vertexData.push(0, i, 1, i); // Vertical lines
            vertexData.push(i, 0, i, 1); // Horizontal lines
        }
        const buffer = new GLBuffer();
        buffer.create(new Float32Array(vertexData), [2], gl.STATIC_DRAW);
        this.buffer = buffer;
    }

    public draw(prog: GLProgram): void {
        prog.bind({ color: this.color });
        this.buffer.Draw(gl.LINES, 4);

        prog.bind({ color: Colors.LIME });
        this.buffer.Draw(gl.LINES, 0, 4);
    }
}