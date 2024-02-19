/// <reference path="INode.ts" />
/// <reference path="../Shaders.ts" />
/// <reference path="../Buffer.ts" />
/// <reference path="../Colors.ts" />


class Signal implements INode {
    public name: string;
    public shader: string;
    private buffer: GLBuffer;

    public comp: number;
    public stride: number;
    public width: number;
    public offset: number = 0;
    public scale: number = 0;
    public color: number[];

    constructor(name: string, data: ArrayBuffer, comp: number = 1, stride: number = 1, width: number = 40, color: number[] = [1, 1, 0, 1]) {
        this.name = name;
        this.shader = 'reg';
        this.comp = comp;
        this.stride = stride * comp;
        this.width = width;
        this.color = color;
        this.update(data);
    }
    public update(data: ArrayBuffer): void {
        console.log("Signal update");
        const fdata = new Float32Array(data);
        const vertexData = [];
        const { comp, stride, width } = this;
        const count = fdata.length / stride;
        const space: number = width / count;
        let x = -0.5 * width;
        for (let i = 0; i < fdata.length; i += stride) {
            vertexData.push(x, fdata[i]);
            x += space;
        }

        const buffer = new GLBuffer();
        buffer.create(new Float32Array(vertexData), [2], gl.STATIC_DRAW);
        this.buffer = buffer;

    }
    public draw(prog: GLProgram): void {
        prog.bind({ color: this.color });

        this.buffer.Draw(gl.LINE_STRIP);
    }


}