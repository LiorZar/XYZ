/// <reference path="INode.ts" />
/// <reference path="../Shaders.ts" />
/// <reference path="../Buffer.ts" />
/// <reference path="../Colors.ts" />


class Signal implements INode {
    public name: string;
    public shader: string;
    private buffer: GLBuffer;
    private rawData: Float32Array;

    public comp: number;
    public stride: number;
    public offset: number = 0;
    public scale: number[] = [40, 1];
    public color: number[];
    public enabled: boolean = true;

    constructor(name: string, data: ArrayBuffer, comp: number = 1, stride: number = 1, width: number = 40, color: number[] = [1, 1, 0, 1]) {
        this.name = name;
        this.shader = 'signal';
        this.comp = comp;
        this.stride = stride * comp;
        this.scale = [width, 1];
        this.color = color;
        this.update(data);
    }
    public update(data: ArrayBuffer): void {
        console.log("Signal update");
        this.rawData = new Float32Array(data);
        this.recreate();
    }
    public recreate(): void {
        const fdata = this.rawData;
        const vertexData = [];
        const { comp, stride, offset } = this;
        const cstride = comp * stride;
        const count = fdata.length / cstride;
        const space: number = 1.0 / count;
        let x = -0.5;
        for (let i = offset; i < fdata.length; i += cstride) {
            vertexData.push(fdata[i]);
            x += space;
        }

        const buffer = new GLBuffer();
        buffer.create(new Float32Array(vertexData), [1], gl.STATIC_DRAW);
        this.buffer = buffer;

    }
    public draw(prog: GLProgram): void {
        if (!this.enabled)
            return;
        prog.bind({
            color: this.color,
            count: this.buffer.elementCount,
            uModelScale: this.scale
        });

        this.buffer.Draw(gl.LINE_STRIP);
    }


}