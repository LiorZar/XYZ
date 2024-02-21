/// <reference path="INode.ts" />
/// <reference path="../Shaders.ts" />
/// <reference path="../Buffer.ts" />
/// <reference path="../Colors.ts" />

class Signal implements INode {
    public name: string;
    public shader: string;
    public func: string;
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
        this.func = 'signal';
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
        for (let i = offset; i < fdata.length; i += cstride)
            vertexData.push(fdata[i]);
        const vData = this.proccess(vertexData);
        const buffer = new GLBuffer();
        buffer.create(new Float32Array(vData), [1], gl.STATIC_DRAW);
        this.buffer = buffer;
    }
    private proccess(vertexData: number[]): number[] {
        const { func } = this;
        if (func === 'signal') {
            this.scale[1] = 1.0;
            return vertexData;
        }
        if (func === 'derivative') {
            const result: number[] = [];
            for (let i = 1; i < vertexData.length; i++)
                result.push(vertexData[i] - vertexData[i - 1]);
            this.scale[1] = 1.0;
            return result;
        }
        if (func === 'integral') {
            const result: number[] = [];
            let sum: number = 0, maxVal: number = 0;
            for (let i = 0; i < vertexData.length; i++) {
                sum += vertexData[i];
                if (Math.abs(sum) > maxVal)
                    maxVal = Math.abs(sum);
                result.push(sum);
            }
            this.scale[1] = 1.0 / maxVal;
            return result;
        }
        if (func === 'average') {
            const result: number[] = [];
            let sum: number = 0;
            for (let i = 0; i < vertexData.length; i++) {
                sum += vertexData[i];
                result.push(sum / (i + 1));
            }
            this.scale[1] = 1.0;
            return result;
        }
        if (func === 'magnitude') {
            const result: number[] = [];
            for (let i = 0; i < vertexData.length; i += 2) {
                const x = vertexData[i];
                const y = vertexData[i + 1];
                result.push(Math.sqrt(x * x + y * y));
            }
            this.scale[1] = 1.0;
            return result;
        }
        if (func === 'phase') {
            const result: number[] = [];
            for (let i = 0; i < vertexData.length; i += 2) {
                const x = vertexData[i];
                const y = vertexData[i + 1];
                result.push(Math.atan2(y, x));
            }
            this.scale[1] = 1.0;
            return result;
        }
        if (func === 'energy') {
            let e: number = 0, maxVal: number = 0;
            const result: number[] = [];
            for (let i = 0; i < vertexData.length; i += 2) {
                const x = vertexData[i];
                const y = vertexData[i + 1];
                e += x * x + y * y;
                if (e > maxVal)
                    maxVal = e;
                result.push(e);
            }
            this.scale[1] = 1.0 / maxVal;
            return result;
        }
        return vertexData;
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