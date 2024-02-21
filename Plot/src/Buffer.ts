/// <reference path="glo.ts" />

class GLBuffer {
    private type: GLenum;
    private base: GLuint;
    private buffer: WebGLBuffer | null;
    private attributeLocations: { location: number, numComponents: number, offset: number }[];
    private stride: number = 0;
    public elementCount: number = 0;

    constructor(type: GLenum = gl.ARRAY_BUFFER, base: GLuint = 0) {
        this.type = type;
        this.base = base;
        this.buffer = null;
        this.attributeLocations = [];
    }

    public create(data: Float32Array, numComponents: number[], usage: GLenum): void {
        this.attributeLocations = [];
        let location: number = 0;
        this.stride = 0;
        if (this.type === gl.ARRAY_BUFFER) {
            for (let c of numComponents) {
                this.attributeLocations.push({ location, numComponents: c, offset: this.stride });
                location++;
                this.stride += c * Float32Array.BYTES_PER_ELEMENT;
            }
        }
        const count = this.stride / Float32Array.BYTES_PER_ELEMENT;
        if (data.length % count !== 0) {
            throw new Error('Data length does not divide evenly with the number of components');
        }
        this.elementCount = data.length / count;

        this.buffer = gl.createBuffer();
        gl.bindBuffer(this.type, this.buffer);
        gl.bufferData(this.type, data, usage);
        gl.bindBuffer(this.type, null);

    }
    public destroy(): void {
        gl.deleteBuffer(this.buffer);
        this.buffer = null;
        this.attributeLocations = [];
    }
    public Draw(mode: GLenum = gl.TRIANGLE_STRIP, first: GLint = 0, count: GLsizei = -1): void {
        this.bind();
        this.draw(mode, first, count);
        this.unbind();
    }
    public bind(): void {
        const stride = this.stride;

        if (this.type === gl.ARRAY_BUFFER) {
            gl.bindBuffer(this.type, this.buffer);
            for (let loc of this.attributeLocations) {
                gl.enableVertexAttribArray(loc.location);
                gl.vertexAttribPointer(loc.location, loc.numComponents, gl.FLOAT, false, stride, loc.offset);
            }
        }
        else if (this.type === gl.ELEMENT_ARRAY_BUFFER) {
            gl.bindBuffer(this.type, this.buffer);
        }
        else if (this.type === gl.UNIFORM_BUFFER) {
            gl.bindBufferBase(this.type, this.base, this.buffer);
        }
    }
    public draw(mode: GLenum = gl.TRIANGLE_STRIP, first: GLint = 0, count: GLsizei = -1): void {
        if (count < 0)
            count = this.elementCount;
        if (first + count > this.elementCount)
            count = this.elementCount - first;

        gl.drawArrays(mode, first, count);
    }
    public unbind(): void {
        for (const { location } of this.attributeLocations) {
            gl.disableVertexAttribArray(location);
        }
        gl.bindBuffer(this.type, null);
    }


}



