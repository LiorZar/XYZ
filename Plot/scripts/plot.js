"use strict";
/// <reference path="Nodes/INode.ts" />
console.clear();
const canvasDiv = document.getElementById('__canvas__');
const gl = canvasDiv.getContext('webgl2');
gl.clearColor(0, 0, 0, 1);
gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
class Canvas {
    constructor() {
        this.layers = [];
        this.clearColor = [0, 0, 0, 1];
    }
    getNode(layerName, nodeName) {
        const layer = this.layers.find(layer => layer.name === layerName);
        if (layer)
            return layer.nodes.find(node => node.name === nodeName);
    }
    addNode(layerName, node) {
        const layer = this.layers.find(layer => layer.name === layerName);
        if (!layer)
            this.layers.push({ name: layerName, nodes: [node] });
        else {
            const index = layer.nodes.findIndex(n => n.name === node.name);
            if (index !== -1)
                layer.nodes.splice(index, 1);
            layer.nodes.push(node);
        }
    }
    remNode(layerName, nodeName) {
        const layer = this.layers.find(layer => layer.name === layerName);
        if (layer) {
            const index = layer.nodes.findIndex(n => n.name === nodeName);
            if (index !== -1)
                layer.nodes.splice(index, 1);
        }
    }
    setClearColor(r, g, b, a) {
        this.clearColor = [r, g, b, a];
        gl.clearColor(this.clearColor[0], this.clearColor[1], this.clearColor[2], this.clearColor[3]);
    }
    clear() {
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    }
}
;
const canvas = new Canvas();
class Mat3 {
    constructor() {
        this.data = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    }
    fromArray(data) {
        for (let i = 0; i < 9; i++) {
            this.data[i] = data[i];
        }
    }
    Set(row, column, value) {
        this.data[row * 3 + column] = value;
    }
    Get(row, column) {
        return this.data[row * 3 + column];
    }
    add(matrix) {
        const result = new Mat3();
        for (let i = 0; i < 9; i++) {
            result.Set(Math.floor(i / 3), i % 3, this.Get(Math.floor(i / 3), i % 3) + matrix.Get(Math.floor(i / 3), i % 3));
        }
        return result;
    }
    sub(matrix) {
        const result = new Mat3();
        for (let i = 0; i < 9; i++) {
            result.Set(Math.floor(i / 3), i % 3, this.Get(Math.floor(i / 3), i % 3) - matrix.Get(Math.floor(i / 3), i % 3));
        }
        return result;
    }
    mul(matrix) {
        const result = new Mat3();
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                let sum = 0;
                for (let k = 0; k < 3; k++) {
                    sum += this.Get(i, k) * matrix.Get(k, j);
                }
                result.Set(i, j, sum);
            }
        }
        return result;
    }
    mulScalar(scalar) {
        const result = new Mat3();
        for (let i = 0; i < 9; i++) {
            result.Set(Math.floor(i / 3), i % 3, this.Get(Math.floor(i / 3), i % 3) * scalar);
        }
        return result;
    }
    transformPoint(point) {
        const x = this.Get(0, 0) * point[0] + this.Get(0, 1) * point[1] + this.Get(0, 2);
        const y = this.Get(1, 0) * point[0] + this.Get(1, 1) * point[1] + this.Get(1, 2);
        return [x, y];
    }
    transpose() {
        const result = new Mat3();
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                result.Set(i, j, this.Get(j, i));
            }
        }
        return result;
    }
    determinant() {
        return this.Get(0, 0) * (this.Get(1, 1) * this.Get(2, 2) - this.Get(1, 2) * this.Get(2, 1)) -
            this.Get(0, 1) * (this.Get(1, 0) * this.Get(2, 2) - this.Get(1, 2) * this.Get(2, 0)) +
            this.Get(0, 2) * (this.Get(1, 0) * this.Get(2, 1) - this.Get(1, 1) * this.Get(2, 0));
    }
    inverse() {
        const result = new Mat3();
        const det = this.determinant();
        if (det === 0) {
            throw new Error("Matrix is not invertible");
        }
        const invDet = 1 / det;
        result.Set(0, 0, (this.Get(1, 1) * this.Get(2, 2) - this.Get(1, 2) * this.Get(2, 1)) * invDet);
        result.Set(0, 1, (this.Get(0, 2) * this.Get(2, 1) - this.Get(0, 1) * this.Get(2, 2)) * invDet);
        result.Set(0, 2, (this.Get(0, 1) * this.Get(1, 2) - this.Get(0, 2) * this.Get(1, 1)) * invDet);
        result.Set(1, 0, (this.Get(1, 2) * this.Get(2, 0) - this.Get(1, 0) * this.Get(2, 2)) * invDet);
        result.Set(1, 1, (this.Get(0, 0) * this.Get(2, 2) - this.Get(0, 2) * this.Get(2, 0)) * invDet);
        result.Set(1, 2, (this.Get(0, 2) * this.Get(1, 0) - this.Get(0, 0) * this.Get(1, 2)) * invDet);
        result.Set(2, 0, (this.Get(1, 0) * this.Get(2, 1) - this.Get(1, 1) * this.Get(2, 0)) * invDet);
        result.Set(2, 1, (this.Get(0, 1) * this.Get(2, 0) - this.Get(0, 0) * this.Get(2, 1)) * invDet);
        result.Set(2, 2, (this.Get(0, 0) * this.Get(1, 1) - this.Get(0, 1) * this.Get(1, 0)) * invDet);
        return result;
    }
    translate(x, y) {
        const translation = Mat3.translation(x, y);
        this.fromArray(this.mul(translation).data);
    }
    scale(x, y) {
        const scale = Mat3.scale(x, y);
        this.fromArray(this.mul(scale).data);
    }
    static identity() {
        const result = new Mat3();
        result.Set(0, 0, 1);
        result.Set(1, 1, 1);
        result.Set(2, 2, 1);
        return result;
    }
    static translation(x, y) {
        const result = Mat3.identity();
        result.Set(0, 2, x);
        result.Set(1, 2, y);
        return result;
    }
    static scale(x, y) {
        const result = Mat3.identity();
        result.Set(0, 0, x);
        result.Set(1, 1, y);
        return result;
    }
}
/// <reference path="Mat3.ts" />
class GLProgram {
    constructor() {
        this.program = null;
    }
    create(vertexShaderSource, fragmentShaderSource) {
        // Create vertex shader
        const vertexShader = gl.createShader(gl.VERTEX_SHADER);
        gl.shaderSource(vertexShader, vertexShaderSource);
        gl.compileShader(vertexShader);
        // Create fragment shader
        const fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
        gl.shaderSource(fragmentShader, fragmentShaderSource);
        gl.compileShader(fragmentShader);
        // Create shader program
        const program = gl.createProgram();
        gl.attachShader(program, vertexShader);
        gl.attachShader(program, fragmentShader);
        gl.linkProgram(program);
        // Check for errors
        if (!gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
            console.error('Vertex shader compilation error:', gl.getShaderInfoLog(vertexShader));
            return false;
        }
        if (!gl.getShaderParameter(fragmentShader, gl.COMPILE_STATUS)) {
            console.error('Fragment shader compilation error:', gl.getShaderInfoLog(fragmentShader));
            return false;
        }
        if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
            console.error('Shader program linking error:', gl.getProgramInfoLog(program));
            return false;
        }
        // Store the program
        this.program = program;
        this.uniforms = new Map();
        const numUniforms = gl.getProgramParameter(this.program, gl.ACTIVE_UNIFORMS);
        for (let i = 0; i < numUniforms; i++) {
            const uniformInfo = gl.getActiveUniform(this.program, i);
            const location = gl.getUniformLocation(this.program, uniformInfo.name);
            this.uniforms.set(uniformInfo.name, location); // Fix: Use set() method instead of push()
        }
        return true;
    }
    use() {
        gl.useProgram(this.program);
    }
    getProgram() {
        return this.program;
    }
    bind(data) {
        for (let key in data) {
            const value = data[key];
            if (value instanceof Mat3) {
                this.uniformMatrix3fv(key, true, value.data);
            }
            else if (Array.isArray(value)) {
                switch (value.length) {
                    case 1:
                        this.uniform1f(key, value[0]);
                        break;
                    case 2:
                        this.uniform2f(key, value[0], value[1]);
                        break;
                    case 3:
                        this.uniform3f(key, value[0], value[1], value[2]);
                        break;
                    case 4:
                        this.uniform4f(key, value[0], value[1], value[2], value[3]);
                        break;
                    case 9:
                        this.uniformMatrix3fv(key, true, value);
                        break;
                    case 16:
                        this.uniformMatrix4fv(key, true, value);
                        break;
                }
            }
            else
                this.uniform1f(key, value);
        }
    }
    uniform1f(name, value) {
        const location = this.uniforms.get(name);
        if (location !== undefined)
            gl.uniform1f(location, value);
    }
    uniform2f(name, value1, value2) {
        const location = this.uniforms.get(name);
        if (location !== undefined)
            gl.uniform2f(location, value1, value2);
    }
    uniform3f(name, value1, value2, value3) {
        const location = this.uniforms.get(name);
        if (location !== undefined)
            gl.uniform3f(location, value1, value2, value3);
    }
    uniform4f(name, value1, value2, value3, value4) {
        const location = this.uniforms.get(name);
        if (location !== undefined)
            gl.uniform4f(location, value1, value2, value3, value4);
    }
    uniform1i(name, value) {
        const location = this.uniforms.get(name);
        if (location !== undefined)
            gl.uniform1i(location, value);
    }
    uniform2i(name, value1, value2) {
        const location = this.uniforms.get(name);
        if (location !== undefined)
            gl.uniform2i(location, value1, value2);
    }
    uniform3i(name, value1, value2, value3) {
        const location = this.uniforms.get(name);
        if (location !== undefined)
            gl.uniform3i(location, value1, value2, value3);
    }
    uniform4i(name, value1, value2, value3, value4) {
        const location = this.uniforms.get(name);
        if (location !== undefined)
            gl.uniform4i(location, value1, value2, value3, value4);
    }
    uniformMatrix3fv(name, transpose, value) {
        const location = this.uniforms.get(name);
        if (location !== undefined)
            gl.uniformMatrix3fv(location, transpose, new Float32Array(value));
    }
    uniformMatrix4fv(name, transpose, value) {
        const location = this.uniforms.get(name);
        if (location !== undefined)
            gl.uniformMatrix4fv(location, transpose, new Float32Array(value));
    }
}
const unfBlock = `#version 300 es

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
const shader_reg = {
    vert: `
in vec2 position;

void main()
{
    gl_Position = vec4(ModelProjectionPosition(position), 0.0, 1.0);
}
`,
    frag: `

uniform vec4 color;
out vec4 fragColor;

void main() 
{
    fragColor = color;
}
`
};
const shader_regc = {
    vert: `
in vec2 position;
in vec4 color;

out vec4 vColor;

void main()
{
    vColor = color;
    gl_Position = vec4(position, 0.0, 1.0);
}
`,
    frag: `
in vec4 vColor;
out vec4 fragColor;

void main() 
{
    fragColor = vColor;
}
`
};
const shader_grid = {
    vert: `
in vec2 position;
void main()
{
    vec4 grid = uGrid*2.0;
    vec2 pos = vec2(mix(grid.x, grid.y, position.x), mix(grid.z, grid.w, position.y));
    gl_Position = vec4(ModelProjectionPosition(pos), 0.0, 1.0);
}
`,
    frag: `

uniform vec4 color;
out vec4 fragColor;

void main() 
{
    fragColor = color;
}
`
};
const shader_signal = {
    vert: `

in float amplitude;

uniform float count;
uniform vec2 uModelScale;

void main()
{
    float x = float(gl_VertexID) / (count-1.0) - 0.5;
    vec2 position = vec2(x, amplitude);
    gl_Position = vec4(ModelProjectionPosition(position*uModelScale), 0.0, 1.0);
}
`,
    frag: `

uniform vec4 color;
out vec4 fragColor;

void main() 
{
    fragColor = color;
}
`
};
const shader_energy = {
    vert: `

in float amplitude;

uniform float count;
uniform vec2 uModelScale;

void main()
{
    float x = float(gl_VertexID) / (count-1.0) - 0.5;
    vec2 position = vec2(x, amplitude);
    gl_Position = vec4(ModelProjectionPosition(position*uModelScale), 0.0, 1.0);
}
`,
    frag: `

uniform vec4 color;
out vec4 fragColor;

void main() 
{
    fragColor = color;
}
`
};
const shader_derivative = {
    vert: `

in float amplitude;

uniform float count;
uniform vec2 uModelScale;

void main()
{
    float x = float(gl_VertexID) / (count-1.0) - 0.5;
    vec2 position = vec2(x, amplitude);
    gl_Position = vec4(ModelProjectionPosition(position*uModelScale), 0.0, 1.0);
}
`,
    frag: `

uniform vec4 color;
out vec4 fragColor;

void main() 
{
    fragColor = color;
}
`
};
const shader_magnitude = {
    vert: `

in float amplitude;

uniform float count;
uniform vec2 uModelScale;

void main()
{
    float x = float(gl_VertexID) / (count-1.0) - 0.5;
    vec2 position = vec2(x, amplitude);
    gl_Position = vec4(ModelProjectionPosition(position*uModelScale), 0.0, 1.0);
}
`,
    frag: `

uniform vec4 color;
out vec4 fragColor;

void main() 
{
    fragColor = color;
}
`
};
/// <reference path="Program.ts" />
/// <reference path="shaders/unf.ts" />
/// <reference path="shaders/reg.ts" />
/// <reference path="shaders/regc.ts" />
/// <reference path="shaders/grid.ts" />
/// <reference path="shaders/signal.ts" />
/// <reference path="shaders/energy.ts" />
/// <reference path="shaders/derivative.ts" />
/// <reference path="shaders/magnitude.ts" />
class ShaderMap {
    constructor() {
        this.shaders = new Map();
    }
    addProgram(name, vertexShader, fragmentShader) {
        const program = new GLProgram();
        if (false == program.create(unfBlock + vertexShader, unfBlock + fragmentShader))
            return false;
        this.shaders.set(name, program);
        return true;
    }
    getProgram(name) {
        return this.shaders.get(name);
    }
    use(name) {
        const program = this.shaders.get(name);
        if (program)
            program.use();
        return program;
    }
}
;
const shaders = new ShaderMap();
shaders.addProgram('reg', shader_reg.vert, shader_reg.frag);
shaders.addProgram('regc', shader_regc.vert, shader_regc.frag);
shaders.addProgram('grid', shader_grid.vert, shader_grid.frag);
shaders.addProgram('signal', shader_signal.vert, shader_signal.frag);
shaders.addProgram('energy', shader_energy.vert, shader_energy.frag);
shaders.addProgram('derivative', shader_derivative.vert, shader_derivative.frag);
shaders.addProgram('magnitude', shader_magnitude.vert, shader_magnitude.frag);
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
const wnd = window;
class FileSystemAccess {
    constructor() {
        this.directoryHandle = null;
        this.files = new Map();
    }
    // Requests directory access from the user
    RequestDirectoryAccess() {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                this.directoryHandle = yield wnd.showDirectoryPicker();
                console.log("Directory access granted");
            }
            catch (error) {
                console.error("Directory access denied", error);
            }
        });
    }
    ListenToFile(fileName, callback) {
        this.files.set(fileName, 0);
        this.watchFile(fileName, callback);
    }
    // Listens to changes in a file
    watchFile(fileName, callback) {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                if (!this.directoryHandle)
                    throw new Error("Directory access not granted");
                while (true) {
                    if (!this.files.has(fileName))
                        break;
                    const lastModified = this.files.get(fileName);
                    const fileHandle = yield this.directoryHandle.getFileHandle(fileName, { create: false });
                    const file = yield fileHandle.getFile();
                    if (file.size <= 0)
                        continue;
                    if (lastModified === file.lastModified)
                        continue;
                    const contents = yield file.arrayBuffer();
                    this.files.set(fileName, file.lastModified);
                    callback(contents);
                }
            }
            catch (error) {
                console.error(`Error watching file = ${fileName}`, error);
            }
            console.log(`Stopped watching file = ${fileName} `);
        });
    }
    // Reads a file from the directory
    readFile(fileName) {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                if (!this.directoryHandle) {
                    throw new Error("Directory access not granted");
                }
                const fileHandle = yield this.directoryHandle.getFileHandle(fileName, { create: false });
                const file = yield fileHandle.getFile();
                const contents = yield file.text();
                return contents;
            }
            catch (error) {
                console.error("Error reading file", error);
                return null;
            }
        });
    }
    // Writes to a file in the directory
    writeFile(fileName, contents) {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                if (!this.directoryHandle) {
                    throw new Error("Directory access not granted");
                }
                const fileHandle = yield this.directoryHandle.getFileHandle(fileName, { create: true });
                const writableStream = yield fileHandle.createWritable();
                yield writableStream.write(contents);
                yield writableStream.close();
                console.log("File written successfully");
            }
            catch (error) {
                console.error("Error writing file", error);
            }
        });
    }
}
const fs = new FileSystemAccess();
class glo {
    static Set(obj, key, value) {
        obj[key] = value;
    }
    static ToResL(val, res) {
        return Math.floor(val / res) * res;
    }
    static ToResH(val, res) {
        return Math.ceil(val / res) * res;
    }
    static AddRes(val, add, res) {
        if (add > 0)
            return this.ToResH(val + add, res);
        return this.ToResL(val + add, res);
    }
    static MulRes(val, mul, res) {
        if (mul > 1)
            return this.ToResH(val * mul, res);
        return this.ToResL(val * mul, res);
    }
    static ToRGBHex(numbers) {
        const r = numbers[0] * 255.0;
        const g = numbers[1] * 255.0;
        const b = numbers[2] * 255.0;
        let hex = ((r << 16) | (g << 8) | b).toString(16);
        while (hex.length < 6)
            hex = '0' + hex;
        return '#' + hex;
    }
    static HexToRGB(hex) {
        const r = parseInt(hex.substring(1, 3), 16) / 255.0;
        const g = parseInt(hex.substring(3, 5), 16) / 255.0;
        const b = parseInt(hex.substring(5, 7), 16) / 255.0;
        return [r, g, b, 1];
    }
    static lerp(a, b, t) {
        return a + (b - a) * t; // 0..1
    }
    static lerpZ(a, b, t) {
        return a + (b - a) * (t + 1) * 0.5; // 0..1 -> -1..1
    }
}
glo.bal = {};
/// <reference path="glo.ts" />
class GLBuffer {
    constructor(type = gl.ARRAY_BUFFER, base = 0) {
        this.stride = 0;
        this.elementCount = 0;
        this.type = type;
        this.base = base;
        this.buffer = null;
        this.attributeLocations = [];
    }
    create(data, numComponents, usage) {
        this.attributeLocations = [];
        let location = 0;
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
    destroy() {
        gl.deleteBuffer(this.buffer);
        this.buffer = null;
        this.attributeLocations = [];
    }
    Draw(mode = gl.TRIANGLE_STRIP, first = 0, count = -1) {
        this.bind();
        this.draw(mode, first, count);
        this.unbind();
    }
    bind() {
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
    draw(mode = gl.TRIANGLE_STRIP, first = 0, count = -1) {
        if (count < 0)
            count = this.elementCount;
        if (first + count > this.elementCount)
            count = this.elementCount - first;
        gl.drawArrays(mode, first, count);
    }
    unbind() {
        for (const { location } of this.attributeLocations) {
            gl.disableVertexAttribArray(location);
        }
        gl.bindBuffer(this.type, null);
    }
}
/// <reference path="INode.ts" />
/// <reference path="../Shaders.ts" />
/// <reference path="../Buffer.ts" />
class RNode {
    constructor(name, shader, vertexData, numComponents) {
        this.name = name;
        this.shader = shader;
        const buffer = new GLBuffer();
        buffer.create(new Float32Array(vertexData), numComponents, gl.STATIC_DRAW);
        this.buffer = buffer;
    }
    draw(prog) {
        this.buffer.Draw();
    }
}
class Colors {
}
Colors.RED = [1, 0, 0, 1];
Colors.GREEN = [0, 1, 0, 1];
Colors.BLUE = [0, 0, 1, 1];
Colors.WHITE = [1, 1, 1, 1];
Colors.BLACK = [0, 0, 0, 1];
Colors.YELLOW = [1, 1, 0, 1];
Colors.CYAN = [0, 1, 1, 1];
Colors.MAGENTA = [1, 0, 1, 1];
Colors.ORANGE = [1, 0.5, 0, 1];
Colors.GRAY = [0.5, 0.5, 0.5, 1];
Colors.BROWN = [0.5, 0.25, 0, 1];
Colors.PINK = [1, 0.5, 0.5, 1];
Colors.PURPLE = [0.5, 0, 0.5, 1];
Colors.LIME = [0.5, 1, 0, 1];
Colors.TEAL = [0, 0.5, 0.5, 1];
Colors.OLIVE = [0.5, 0.5, 0, 1];
Colors.MAROON = [0.5, 0, 0, 1];
Colors.NAVY = [0, 0, 0.5, 1];
Colors.INDIGO = [0, 0.5, 1, 1];
Colors.TURQUOISE = [0, 1, 0.5, 1];
Colors.LAVENDER = [0.5, 0, 1, 1];
Colors.BEIGE = [1, 1, 0.5, 1];
Colors.MINT = [0.5, 1, 0.5, 1];
Colors.APRICOT = [1, 0.5, 0, 1];
Colors.LILAC = [0.5, 0, 1, 1];
Colors.PEAR = [1, 0.5, 1, 1];
Colors.COBALT = [0, 0.5, 1, 1];
Colors.CERULEAN = [0, 1, 0.5, 1];
Colors.AQUAMARINE = [0.5, 1, 1, 1];
Colors.SALMON = [1, 0.5, 0.5, 1];
Colors.TANGERINE = [1, 0.5, 0, 1];
Colors.LEMON = [1, 1, 0, 1];
/// <reference path="INode.ts" />
/// <reference path="../Shaders.ts" />
/// <reference path="../Buffer.ts" />
/// <reference path="../Colors.ts" />
class Grid {
    constructor(name, count, color = [0.2, 0.2, 0.2, 1]) {
        this.name = name;
        this.shader = 'grid';
        this.color = color;
        const vertexData = [];
        const space = 1.0 / count;
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
    draw(prog) {
        prog.bind({ color: this.color });
        this.buffer.Draw(gl.LINES, 4);
        prog.bind({ color: Colors.LIME });
        this.buffer.Draw(gl.LINES, 0, 4);
    }
}
/// <reference path="INode.ts" />
/// <reference path="../Shaders.ts" />
/// <reference path="../Buffer.ts" />
class Quad {
    constructor(name, shader, vertexData, color = [1, 0.2, 0.2, 1]) {
        this.name = name;
        this.shader = shader;
        this.color = color;
        const buffer = new GLBuffer();
        buffer.create(new Float32Array(vertexData), [2], gl.STATIC_DRAW);
        this.buffer = buffer;
    }
    draw(prog) {
        prog.bind({ color: this.color });
        this.buffer.Draw();
    }
}
/// <reference path="INode.ts" />
/// <reference path="../Shaders.ts" />
/// <reference path="../Buffer.ts" />
class Lines {
    constructor(name, shader, vertexData, color = [0.2, 1, 0.2, 1]) {
        this.name = name;
        this.shader = shader;
        this.color = color;
        const buffer = new GLBuffer();
        buffer.create(new Float32Array(vertexData), [2], gl.STATIC_DRAW);
        this.buffer = buffer;
    }
    draw(prog) {
        prog.bind({ color: this.color });
        this.buffer.Draw(gl.LINE_STRIP);
    }
}
/// <reference path="INode.ts" />
/// <reference path="../Shaders.ts" />
/// <reference path="../Buffer.ts" />
/// <reference path="../Colors.ts" />
class Signal {
    constructor(name, data, comp = 1, stride = 1, width = 40, color = [1, 1, 0, 1]) {
        this.offset = 0;
        this.scale = [40, 1];
        this.enabled = true;
        this.name = name;
        this.shader = 'signal';
        this.func = 'signal';
        this.comp = comp;
        this.stride = stride * comp;
        this.scale = [width, 1];
        this.color = color;
        this.update(data);
    }
    update(data) {
        console.log("Signal update");
        this.rawData = new Float32Array(data);
        this.recreate();
    }
    recreate() {
        const fdata = this.rawData;
        const vertexData = [];
        const { comp, stride, offset } = this;
        const cstride = comp * stride;
        const count = fdata.length / cstride;
        const space = 1.0 / count;
        for (let i = offset; i < fdata.length; i += cstride)
            vertexData.push(fdata[i]);
        const vData = this.proccess(vertexData);
        const buffer = new GLBuffer();
        buffer.create(new Float32Array(vData), [1], gl.STATIC_DRAW);
        this.buffer = buffer;
    }
    proccess(vertexData) {
        const { func } = this;
        if (func === 'signal') {
            this.scale[1] = 1.0;
            return vertexData;
        }
        if (func === 'derivative') {
            const result = [];
            for (let i = 1; i < vertexData.length; i++)
                result.push(vertexData[i] - vertexData[i - 1]);
            this.scale[1] = 1.0;
            return result;
        }
        if (func === 'integral') {
            const result = [];
            let sum = 0, maxVal = 0;
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
            const result = [];
            let sum = 0;
            for (let i = 0; i < vertexData.length; i++) {
                sum += vertexData[i];
                result.push(sum / (i + 1));
            }
            this.scale[1] = 1.0;
            return result;
        }
        if (func === 'magnitude') {
            const result = [];
            for (let i = 0; i < vertexData.length; i += 2) {
                const x = vertexData[i];
                const y = vertexData[i + 1];
                result.push(Math.sqrt(x * x + y * y));
            }
            this.scale[1] = 1.0;
            return result;
        }
        if (func === 'phase') {
            const result = [];
            for (let i = 0; i < vertexData.length; i += 2) {
                const x = vertexData[i];
                const y = vertexData[i + 1];
                result.push(Math.atan2(y, x));
            }
            this.scale[1] = 1.0;
            return result;
        }
        if (func === 'energy') {
            let e = 0, maxVal = 0;
            const result = [];
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
    draw(prog) {
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
/// <reference path="Canvas.ts" />
/// <reference path="Shaders.ts" />
/// <reference path="FileSystemAccess.ts" />
/// <reference path="Nodes/RNode.ts" />
/// <reference path="Nodes/Grid.ts" />
/// <reference path="Nodes/Quad.ts" />
/// <reference path="Nodes/Lines.ts" />
/// <reference path="Nodes/Signal.ts" />
const RESOLUTION = 0.01;
const MOVE_STEP = 0.1;
class App {
    constructor() {
        this.unfData = {
            uResolution: [0.0, 0.0],
            uGrid: [-10.0, 10.0, -10.0, 10.0],
            uScale: 1.0,
            uTranslate: [0.0, 0.0],
            uModelMatrix: Mat3.identity(),
            uProjectionMatrix: Mat3.identity(),
            uModelProjectionMatrix: Mat3.identity(),
            uInverseModelProjectionMatrix: Mat3.identity()
        };
        this.signalBox = document.getElementById("signalBox");
        this.shaderBox = document.getElementById("shaderBox");
        this.signalColor = document.getElementById("signalColor");
        this.enableCheckbox = document.getElementById("enableCheckbox");
        this.spinScale = document.getElementById("spinScale");
        this.spinWidth = document.getElementById("spinWidth");
        this.spinOffset = document.getElementById("spinOffset");
        this.spinComp = document.getElementById("spinComp");
        this.spinStride = document.getElementById("spinStride");
        this.aspect = 1;
        this.scaleX = 1;
        this.scaleY = 1;
        this.isMouseDown = false;
        this.gridMouse = [0, 0];
        canvasDiv.addEventListener("mousedown", this.onMouseDown.bind(this));
        canvasDiv.addEventListener("mousemove", this.onMouseMove.bind(this));
        canvasDiv.addEventListener("mouseup", this.onMouseUp.bind(this));
        canvasDiv.addEventListener("wheel", this.onMouseWheel.bind(this));
        window.addEventListener("keypress", this.onKeyDown.bind(this));
        const grid = new Grid("grid", 40);
        canvas.addNode("bk", grid);
        canvas.addNode("elems", new Lines("node3", "reg", [-1, 1, 2, 5, 3, 8]));
        canvas.addNode("elems", new Quad("node2", "reg", [3.3, 0, 4.5, 0, 3.3, 1.2, 4.5, 1.2]));
        this.signalBox.innerHTML = "";
    }
    resizeCanvasToDisplaySize(force = false) {
        const { unfData } = this;
        const width = canvasDiv.clientWidth;
        const height = canvasDiv.clientHeight;
        if (!(force || canvasDiv.width !== width || canvasDiv.height !== height))
            return false;
        canvasDiv.width = width;
        canvasDiv.height = height;
        gl.viewport(0, 0, width, height);
        unfData.uResolution[0] = width;
        unfData.uResolution[1] = height;
        const aspectRatio = width / height;
        this.aspect = aspectRatio;
        let sx = 1.0, sy = 1.0;
        if (aspectRatio > 1)
            sx /= aspectRatio;
        else
            sy *= aspectRatio;
        const rl = sx / (unfData.uGrid[1] - unfData.uGrid[0]);
        const tb = sy / (unfData.uGrid[3] - unfData.uGrid[2]);
        const tx = sx * -(unfData.uGrid[1] + unfData.uGrid[0]) * rl;
        const ty = sy * -(unfData.uGrid[3] + unfData.uGrid[2]) * tb;
        unfData.uProjectionMatrix.fromArray([2.0 * rl, 0.0, 0.0, 0.0, 2.0 * tb, 0.0, tx, ty, 1.0]);
        this.refreshMVP();
    }
    drawScene() {
        this.resizeCanvasToDisplaySize();
        canvas.clear();
        let shaderName = "";
        let program;
        for (const layer of canvas.layers) {
            for (const node of layer.nodes) {
                if (shaderName !== node.shader) {
                    shaderName = node.shader;
                    program = shaders.use(shaderName);
                    program.bind(this.unfData);
                }
                node.draw(program);
            }
        }
    }
    Scale(factor) {
        if (factor < 0)
            return;
        this.unfData.uModelMatrix.scale(factor, factor);
        this.refreshMVP();
    }
    Translate(x, y) {
        this.unfData.uModelMatrix.translate(x, y);
        this.refreshMVP();
    }
    resetModelMatrix() {
        const { unfData } = this;
        unfData.uModelMatrix.fromArray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        this.refreshMVP();
    }
    refreshMVP() {
        const { unfData } = this;
        unfData.uModelProjectionMatrix = unfData.uProjectionMatrix.mul(unfData.uModelMatrix);
        unfData.uInverseModelProjectionMatrix = unfData.uModelProjectionMatrix.inverse();
    }
    onButton(name, value) {
        console.log("onButton", name);
        switch (name) {
            case "in":
                this.Scale(1.1);
                break;
            case "out":
                this.Scale(0.9);
                break;
            case "left":
                this.Translate(-1, 0);
                break;
            case "right":
                this.Translate(1, 0);
                break;
            case "up":
                this.Translate(0, 1);
                break;
            case "down":
                this.Translate(0, -1);
                break;
            case "dir":
                fs.RequestDirectoryAccess();
                break;
            case "file":
                fs.ListenToFile("tmp.bin", (data) => { this.onFile("tmp.bin", data); });
                break;
        }
    }
    onChange(name, value) {
        console.log("onChange", name, value);
        const { signal } = this;
        switch (name) {
            case "file":
                fs.ListenToFile(value, (data) => { this.onFile(value, data); });
                break;
            case "signal":
                this.SelectSignal(value);
                break;
            case "color":
                if (signal)
                    signal.color = glo.HexToRGB(value);
                break;
            case "scale":
                if (signal)
                    signal.scale[1] = value;
                break;
            case "width":
                if (signal)
                    signal.scale[0] = value;
                break;
            case "enable":
                if (signal)
                    signal.enabled = value;
                break;
            case "stride":
                if (signal) {
                    signal.stride = value;
                    signal.recreate();
                    this.SelectSignalNode(signal);
                }
                break;
            case "comp":
                if (signal) {
                    signal.comp = value;
                    signal.recreate();
                    this.SelectSignalNode(signal);
                }
                break;
            case "offset":
                if (signal) {
                    signal.offset = value;
                    signal.recreate();
                    this.SelectSignalNode(signal);
                }
                break;
            case "shader":
                if (signal) {
                    signal.func = value;
                    signal.recreate();
                    this.SelectSignalNode(signal);
                }
                break;
        }
    }
    onFile(name, data) {
        console.log("onFile", data);
        const node = canvas.getNode("signals", name);
        if (node)
            node.update(data);
        else {
            this.signalBox.innerHTML += `<option value="${name}">${name}</option>`;
            canvas.addNode("signals", new Signal(name, data));
            this.SelectSignal(name);
        }
    }
    SelectSignal(name) {
        this.SelectSignalNode(canvas.getNode("signals", name));
    }
    SelectSignalNode(signal) {
        this.signal = signal;
        if (this.signal) {
            this.signalBox.value = this.signal.name;
            this.shaderBox.value = this.signal.func;
            this.signalColor.value = glo.ToRGBHex(this.signal.color);
            this.enableCheckbox.checked = this.signal.enabled;
            this.spinScale.value = this.signal.scale[1].toString();
            this.spinWidth.value = this.signal.scale[0].toString();
            this.spinOffset.value = this.signal.offset.toString();
            this.spinComp.value = this.signal.comp.toString();
            this.spinStride.value = this.signal.stride.toString();
        }
        else {
            this.signalBox.value = "";
            this.shaderBox.value = "";
            this.signalColor.value = "#000000";
            this.enableCheckbox.checked = false;
            this.spinScale.value = "";
            this.spinWidth.value = "";
            this.spinOffset.value = "";
            this.spinComp.value = "";
            this.spinStride.value = "";
        }
    }
    ClientToNDC(clientX, clientY) {
        const rect = canvasDiv.getBoundingClientRect();
        const x = clientX - rect.left;
        const y = rect.bottom - clientY;
        const ndcX = (x / canvasDiv.width) * 2 - 1;
        const ndcY = (y / canvasDiv.height) * 2 - 1;
        return [ndcX, ndcY];
    }
    NDCtoGrid(ndc) {
        return this.unfData.uInverseModelProjectionMatrix.transformPoint(ndc);
    }
    ClientToGrid(clientX, clientY) {
        const ndc = this.ClientToNDC(clientX, clientY);
        return this.NDCtoGrid(ndc);
    }
    onMouseDown(event) {
        this.isMouseDown = true;
        this.gridMouse = this.ClientToGrid(event.clientX, event.clientY);
    }
    onMouseMove(event) {
        if (!this.isMouseDown)
            return;
        const prevGrid = this.gridMouse;
        const gridMouse = this.ClientToGrid(event.clientX, event.clientY);
        const deltaX = gridMouse[0] - prevGrid[0];
        const deltaY = gridMouse[1] - prevGrid[1];
        if (event.ctrlKey) {
            // Scale
            const scaleFactor = Math.exp(deltaY * RESOLUTION * 0.5);
            this.Scale(scaleFactor);
        }
        else {
            // Translate
            const translateX = deltaX * this.scaleX;
            const translateY = deltaY * this.scaleY;
            this.Translate(translateX, translateY);
        }
    }
    onMouseUp() {
        this.isMouseDown = false;
    }
    onMouseWheel(event) {
        event.preventDefault();
        const scaleFactor = Math.exp(event.deltaY * RESOLUTION * 0.1);
        this.Scale(scaleFactor);
    }
    onKeyDown(event) {
        if (event.key === "0") {
            this.unfData.uScale = 1.0;
            this.unfData.uTranslate = [0.0, 0.0];
            this.resetModelMatrix();
        }
    }
}
const app = new App();
function renderLoop() {
    // console.log('renderLoop');
    app.drawScene();
    requestAnimationFrame(renderLoop);
}
renderLoop();
/// <reference path="src/App.ts" />
//# sourceMappingURL=plot.js.map