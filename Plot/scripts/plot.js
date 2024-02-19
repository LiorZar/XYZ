"use strict";
/// <reference path="INode.ts" />
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
    addNode(layerName, node) {
        const layer = this.layers.find(layer => layer.name === layerName);
        if (!layer)
            this.layers.push({ name: layerName, nodes: [node] });
        else {
            const enode = layer.nodes.find(n => n.name === node.name);
            if (enode)
                throw new Error(`Node with name ${node.name} already exists in layer ${layerName}`);
            layer.nodes.push(node);
        }
    }
    remNode(layerName, nodeName) {
        const layer = this.layers.find(layer => layer.name === layerName);
        if (layer) {
            const index = layer.nodes.findIndex(node => node.name === nodeName);
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
    bindData(data) {
        for (let key in data) {
            const value = data[key];
            if (Array.isArray(value)) {
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
}
const unfBlock = `#version 300 es

precision highp float;

uniform vec2 uResolution;
uniform vec4 uGrid;
uniform vec2 uScale;   
uniform vec2 uTranslate;


`;
const shader_reg = {
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
/// <reference path="Program.ts" />
/// <reference path="shaders/unf.ts" />
/// <reference path="shaders/reg.ts" />
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
class glo {
    static Set(obj, key, value) {
        obj[key] = value;
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
            throw new Error('Invalid count');
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
/// <reference path="Shaders.ts" />
/// <reference path="Buffer.ts" />
class RNode {
    constructor(name, shader, vertexData, numComponents) {
        this.name = name;
        this.shader = shader;
        const buffer = new GLBuffer();
        buffer.create(new Float32Array(vertexData), numComponents, gl.STATIC_DRAW);
        this.buffer = buffer;
    }
    draw() {
        this.buffer.Draw();
    }
}
/// <reference path="Canvas.ts" />
/// <reference path="Shaders.ts" />
/// <reference path="RNode.ts" />
class App {
    constructor() {
        this.unfData = {
            resolution: [0.0, 0.0],
            grid: [0.0, 0.0, 0.0, 0.0],
            scale: [0.0, 0.0],
            translate: [0.0, 0.0]
        };
        const vertexData = [
            // Position    // Color
            -0.5, 0.5, 1.0, 0.0, 0.0, 1.0,
            0.5, 0.5, 0.0, 1.0, 0.0, 1.0,
            -0.5, -0.5, 0.0, 0.0, 1.0, 1.0,
            0.5, -0.5, 1.0, 1.0, 0.0, 1.0 // Bottom right (yellow)
        ];
        const node = new RNode("node1", "reg", vertexData, [2, 4]);
        canvas.addNode("bk", node);
    }
    resizeCanvasToDisplaySize() {
        const width = canvasDiv.clientWidth;
        const height = canvasDiv.clientHeight;
        if (canvasDiv.width !== width || canvasDiv.height !== height) {
            canvasDiv.width = width;
            canvasDiv.height = height;
            gl.viewport(0, 0, width, height);
            return true; // The canvas size was changed
        }
        return false; // The canvas size was not changed
    }
    drawScene() {
        this.resizeCanvasToDisplaySize();
        canvas.clear();
        let shaderName = "";
        for (const layer of canvas.layers) {
            for (const node of layer.nodes) {
                if (shaderName !== node.shader) {
                    shaderName = node.shader;
                    shaders.use(shaderName).bindData(this.unfData);
                }
                node.draw();
            }
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