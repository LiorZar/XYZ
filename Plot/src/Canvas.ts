/// <reference path="INode.ts" />

console.clear();

const canvasDiv: HTMLCanvasElement = document.getElementById('__canvas__') as HTMLCanvasElement;
const gl: WebGL2RenderingContext = canvasDiv.getContext('webgl2');
gl.clearColor(0, 0, 0, 1);
gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

class Canvas {
    public clearColor: number[];
    public inodes: { [name: string]: INode } = {};
    public layers: { [name: string]: INode[] } = {};

    addNode(layerName: string, node: INode) {
        if (!this.layers[layerName]) {
            this.layers[layerName] = [];
        }
        this.layers[layerName].push(node);
        this.inodes[node.name] = node;
    }

    remNode(layerName: string, nodeName: string) {
        const layer = this.layers[layerName];
        if (layer) {
            const index = layer.findIndex(node => node.name === nodeName);
            if (index !== -1) {
                layer.splice(index, 1);
                delete this.inodes[nodeName];
            }
        }
    }

    constructor() {
        this.clearColor = [0, 0, 0, 1];
    }
    setClearColor(r: number, g: number, b: number, a: number) {
        this.clearColor = [r, g, b, a];
        gl.clearColor(this.clearColor[0], this.clearColor[1], this.clearColor[2], this.clearColor[3]);
    }
    clear() {
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    }
};
const canvas: Canvas = new Canvas();
