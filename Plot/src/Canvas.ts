/// <reference path="Nodes/INode.ts" />

console.clear();

const canvasDiv: HTMLCanvasElement = document.getElementById('__canvas__') as HTMLCanvasElement;
const gl: WebGL2RenderingContext = canvasDiv.getContext('webgl2');
gl.clearColor(0, 0, 0, 1);
gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

class Canvas {
    public clearColor: number[];
    public layers: { name: string, nodes: INode[] }[] = [];

    getNode(layerName: string, nodeName: string): INode | undefined {
        const layer = this.layers.find(layer => layer.name === layerName);
        if (layer)
            return layer.nodes.find(node => node.name === nodeName);
    }
    addNode(layerName: string, node: INode) {
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
    remNode(layerName: string, nodeName: string) {
        const layer = this.layers.find(layer => layer.name === layerName);
        if (layer) {
            const index = layer.nodes.findIndex(n => n.name === nodeName);
            if (index !== -1)
                layer.nodes.splice(index, 1);
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
