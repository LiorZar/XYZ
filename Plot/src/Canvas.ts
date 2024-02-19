/// <reference path="INode.ts" />

console.clear();

const canvasDiv: HTMLCanvasElement = document.getElementById('__canvas__') as HTMLCanvasElement;
const gl: WebGL2RenderingContext = canvasDiv.getContext('webgl2');
gl.clearColor(0, 0, 0, 1);
gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

class Canvas {
    public clearColor: number[];
    public layers: { name: string, nodes: INode[] }[] = [];

    addNode(layerName: string, node: INode) {
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
    remNode(layerName: string, nodeName: string) {
        const layer = this.layers.find(layer => layer.name === layerName);
        if (layer) {
            const index = layer.nodes.findIndex(node => node.name === nodeName);
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
