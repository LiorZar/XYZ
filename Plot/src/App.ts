/// <reference path="Canvas.ts" />
/// <reference path="Shaders.ts" />
/// <reference path="RNode.ts" />

class App {
    public unfData = {
        resolution: [0.0, 0.0],
        grid: [0.0, 0.0, 0.0, 0.0],
        scale: [0.0, 0.0],
        translate: [0.0, 0.0]
    }

    constructor() {
        const vertexData = [
            // Position    // Color
            -0.5, 0.5, 1.0, 0.0, 0.0, 1.0, // Top left (red)
            0.5, 0.5, 0.0, 1.0, 0.0, 1.0, // Top right (green)
            -0.5, -0.5, 0.0, 0.0, 1.0, 1.0, // Bottom left (blue)
            0.5, -0.5, 1.0, 1.0, 0.0, 1.0  // Bottom right (yellow)
        ];
        const node = new RNode("node1", "reg", vertexData, [2, 4]);
        canvas.addNode("bk", node);
    }
    private resizeCanvasToDisplaySize() {
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

    drawScene() { // Code to draw the scene goes here }       
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

const app: App = new App();
function renderLoop() {
    // console.log('renderLoop');
    app.drawScene();
    requestAnimationFrame(renderLoop);
}
renderLoop();