/// <reference path="Canvas.ts" />
/// <reference path="Shaders.ts" />
/// <reference path="RNode.ts" />

class App {
    public unfData = {
        scale: [0.0, 0.0],
        translate: [0.0, 0.0],
        rotation: 0.0,
        resolution: [0.0, 0.0],
        time: 0.0
    }

    constructor() {
        const vertexData = [
            // Position    // Color
            -0.5, 0.5, 1.0, 0.0, 0.0, 1.0, // Top left (red)
            0.5, 0.5, 0.0, 1.0, 0.0, 1.0, // Top right (green)
            -0.5, -0.5, 0.0, 0.0, 1.0, 1.0, // Bottom left (blue)
            0.5, -0.5, 1.0, 1.0, 0.0, 1.0  // Bottom right (yellow)
        ];
        const node = new RNode("node1", "bk", vertexData, [2, 4]);
        canvas.addNode("main", node);
    }

    drawScene() { // Code to draw the scene goes here }       
        canvas.clear();
        for (const name in canvas.layers) {
            const layer = canvas.layers[name];
            for (const node of layer) {
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