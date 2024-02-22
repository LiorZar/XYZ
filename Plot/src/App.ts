/// <reference path="Canvas.ts" />
/// <reference path="Shaders.ts" />
/// <reference path="FileSystemAccess.ts" />
/// <reference path="Nodes/RNode.ts" />
/// <reference path="Nodes/Grid.ts" />
/// <reference path="Nodes/Quad.ts" />
/// <reference path="Nodes/Lines.ts" />
/// <reference path="Nodes/Signal.ts" />

const RESOLUTION: number = 0.01;
const MOVE_STEP: number = 0.1;

class App {
    public unfData = {
        uResolution: [0.0, 0.0],
        uGrid: [-10.0, 10.0, -10.0, 10.0],
        uScale: 1.0,
        uTranslate: [0.0, 0.0],
        uModelMatrix: Mat3.identity(),
        uProjectionMatrix: Mat3.identity(),
        uModelProjectionMatrix: Mat3.identity(),
        uInverseModelProjectionMatrix: Mat3.identity()
    }
    private signalBox: HTMLSelectElement = document.getElementById("signalBox") as HTMLSelectElement;
    private shaderBox: HTMLSelectElement = document.getElementById("shaderBox") as HTMLSelectElement;
    private signalColor: HTMLInputElement = document.getElementById("signalColor") as HTMLInputElement;
    private enableCheckbox: HTMLInputElement = document.getElementById("enableCheckbox") as HTMLInputElement;
    private spinScale: HTMLInputElement = document.getElementById("spinScale") as HTMLInputElement;
    private spinWidth: HTMLInputElement = document.getElementById("spinWidth") as HTMLInputElement;
    private spinOffset: HTMLInputElement = document.getElementById("spinOffset") as HTMLInputElement;
    private spinComp: HTMLInputElement = document.getElementById("spinComp") as HTMLInputElement;
    private spinStride: HTMLInputElement = document.getElementById("spinStride") as HTMLInputElement;
    private signal: Signal | undefined;

    private aspect: number = 1;
    private scaleX: number = 1;
    private scaleY: number = 1;
    private isMouseDown: boolean = false;
    private gridMouse: number[] = [0, 0];

    constructor() {
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
    private resizeCanvasToDisplaySize(force = false) {
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

    public drawScene() {
        this.resizeCanvasToDisplaySize();
        canvas.clear();

        let shaderName = "";
        let program: GLProgram | undefined;
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
    private Scale(factor: number, gMouse: [number, number]) {
        if (factor < 0)
            return;
        const { unfData } = this;
        unfData.uModelMatrix.translate(gMouse[0], gMouse[1]);
        unfData.uModelMatrix.scale(factor, factor);
        unfData.uModelMatrix.translate(-gMouse[0], -gMouse[1]);
        this.refreshMVP();
    }
    private Translate(x: number, y: number) {
        this.unfData.uModelMatrix.translate(x, y);
        this.refreshMVP();
    }
    private resetModelMatrix() {
        const { unfData } = this;
        unfData.uModelMatrix.fromArray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        this.refreshMVP();
    }
    private refreshMVP() {
        const { unfData } = this;
        unfData.uModelProjectionMatrix = unfData.uProjectionMatrix.mul(unfData.uModelMatrix);
        unfData.uInverseModelProjectionMatrix = unfData.uModelProjectionMatrix.inverse();
    }
    public onButton(name: string, value?: any) {
        console.log("onButton", name);
        switch (name) {
            case "in": this.Scale(1.1, [0, 0]); break;
            case "out": this.Scale(0.9, [0, 0]); break;

            case "left": this.Translate(-1, 0); break;
            case "right": this.Translate(1, 0); break;
            case "up": this.Translate(0, 1); break;
            case "down": this.Translate(0, -1); break;
            case "dir": fs.RequestDirectoryAccess(); break;
            case "file": fs.ListenToFile("tmp.bin", (data: any) => { this.onFile("tmp.bin", data); }); break;
        }
    }
    public onChange(name: string, value?: any) {
        console.log("onChange", name, value);
        const { signal } = this;

        switch (name) {
            case "file": fs.ListenToFile(value, (data: any) => { this.onFile(value, data); }); break;
            case "signal": this.SelectSignal(value); break;
            case "color": if (signal) signal.color = glo.HexToRGB(value); break;
            case "scale": if (signal) signal.scale[1] = value; break;
            case "width": if (signal) signal.scale[0] = value; break;
            case "enable": if (signal) signal.enabled = value; break;
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
    private onFile(name: string, data: any) {
        console.log("onFile", data);
        const node: Signal = canvas.getNode("signals", name) as Signal;
        if (node)
            node.update(data);
        else {
            this.signalBox.innerHTML += `<option value="${name}">${name}</option>`;
            canvas.addNode("signals", new Signal(name, data));
            this.SelectSignal(name);
        }
    }
    private SelectSignal(name: string) {
        this.SelectSignalNode(canvas.getNode("signals", name) as Signal);
    }
    private SelectSignalNode(signal?: Signal) {
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
    private ClientToNDC(clientX: number, clientY: number): [number, number] {
        const rect = canvasDiv.getBoundingClientRect();
        const x = clientX - rect.left;
        const y = rect.bottom - clientY;

        const ndcX = (x / canvasDiv.width) * 2 - 1;
        const ndcY = (y / canvasDiv.height) * 2 - 1;
        return [ndcX, ndcY];
    }
    private NDCtoGrid(ndc: [number, number]): [number, number] {
        return this.unfData.uInverseModelProjectionMatrix.transformPoint(ndc);
    }
    private ClientToGrid(clientX: number, clientY: number): [number, number] {
        const ndc = this.ClientToNDC(clientX, clientY);
        return this.NDCtoGrid(ndc);
    }
    private onMouseDown(event: MouseEvent) {
        this.isMouseDown = true;
        this.gridMouse = this.ClientToGrid(event.clientX, event.clientY);
    }

    private onMouseMove(event: MouseEvent) {
        if (!this.isMouseDown) return;

        const prevGrid = this.gridMouse;
        const gridMouse = this.ClientToGrid(event.clientX, event.clientY);
        const deltaX = gridMouse[0] - prevGrid[0];
        const deltaY = gridMouse[1] - prevGrid[1];

        if (event.ctrlKey) {
            // Scale
            const scaleFactor = Math.exp(deltaY * RESOLUTION * 0.5);
            this.Scale(scaleFactor, gridMouse);
        }
        else {
            // Translate
            const translateX = deltaX * this.scaleX;
            const translateY = deltaY * this.scaleY;
            this.Translate(translateX, translateY);
        }
    }

    private onMouseUp() {
        this.isMouseDown = false;
    }

    private onMouseWheel(event: WheelEvent) {
        event.preventDefault();
        const gridMouse = this.ClientToGrid(event.clientX, event.clientY);

        const scaleFactor = Math.exp(event.deltaY * RESOLUTION * 0.1);
        this.Scale(scaleFactor, gridMouse);
    }
    private onKeyDown(event: KeyboardEvent) {
        if (event.key === "0") {
            this.unfData.uScale = 1.0;
            this.unfData.uTranslate = [0.0, 0.0];
            this.resetModelMatrix();
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