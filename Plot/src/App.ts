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
        uTranslate: [0.0, 0.0]
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

        const grid = new Grid("grid", 40);
        canvas.addNode("bk", grid);
        canvas.addNode("elems", new Lines("node3", "reg", [0, 0, -5.0, 5]));
        canvas.addNode("elems", new Quad("node2", "reg", [3.3, 0, 4.5, 0, 3.3, 1.2, 4.5, 1.2]));

        this.signalBox.innerHTML = "";
    }
    private resizeCanvasToDisplaySize(force = false) {
        const width = canvasDiv.clientWidth;
        const height = canvasDiv.clientHeight;
        if (!(force || canvasDiv.width !== width || canvasDiv.height !== height))
            return false;

        canvasDiv.width = width;
        canvasDiv.height = height;
        gl.viewport(0, 0, width, height);
        this.unfData.uResolution[0] = width;
        this.unfData.uResolution[1] = height;
        this.aspect = width / height;

        return true;
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
    private Scale(factor: number) {
        if (factor < 0)
            return;
        this.unfData.uScale = Math.max(RESOLUTION, glo.MulRes(this.unfData.uScale, factor, RESOLUTION));
        this.scaleX = this.scaleY = this.unfData.uScale;
        if (this.aspect > 1)
            this.scaleX /= this.aspect;
        else
            this.scaleY *= this.aspect;
        console.log("Scale", this.unfData.uScale);
    }
    private Translate(x: number, y: number) {
        this.unfData.uTranslate[0] = glo.AddRes(this.unfData.uTranslate[0], x * MOVE_STEP, RESOLUTION);
        this.unfData.uTranslate[1] = glo.AddRes(this.unfData.uTranslate[1], y * MOVE_STEP, RESOLUTION);
        console.log("Translate", this.unfData.uTranslate);
    }

    public onButton(name: string, value?: any) {
        console.log("onButton", name);
        switch (name) {
            case "in": this.Scale(1.1); break;
            case "out": this.Scale(0.9); break;

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
            case "shader": if (signal) signal.shader = value; break;
            case "color": if (signal) signal.color = glo.HexToRGB(value); break;
            case "scale": if (signal) signal.scale[1] = value; break;
            case "width": if (signal) signal.scale[0] = value; break;
            case "enable": if (signal) signal.enabled = value; break;
            case "stride":
                if (signal) {
                    signal.stride = value;
                    signal.recreate();
                }
                break;
            case "comp":
                if (signal) {
                    signal.comp = value;
                    signal.recreate();
                }
                break;
            case "offset":
                if (signal) {
                    signal.offset = value;
                    signal.recreate();
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
        this.signal = canvas.getNode("signals", name) as Signal;
        if (this.signal) {
            this.signalBox.value = this.signal.name;
            this.shaderBox.value = this.signal.shader;
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
    private ClientToGrid(clientX: number, clientY: number): [number, number] {
        const { aspect, unfData, scaleX, scaleY } = this
        const rect = canvasDiv.getBoundingClientRect();
        const x = clientX - rect.left;
        const y = rect.bottom - clientY;

        // Convert pixel to OpenGL world location
        const glWidth = canvasDiv.width;
        const glHeight = canvasDiv.height;
        const ndcX = (x / glWidth) * 2 - 1;
        const ndcY = (y / glHeight) * 2 - 1;


        const gridX = (glo.lerpZ(unfData.uGrid[0], unfData.uGrid[1], ndcX) - unfData.uTranslate[0]) / scaleX;
        const gridY = (glo.lerpZ(unfData.uGrid[2], unfData.uGrid[3], ndcY) - unfData.uTranslate[1]) / scaleY;

        return [gridX, gridY];
    }
    private onMouseDown(event: MouseEvent) {
        this.isMouseDown = true;
        this.gridMouse = this.ClientToGrid(event.clientX, event.clientY);
    }

    private onMouseMove(event: MouseEvent) {
        // console.log("onMouseMove", event.clientX, event.clientY, this.ClientToGrid(event.clientX, event.clientY));
        if (!this.isMouseDown) return;

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

    private onMouseUp() {
        this.isMouseDown = false;
    }

    private onMouseWheel(event: WheelEvent) {
        event.preventDefault();

        const scaleFactor = Math.exp(event.deltaY * RESOLUTION * 0.1);
        this.Scale(scaleFactor);
    }
}
const app: App = new App();
function renderLoop() {
    // console.log('renderLoop');
    app.drawScene();
    requestAnimationFrame(renderLoop);
}
renderLoop();