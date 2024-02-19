
class GLProgram {
    private program: WebGLProgram | null;
    private uniforms: Map<string, WebGLUniformLocation>;

    constructor() {
        this.program = null;
    }

    public create(vertexShaderSource: string, fragmentShaderSource: string): boolean {
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

        this.uniforms = new Map<string, WebGLUniformLocation>();
        const numUniforms = gl.getProgramParameter(this.program, gl.ACTIVE_UNIFORMS);

        for (let i = 0; i < numUniforms; i++) {
            const uniformInfo = gl.getActiveUniform(this.program, i);
            const location = gl.getUniformLocation(this.program, uniformInfo.name);
            this.uniforms.set(uniformInfo.name, location); // Fix: Use set() method instead of push()
        }

        return true;
    }

    public use(): void {
        gl.useProgram(this.program);
    }

    public getProgram(): WebGLProgram | null {
        return this.program;
    }

    public bindData(data: any): void {
        for (let [key, value] of data) {
            if (Array.isArray(value)) {
                if (value.length === 1)
                    this.uniform1f(key, value[0]);
                else if (value.length === 2)
                    this.uniform2f(key, value[0], value[1]);
                else if (value.length === 3)
                    this.uniform3f(key, value[0], value[1], value[2]);
                else if (value.length === 4)
                    this.uniform4f(key, value[0], value[1], value[2], value[3]);
            }
            else
                this.uniform1f(key, value);
        }
    }
    public uniform1f(name: string, value: number): void {
        const location = this.uniforms.get(name);
        if (location !== undefined)
            gl.uniform1f(location, value);
    }

    public uniform2f(name: string, value1: number, value2: number): void {
        const location = this.uniforms.get(name);
        if (location !== undefined)
            gl.uniform2f(location, value1, value2);
    }

    public uniform3f(name: string, value1: number, value2: number, value3: number): void {
        const location = this.uniforms.get(name);
        if (location !== undefined)
            gl.uniform3f(location, value1, value2, value3);
    }

    public uniform4f(name: string, value1: number, value2: number, value3: number, value4: number): void {
        const location = this.uniforms.get(name);
        if (location !== undefined)
            gl.uniform4f(location, value1, value2, value3, value4);
    }

    public uniform1i(name: string, value: number): void {
        const location = this.uniforms.get(name);
        if (location !== undefined)
            gl.uniform1i(location, value);
    }

    public uniform2i(name: string, value1: number, value2: number): void {
        const location = this.uniforms.get(name);
        if (location !== undefined)
            gl.uniform2i(location, value1, value2);
    }

    public uniform3i(name: string, value1: number, value2: number, value3: number): void {
        const location = this.uniforms.get(name);
        if (location !== undefined)
            gl.uniform3i(location, value1, value2, value3);
    }

    public uniform4i(name: string, value1: number, value2: number, value3: number, value4: number): void {
        const location = this.uniforms.get(name);
        if (location !== undefined)
            gl.uniform4i(location, value1, value2, value3, value4);
    }
}