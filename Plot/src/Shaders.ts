/// <reference path="Program.ts" />
/// <reference path="shaders/unf.ts" />
/// <reference path="shaders/bk.ts" />

class ShaderMap {
    private shaders: Map<string, GLProgram>;

    constructor() {
        this.shaders = new Map<string, GLProgram>();
    }

    addProgram(name: string, vertexShader: string, fragmentShader: string): boolean {
        const program = new GLProgram();
        if (false == program.create(unfBlock + vertexShader, unfBlock + fragmentShader))
            return false;

        this.shaders.set(name, program);
        return true;
    }
    getProgram(name: string): GLProgram | undefined {
        return this.shaders.get(name);
    }
    use(name: string): void {
        const program = this.shaders.get(name);
        if (program)
            program.use();
    }
};
const shaders: ShaderMap = new ShaderMap();
shaders.addProgram('bk', shader_bk.vert, shader_bk.frag);