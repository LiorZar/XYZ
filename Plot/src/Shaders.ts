/// <reference path="Program.ts" />
/// <reference path="shaders/unf.ts" />
/// <reference path="shaders/reg.ts" />
/// <reference path="shaders/regc.ts" />
/// <reference path="shaders/grid.ts" />
/// <reference path="shaders/signal.ts" />
/// <reference path="shaders/energy.ts" />
/// <reference path="shaders/derivative.ts" />
/// <reference path="shaders/magnitude.ts" />

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
    use(name: string): GLProgram {
        const program = this.shaders.get(name);
        if (program)
            program.use();
        return program;
    }
};
const shaders: ShaderMap = new ShaderMap();
shaders.addProgram('reg', shader_reg.vert, shader_reg.frag);
shaders.addProgram('regc', shader_regc.vert, shader_regc.frag);
shaders.addProgram('grid', shader_grid.vert, shader_grid.frag);
shaders.addProgram('signal', shader_signal.vert, shader_signal.frag);
shaders.addProgram('energy', shader_energy.vert, shader_energy.frag);
shaders.addProgram('derivative', shader_derivative.vert, shader_derivative.frag);
shaders.addProgram('magnitude', shader_magnitude.vert, shader_magnitude.frag);

