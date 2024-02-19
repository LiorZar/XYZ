interface INode {
    name: string;
    shader: string;
    draw(prog: GLProgram): void;
}