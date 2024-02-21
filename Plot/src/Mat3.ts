class Mat3 {
    public data: number[];

    constructor() {
        this.data = [0, 0, 0, 0, 0, 0, 0, 0, 0];
    }
    public fromArray(data: number[]): void {
        for (let i = 0; i < 9; i++) {
            this.data[i] = data[i];
        }
    }
    public Set(row: number, column: number, value: number): void {
        this.data[row * 3 + column] = value;
    }
    public Get(row: number, column: number): number {
        return this.data[row * 3 + column];
    }
    public add(matrix: Mat3): Mat3 {
        const result = new Mat3();
        for (let i = 0; i < 9; i++) {
            result.Set(Math.floor(i / 3), i % 3, this.Get(Math.floor(i / 3), i % 3) + matrix.Get(Math.floor(i / 3), i % 3));
        }
        return result;
    }
    public sub(matrix: Mat3): Mat3 {
        const result = new Mat3();
        for (let i = 0; i < 9; i++) {
            result.Set(Math.floor(i / 3), i % 3, this.Get(Math.floor(i / 3), i % 3) - matrix.Get(Math.floor(i / 3), i % 3));
        }
        return result;
    }
    public mul(matrix: Mat3): Mat3 {
        const result = new Mat3();
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                let sum = 0;
                for (let k = 0; k < 3; k++) {
                    sum += this.Get(i, k) * matrix.Get(k, j);
                }
                result.Set(i, j, sum);
            }
        }
        return result;
    }
    public mulScalar(scalar: number): Mat3 {
        const result = new Mat3();
        for (let i = 0; i < 9; i++) {
            result.Set(Math.floor(i / 3), i % 3, this.Get(Math.floor(i / 3), i % 3) * scalar);
        }
        return result;
    }
    public transformPoint(point: [number, number]): [number, number] {
        const x = this.Get(0, 0) * point[0] + this.Get(0, 1) * point[1] + this.Get(0, 2);
        const y = this.Get(1, 0) * point[0] + this.Get(1, 1) * point[1] + this.Get(1, 2);

        return [x, y];
    }
    public transpose(): Mat3 {
        const result = new Mat3();
        for (let i = 0; i < 3; i++) {
            for (let j = 0; j < 3; j++) {
                result.Set(i, j, this.Get(j, i));
            }
        }
        return result;
    }
    public determinant(): number {
        return this.Get(0, 0) * (this.Get(1, 1) * this.Get(2, 2) - this.Get(1, 2) * this.Get(2, 1)) -
            this.Get(0, 1) * (this.Get(1, 0) * this.Get(2, 2) - this.Get(1, 2) * this.Get(2, 0)) +
            this.Get(0, 2) * (this.Get(1, 0) * this.Get(2, 1) - this.Get(1, 1) * this.Get(2, 0));
    }
    public inverse(): Mat3 {
        const result = new Mat3();
        const det = this.determinant();
        if (det === 0) {
            throw new Error("Matrix is not invertible");
        }
        const invDet = 1 / det;

        result.Set(0, 0, (this.Get(1, 1) * this.Get(2, 2) - this.Get(1, 2) * this.Get(2, 1)) * invDet);
        result.Set(0, 1, (this.Get(0, 2) * this.Get(2, 1) - this.Get(0, 1) * this.Get(2, 2)) * invDet);
        result.Set(0, 2, (this.Get(0, 1) * this.Get(1, 2) - this.Get(0, 2) * this.Get(1, 1)) * invDet);

        result.Set(1, 0, (this.Get(1, 2) * this.Get(2, 0) - this.Get(1, 0) * this.Get(2, 2)) * invDet);
        result.Set(1, 1, (this.Get(0, 0) * this.Get(2, 2) - this.Get(0, 2) * this.Get(2, 0)) * invDet);
        result.Set(1, 2, (this.Get(0, 2) * this.Get(1, 0) - this.Get(0, 0) * this.Get(1, 2)) * invDet);

        result.Set(2, 0, (this.Get(1, 0) * this.Get(2, 1) - this.Get(1, 1) * this.Get(2, 0)) * invDet);
        result.Set(2, 1, (this.Get(0, 1) * this.Get(2, 0) - this.Get(0, 0) * this.Get(2, 1)) * invDet);
        result.Set(2, 2, (this.Get(0, 0) * this.Get(1, 1) - this.Get(0, 1) * this.Get(1, 0)) * invDet);

        return result;
    }
    public translate(x: number, y: number): void {
        const translation = Mat3.translation(x, y);
        this.fromArray(this.mul(translation).data);
    }
    public scale(x: number, y: number): void {
        const scale = Mat3.scale(x, y);
        this.fromArray(this.mul(scale).data);
    }
    public static identity(): Mat3 {
        const result = new Mat3();
        result.Set(0, 0, 1);
        result.Set(1, 1, 1);
        result.Set(2, 2, 1);
        return result;
    }
    public static translation(x: number, y: number): Mat3 {
        const result = Mat3.identity();
        result.Set(0, 2, x);
        result.Set(1, 2, y);
        return result;
    }
    public static scale(x: number, y: number): Mat3 {
        const result = Mat3.identity();
        result.Set(0, 0, x);
        result.Set(1, 1, y);
        return result;
    }

}