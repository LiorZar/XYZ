class glo {
    public static bal = {};
    public static Set(obj: any, key: string, value: any): void {
        obj[key] = value;
    }
    public static ToResL(val: number, res: number): number {
        return Math.floor(val / res) * res;
    }
    public static ToResH(val: number, res: number): number {
        return Math.ceil(val / res) * res;
    }
    public static AddRes(val: number, add: number, res: number): number {
        if (add > 0)
            return this.ToResH(val + add, res);
        return this.ToResL(val + add, res);
    }
    public static MulRes(val: number, mul: number, res: number): number {
        if (mul > 1)
            return this.ToResH(val * mul, res);
        return this.ToResL(val * mul, res);
    }
    public static ToRGBHex(numbers: number[]): string {
        const r = numbers[0] * 255.0;
        const g = numbers[1] * 255.0;
        const b = numbers[2] * 255.0;
        let hex = ((r << 16) | (g << 8) | b).toString(16);
        while (hex.length < 6)
            hex = '0' + hex;
        return '#' + hex;
    }

    public static HexToRGB(hex: string): number[] {
        const r = parseInt(hex.substring(1, 3), 16) / 255.0;
        const g = parseInt(hex.substring(3, 5), 16) / 255.0;
        const b = parseInt(hex.substring(5, 7), 16) / 255.0;
        return [r, g, b, 1];
    }
}