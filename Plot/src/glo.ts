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
}