const wnd: any = window;
class FileSystemAccess {
    private directoryHandle: any = null;
    private files: Map<string, number> = new Map();

    // Requests directory access from the user
    public async RequestDirectoryAccess() {
        try {
            this.directoryHandle = await wnd.showDirectoryPicker();
            console.log("Directory access granted");
        } catch (error) {
            console.error("Directory access denied", error);
        }
    }
    public ListenToFile(fileName: string, callback: (data: ArrayBuffer) => void) {
        this.files.set(fileName, 0);
        this.watchFile(fileName, callback);
    }
    // Listens to changes in a file
    private async watchFile(fileName: string, callback: (data: ArrayBuffer) => void) {
        try {
            if (!this.directoryHandle)
                throw new Error("Directory access not granted");

            while (true) {
                if (!this.files.has(fileName))
                    break;

                const lastModified = this.files.get(fileName);

                const fileHandle = await this.directoryHandle.getFileHandle(fileName, { create: false });
                const file = await fileHandle.getFile();
                if (file.size <= 0)
                    continue
                if (lastModified === file.lastModified)
                    continue;

                const contents: ArrayBuffer = await file.arrayBuffer();
                this.files.set(fileName, file.lastModified);
                callback(contents);
            }
        } catch (error) {
            console.error(`Error watching file = ${fileName}`, error);
        }
        console.log(`Stopped watching file = ${fileName} `);
    }
    // Reads a file from the directory
    private async readFile(fileName: string) {
        try {
            if (!this.directoryHandle) {
                throw new Error("Directory access not granted");
            }
            const fileHandle = await this.directoryHandle.getFileHandle(fileName, { create: false });
            const file = await fileHandle.getFile();
            const contents: any = await file.text();
            return contents;
        } catch (error) {
            console.error("Error reading file", error);
            return null;
        }
    }

    // Writes to a file in the directory
    private async writeFile(fileName: string, contents: any) {
        try {
            if (!this.directoryHandle) {
                throw new Error("Directory access not granted");
            }
            const fileHandle = await this.directoryHandle.getFileHandle(fileName, { create: true });
            const writableStream = await fileHandle.createWritable();
            await writableStream.write(contents);
            await writableStream.close();
            console.log("File written successfully");
        } catch (error) {
            console.error("Error writing file", error);
        }
    }
}
const fs = new FileSystemAccess();