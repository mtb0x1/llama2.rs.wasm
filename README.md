
# What is this
This is a dirty demo in wasm of llama2.c, you can run the demo localy.

This demo relies on notables implenation higlighted in [here](https://github.com/karpathy/llama2.c) under notable forks->rust section.

# How
1) Download the release tarball and untar it.
2) Run `python3 -m http.server 8080` in `www` folder.
3) Open `http://127.0.0.1:8080/` in your browser.

or

1) run `wasm-pack build --release --target web --out-dir www/pkg/ --verbose`
2) download models :
```bash
mkdir -p stories/
wget -P stories/ https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget -P stories/ https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
wget -P stories/ https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
for i in $(ls -d port*/www/)
do
    cd $i
    ln -s ../../stories/stories15M.bin . 
    ln -s ../../stories/stories42M.bin .
    ln -s ../../stories/stories110M.bin .
done
```
3) Run `python3 -m http.server 8080` in `www` folder.
4) Open `http://127.0.0.1:8080/` in your browser.

or

check it live demo [here](tbd)

## Credit
1. Port1 [A dirty and minimal port of @Gaxler llama2.rs](https://github.com/mtb0x1/llama2.rs.wasm/blob/main/port1/README.md).
2. Port2 [A dirty and minimal port of @Leo-du llama2.rs](https://github.com/mtb0x1/llama2.rs.wasm/blob/main/port2/README.md).
3. Port3 [ A dirty and minimal port of @danielgrittner llama2-rs](https://github.com/mtb0x1/llama2.rs.wasm/blob/main/port3/README.md).
3. Port4 [ A dirty and minimal port of @lintian06 llama2.rs](https://github.com/mtb0x1/llama2.rs.wasm/blob/main/port4/README.md).
3. Port5 [ A dirty and minimal port of @rahoua pecca.rs](https://github.com/mtb0x1/llama2.rs.wasm/blob/main/port5/README.md).
3. Port6 [ A dirty and minimal port of @flaneur2020 llama2.rs](https://github.com/mtb0x1/llama2.rs.wasm/blob/main/port6/README.md).