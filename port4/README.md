## llama2.rs.wasm 🦀
A dirty and minimal port of [@lintian06](https://github.com/lintian06) [llama2.rs](https://github.com/lintian06/llama2.rs/) 

<p align="center">
  <img src="assets/llama_notcute.jpg" width="300" height="300" alt="Cute Llama">
</p>


## How to run?
1. Clone repo
```bash
git clone https://github.com/mtb0x1/llama2.rs.wasm
cd llama2.rs.wasm/port4/
```

2. Download [@Karpathy](https://github.com/karpathy/)'s baby Llama2 ([Orig instructions](https://github.com/karpathy/llama2.c#feel-the-magic)) pretrained on TinyStories dataset and place them in `www` folder.
```bash
wget -P www/ https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
wget -P www/ https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
wget -P www/ https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
```
> stories42M is used by default (for now @todo), you can change this in `index.html`

3. Run (requires [wasm-pack](https://github.com/rustwasm/wasm-pack)) 
    ```bash 
    wasm-pack build --release --target web --out-dir www/pkg/
    ```
4. Run a minimal webserver with `www` folder : 
    1. Run (requires python 3), you can use other webservers if you want
    ```bash
    cd www && python3 -m http.server 8080
    ```
    2. go to http://localhost:8080/
    3. open browser console (@todo)
5. (Optional) if you want to make changes :(reload browser/clear cache after changes)
    1. Changing `lib.rs` content :
        ```bash
        wasm-pack build --release --target web --out-dir www/pkg/
        ```
    2. Changing the frontend `index.html`
    3. Changing model/tokenizer :
        - Follow [@Karpathy](https://github.com/karpathy/)'s instructions in [llama2.c](https://github.com/karpathy/llama2.c)
        - Place new files in `www` folder and edit `index.html` if needed

## Performance
- Temperature : 0.9
- Sequence length: 20

|    tok/s   | 15M | 42M | 110M | 7B
|-------|-----|-----|-----|-----|
| wasm v1 |  ?|   ?   | ? | ?
> Not really sure about result (yet!).

### todo/Next ?
- Tests
- Display bench result in webpage instead of browser console (wip need cleaning and remove console.info hack)
- Infrence based on user inputs (done)
- Optmization : simd, rayon (wip) ... etc

## License
MIT