## llama2.rs.wasm 🦀
A dirty and minimal port of @gaxler [llama2.rs](https://github.com/gaxler/llama2.rs) 

## How to run?
1. Download Karpathy's baby Llama2 (Orig instructions) pretrained on TinyStories dataset and place them in `www` folder.
```bash
    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin
    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin
    wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin
```
> stories42M is used by (for now @todo), you can change this in `index.html`

3. Run (requires (wasm-pack)[https://github.com/rustwasm/wasm-pack]) 
    ```bash 
    wasm-pack build --release --target web --out-dir www/pkg/
    ```
4. Run a minimal webserver with `www``folder : 
    1. Run (requires python 3), you can use other webservers if you want
    ```bash
    python3 -m http.server 8080
    ```
    2. go to http://localhost:8080/
    3. open browser console (@todo)
5. (Optionnal) if you want to make changes :(reload browser/clear cache after changes)
    1. Changing `lib.rs` content :
        ```bash
        wasm-pack build --release --target web --out-dir www/pkg/
        ```
    2. Changing the frontend `index.html`
    3. Changing model/tokenizer :
        - Follow instruction in @karpathy in [llama2.c](https://github.com/karpathy/llama2.c)
        - Place new files in `www` folder and edit `index.html` if needed

## Performance

Not really sure about result (yet!).
|    tok/s   | 15M | 42M | 110M | 7B
|-------|-----|-----|-----|-----|
| Temp :0.9,sql_len:11 |  ?|   ~20   | ? | ?

### todo/Next ?
- Display bench result in webpage instead of browser console
- Infrence based on user inputs
- Optmization : simd, rayon ... etc

## License
MIT