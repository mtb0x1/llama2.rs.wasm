
## llama2.rs.wasm 🦀
> a dirty and minimal port of @gaxler [llama2.rs](https://github.com/gaxler/llama2.rs) 

## How to run?
1. Run (requires (wasm-pack)[https://github.com/rustwasm/wasm-pack]) 
    ```bash 
    wasm-pack build --release --target web --out-dir www/pkg/
    ```
2. Run a minimal webserver with `www``folder : 
    a. Run (requires python 3), you can use other webservers if you want
    ```bash
    python3 -m http.server 8080
    ```
    b. go to http://localhost:8080/
    c. open browser console (@todo)
3. (Optionnal) if you want to make changes :(reload browser/clear cache after changes)
    a. Changing `lib.rs` content :
        ```bash
        wasm-pack build --release --target web --out-dir www/pkg/
        ```
    b. Changing the frontend `index.html`
    c. Changing model/tokenizer :
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