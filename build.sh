#Build 
for i in {1..2}
do
    echo "Building : Port$i"
    cd "port$i" && wasm-pack build --release --target web --out-dir www/pkg/ --verbose && cd ..
done

#Copy to www

for i in {1..2}
do
    echo "Copying : Port$i"
    rm -rf "www/pkg_port$i"
    cp -r "port$i/www/pkg" "www/pkg_port$i"
    rm -rf "www/tokenizer_port$i.bin"
    cp "port$i/www/tokenizer.bin" "www/tokenizer_port$i.bin"
    rm -rf "www/pkg_port$i/README.md"
    rm -rf "www/pkg_port$i/.gitignore"
done

