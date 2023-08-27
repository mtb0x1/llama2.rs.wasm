for i in {1..2}
do
    echo "Building : Port$i"
    cd "port$i" && wasm-pack build --release --target web --out-dir www/pkg/ --verbose && cd ..
done
