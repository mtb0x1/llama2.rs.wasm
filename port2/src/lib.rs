use rayon::current_num_threads;
use rayon::prelude::*;

use std::io::{stdout, BufReader, Read, Write};
use wasm_bindgen::prelude::wasm_bindgen;

// ---------------------------------------------------------------------------
// RNG (Permuted Congruential Generator)
pub struct PCG {
    state: u64,
    inc: u64,
}

impl PCG {
    fn randint(&mut self) -> u32 {
        let old_state: u64 = self.state;
        self.state = old_state
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(self.inc);
        let xor_shifted = (((old_state >> 18u32) ^ old_state) >> 27u32) as u32;
        let rot = (old_state >> 59u32) as u32;
        (xor_shifted >> rot) | (xor_shifted << (((!rot).wrapping_add(1)) & 31))
    }

    // A slightly hacky but good enough way to generate random float
    pub fn rand(&mut self) -> f32 {
        let r_int = self.randint();
        (r_int as f32) * (2.0_f32.powi(-32))
    }

    pub fn new(init_state: u64, init_seq: u64) -> Self {
        let mut rng = PCG {
            state: 0,
            inc: (init_seq << 1u32) | 1u64,
        };
        _ = rng.randint();
        rng.state += init_state;
        _ = rng.randint();
        rng
    }
}

// ---------------------------------------------------------------------------
// Transformer data structures

#[derive(Debug)]
struct Config {
    dim: i32,
    hidden_dim: i32,
    n_layers: i32,
    n_heads: i32,
    #[allow(dead_code)]
    n_kv_heads: i32,
    vocab_size: i32,
    seq_len: i32,
    shared_weight: bool,
}

impl Config {
    fn from_buf_reader(f: &mut BufReader<std::io::Cursor<Vec<u8>>>) -> Self {
        let c = Self {
            dim: read::<i32>(f),
            hidden_dim: read::<i32>(f),
            n_layers: read::<i32>(f),
            n_heads: read::<i32>(f),
            n_kv_heads: read::<i32>(f),
            vocab_size: read::<i32>(f),
            seq_len: read::<i32>(f),
            shared_weight: false,
        };
        Self {
            shared_weight: c.vocab_size > 0,
            vocab_size: c.vocab_size.abs(),
            ..c
        }
    }
}

struct TransformerWeights {
    // token embedding table
    token_embedding_table: Vec<f32>, // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: Vec<f32>, // (layer, dim) rmsnorm weights
    rms_ffn_weight: Vec<f32>, // (layer, dim)
    // weights for matmuls
    wq: Vec<f32>, // (layer, dim, dim)
    wk: Vec<f32>, // (layer, dim, dim)
    wv: Vec<f32>, // (layer, dim, dim)
    wo: Vec<f32>, // (layer, dim, dim)
    // weights for ffn
    w1: Vec<f32>, // (layer, hidden_dim, dim)
    w2: Vec<f32>, // (layer, dim, hidden_dim)
    w3: Vec<f32>, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: Vec<f32>, // (dim,)
    // freq_cis for RoPE relatively positional embeddings
    freq_cis_real: Vec<f32>, // (seq_len, dim/2)
    freq_cis_imag: Vec<f32>, // (seq_len, dim/2)
    // optional output embedding
    wcls: Option<Vec<f32>>, // (vocab_size, dim)
}

impl TransformerWeights {
    fn from_buf_reader(f: &mut BufReader<std::io::Cursor<Vec<u8>>>, c: &Config) -> Self {
        let token_embedding_table = read_vec::<f32>(f, c.vocab_size * c.dim);
        let rms_att_weight = read_vec::<f32>(f, c.n_layers * c.dim);
        let wq = read_vec::<f32>(f, c.n_layers * c.dim * c.dim);
        let wk = read_vec::<f32>(f, c.n_layers * c.dim * c.dim);
        let wv = read_vec::<f32>(f, c.n_layers * c.dim * c.dim);
        let wo = read_vec::<f32>(f, c.n_layers * c.dim * c.dim);
        let rms_ffn_weight = read_vec::<f32>(f, c.n_layers * c.dim);
        let w1 = read_vec::<f32>(f, c.n_layers * c.dim * c.hidden_dim);
        let w2 = read_vec::<f32>(f, c.n_layers * c.hidden_dim * c.dim);
        let w3 = read_vec::<f32>(f, c.n_layers * c.dim * c.hidden_dim);
        let rms_final_weight = read_vec::<f32>(f, c.dim);
        let head_size = c.dim / c.n_heads;
        let freq_cis_real = read_vec::<f32>(f, c.seq_len * head_size / 2);
        let freq_cis_imag = read_vec::<f32>(f, c.seq_len * head_size / 2);
        let wcls = match c.shared_weight {
            true => None,
            false => Some(read_vec::<f32>(f, c.vocab_size * c.dim)),
        };

        Self {
            token_embedding_table,
            rms_att_weight,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_weight,
            w1,
            w2,
            w3,
            rms_final_weight,
            freq_cis_real,
            freq_cis_imag,
            wcls,
        }
    }
}

struct RunState {
    // current wave of activations
    x: Vec<f32>,      // activation at current time stamp (dim,)
    xb: Vec<f32>,     // same, but inside a residual branch (dim,)
    xb2: Vec<f32>,    // an additional buffer just for convenience (dim,)
    hb: Vec<f32>,     // buffer for hidden dimension in the ffn (hidden_dim,)
    hb2: Vec<f32>,    // buffer for hidden dimension in the ffn (hidden_dim,)
    q: Vec<f32>,      // query (dim,)
    k: Vec<f32>,      // key (dim,)
    v: Vec<f32>,      // value (dim,)
    att: Vec<f32>,    // buffer for scores/attention values (n_heads, seq_len)
    logits: Vec<f32>, // output logits (vocab_size,)
    // kv cache
    key_cache: Vec<f32>,   // (layer, seq_len, dim)
    value_cache: Vec<f32>, // (layer, seq_len, dim)
}

impl RunState {
    fn new(c: &Config) -> Self {
        Self {
            x: vec![0.0; c.dim as usize],
            xb: vec![0.0; c.dim as usize],
            xb2: vec![0.0; c.dim as usize],
            hb: vec![0.0; c.hidden_dim as usize],
            hb2: vec![0.0; c.hidden_dim as usize],
            q: vec![0.0; c.dim as usize],
            k: vec![0.0; c.dim as usize],
            v: vec![0.0; c.dim as usize],
            att: vec![0.0; (c.n_heads * c.seq_len) as usize],
            logits: vec![0.0; c.vocab_size as usize],
            key_cache: vec![0.0; (c.n_layers * c.seq_len * c.dim) as usize],
            value_cache: vec![0.0; (c.n_layers * c.seq_len * c.dim) as usize],
        }
    }
}

// ---------------------------------------------------------------------------
// Neural magic
fn rmsnorm(o: &mut Vec<f32>, x: &Vec<f32>, weight: &[f32]) {
    // mean sum of squares
    let mss = x.par_iter().map(|&y| y * y).sum::<f32>() / (x.len() as f32);
    let rsqrt: f32 = 1.0 / (mss + 1e-5f32).sqrt();

    o.par_iter_mut()
        .zip(&x[..])
        .zip(weight)
        .for_each(|((oi, xi), wi)| *oi = *wi * rsqrt * *xi);
}

fn matmul(o: &mut Vec<f32>, x: &[f32], w: &[f32], n: usize, _d: usize) {
    o.par_iter_mut().enumerate().for_each(|(i, oi)| {
        let mut val: f32 = 0.0;
        for j in 0..n {
            val += w[i * n + j] * x[j];
        }
        *oi = val;
    });
}

fn softmax(x: &mut [f32]) {
    let max = x.par_iter().copied().reduce(|| x[0], |a, b| a.max(b));
    x.par_iter_mut().for_each(|a| *a = (*a - max).exp());
    let sum = x.par_iter().sum::<f32>();
    x.par_iter_mut().for_each(|a| *a /= sum);
}

fn transformer(token: i32, pos: usize, p: &Config, s: &mut RunState, w: &TransformerWeights) {
    let token = token as usize;
    let dim = p.dim as usize;
    let hidden_dim = p.hidden_dim as usize;
    let head_size = dim / (p.n_heads as usize);
    let seq_len = p.seq_len as usize;
    let n_heads = p.n_heads as usize;

    // embed token into input vector x
    s.x.copy_from_slice(&w.token_embedding_table[token * dim..(token + 1) * dim]);
    // positional embedding
    let freq_cis_real_row = &w.freq_cis_real[pos * (head_size / 2)..(pos + 1) * (head_size / 2)];
    let freq_cis_imag_row = &w.freq_cis_imag[pos * (head_size / 2)..(pos + 1) * (head_size / 2)];

    // run through layers
    for l in 0..(p.n_layers as usize) {
        // pre-attention norm
        rmsnorm(&mut s.xb, &s.x, &w.rms_att_weight[l * dim..(l + 1) * dim]);

        // qkv projection
        matmul(
            &mut s.q,
            &s.xb,
            &w.wq[l * dim * dim..(l + 1) * dim * dim],
            dim,
            dim,
        );
        matmul(
            &mut s.k,
            &s.xb,
            &w.wk[l * dim * dim..(l + 1) * dim * dim],
            dim,
            dim,
        );
        matmul(
            &mut s.v,
            &s.xb,
            &w.wv[l * dim * dim..(l + 1) * dim * dim],
            dim,
            dim,
        );

        // rotary positional embedding
        for h in 0..n_heads {
            let q = &mut s.q[h * head_size..(h + 1) * head_size];
            let k = &mut s.k[h * head_size..(h + 1) * head_size];

            for i in 0..(head_size / 2) {
                let (fcr, fci) = (freq_cis_real_row[i], freq_cis_imag_row[i]);
                // rotate
                (q[i * 2], q[i * 2 + 1]) = (
                    q[i * 2] * fcr - q[i * 2 + 1] * fci,
                    q[i * 2] * fci + q[i * 2 + 1] * fcr,
                );
                (k[i * 2], k[i * 2 + 1]) = (
                    k[i * 2] * fcr - k[i * 2 + 1] * fci,
                    k[i * 2] * fci + k[i * 2 + 1] * fcr,
                );
            }
        }

        // cache kv values
        let loff = l * seq_len * dim; // layer offset
        s.key_cache[(loff + pos * dim)..(loff + (pos + 1) * dim)].copy_from_slice(&s.k);
        s.value_cache[(loff + pos * dim)..(loff + (pos + 1) * dim)].copy_from_slice(&s.v);

        // multihead attention

        let mut atts: Vec<&mut [f32]> = s.att.chunks_mut(seq_len).collect();
        let qs: Vec<&mut [f32]> = s.q.chunks_mut(head_size).collect();
        let xbs: Vec<&mut [f32]> = s.xb.chunks_mut(head_size).collect();

        atts.par_iter_mut()
            .zip(xbs)
            .enumerate()
            .for_each(|(h, (att, xb))| {
                let q: &[f32] = qs[h];
                for t in 0..(pos + 1) {
                    let koff = loff + t * dim + h * head_size; // key head offset
                    let k: &[f32] = &s.key_cache[koff..(koff + head_size)];
                    att[t] = q.iter().zip(k.iter()).map(|(&a, &b)| a * b).sum::<f32>()
                        / (head_size as f32).sqrt();
                }
                softmax(&mut att[..(pos + 1)]);
                xb.fill(0.0);
                for t in 0..(pos + 1) {
                    let koff = loff + t * dim + h * head_size; // key head offset
                    let v = &s.value_cache[koff..(koff + head_size)];
                    let a = att[t];
                    xb.iter_mut().zip(v).for_each(|(xbi, &vi)| *xbi += a * vi);
                }
            });

        // output projection
        matmul(
            &mut s.xb2,
            &s.xb,
            &w.wo[l * dim * dim..(l + 1) * dim * dim],
            dim,
            dim,
        );

        // residual connection -- add back to x
        s.x.iter_mut().zip(s.xb2.iter()).for_each(|(a, b)| *a += *b);

        // pre-ffn rmsnorm
        rmsnorm(&mut s.xb, &s.x, &w.rms_ffn_weight[l * dim..(l + 1) * dim]);

        // FFN block: self.w2(F.silu(self.w1(x)) * self.w3(x))
        matmul(
            &mut s.hb,
            &s.xb,
            &w.w1[l * hidden_dim * dim..(l + 1) * hidden_dim * dim],
            dim,
            hidden_dim,
        );
        matmul(
            &mut s.hb2,
            &s.xb,
            &w.w3[l * hidden_dim * dim..(l + 1) * hidden_dim * dim],
            dim,
            hidden_dim,
        );

        // apply silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
        s.hb.iter_mut()
            .for_each(|a| *a = *a * (1.0 / (1.0 + (-*a).exp())));

        // elementwise multiply with hb2=w3(x) into hb
        s.hb.iter_mut()
            .zip(s.hb2.iter())
            .for_each(|(a, &b)| *a *= b);

        // w2(...)
        matmul(
            &mut s.xb,
            &s.hb,
            &w.w2[l * dim * hidden_dim..(l + 1) * dim * hidden_dim],
            hidden_dim,
            dim,
        );

        // residual connection
        s.x.iter_mut().zip(s.xb.iter()).for_each(|(a, &b)| *a += b);
    }

    // final rmsnorm
    s.xb.copy_from_slice(&s.x);
    rmsnorm(&mut s.x, &s.xb, &w.rms_final_weight);

    // compute logits
    let wcls = match &w.wcls {
        Some(wcls) => wcls,
        None => &w.token_embedding_table,
    };
    matmul(&mut s.logits, &s.x, wcls, dim, p.vocab_size as usize);
}

// ---------------------------------------------------------------------------
// Tokenizer

type Score = f32;
fn read_tokenizer(vocab_size: usize, tokenizer_buffer: Vec<u8>) -> (Vec<(String, Score)>, u32) {
    let mut rdr = std::io::BufReader::new(std::io::Cursor::new(tokenizer_buffer));
    //BufReader::new(File::open("tokenizer.bin").expect("Couldn't load tokenizer.bin"));
    let max_token_length = read::<u32>(&mut rdr);

    let mut vocab: Vec<(String, Score)> = Vec::with_capacity(vocab_size);
    for _ in 0..vocab_size {
        let score = read::<f32>(&mut rdr);
        let len = read::<i32>(&mut rdr) as usize;
        let mut word = vec![0_u8; len];
        rdr.read_exact(&mut word).unwrap();
        vocab.push((String::from_utf8(word).unwrap(), score));
    }
    (vocab, max_token_length)
}

// ---------------------------------------------------------------------------
// Utilities

fn sample(probabilities: &Vec<f32>, rng: &mut PCG) -> usize {
    let r = rng.rand();
    let mut cdf = 0.0;
    for (i, &p) in probabilities.iter().enumerate() {
        cdf += p;
        if r < cdf {
            return i;
        }
    }
    probabilities.len() - 1
}

fn bpe_encode(text: &[u8], vocab: &[(String, Score)], max_token_length: usize) -> Vec<usize> {
    let mut tokens: Vec<usize> = Vec::new();

    for i in 0..text.len() {
        let char: &str = std::str::from_utf8(&text[i..i + 1]).unwrap();
        let (id, _) = vocab
            .iter()
            .enumerate()
            .find(|x| x.1 .0 == char)
            .expect("illegal character");
        tokens.push(id);
    }

    let mut buffer = String::with_capacity(max_token_length);
    loop {
        let mut best = (-1e10_f32, (usize::MAX, usize::MAX)); // (score, (vocab index, tokens index))

        for i in 0..tokens.len() - 1 {
            buffer.clear();
            buffer.push_str(&vocab[tokens[i]].0);
            buffer.push_str(&vocab[tokens[i + 1]].0);
            if let Some((vid, (_, score))) = vocab.iter().enumerate().find(|x| x.1 .0 == buffer) {
                if *score > best.0 {
                    best = (*score, (vid, i));
                }
            }
        }

        if best.1 .0 == usize::MAX {
            break; // no more possible merges
        }

        // perform merge
        tokens[best.1 .1] = best.1 .0;
        tokens.remove(best.1 .1 + 1);
    }

    tokens
}

// Poor man's num_traits
trait FromBytes {
    fn from_bytes(bytes: [u8; 4]) -> Self;
}
impl FromBytes for i32 {
    fn from_bytes(bytes: [u8; 4]) -> Self {
        i32::from_le_bytes(bytes)
    }
}
impl FromBytes for u32 {
    fn from_bytes(bytes: [u8; 4]) -> Self {
        u32::from_le_bytes(bytes)
    }
}
impl FromBytes for f32 {
    fn from_bytes(bytes: [u8; 4]) -> Self {
        f32::from_le_bytes(bytes)
    }
}

fn read<T: FromBytes>(rdr: &mut BufReader<std::io::Cursor<Vec<u8>>>) -> T {
    //log::info!("read rdr {:?} for 4bytes",rdr);
    let mut buffer = [0u8; 4];
    rdr.read_exact(&mut buffer).expect("Error reading file");
    T::from_bytes(buffer)
}

fn read_vec<T: FromBytes>(rdr: &mut BufReader<std::io::Cursor<Vec<u8>>>, size: i32) -> Vec<T> {
    //log::info!("read_vec with size {:?} and rdr {:?}",size,rdr);
    (0..size).map(|_| read::<T>(rdr)).collect()
}

// need to be called before main_wasm
// takes num_threads:usize, returns promise
pub use wasm_bindgen_rayon::init_thread_pool;

#[wasm_bindgen]
pub fn main_wasm(
    model_buffer: Vec<u8>,     //model_path
    tokenizer_buffer: Vec<u8>, //tokenizer.bin
    temperature: f32,
    steps: usize,
    prompt: String,
) -> String {
    let mut result: String = Default::default();
    let _ = console_log::init_with_level(log::Level::Trace);
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));

    log::info!("Starting inference with prompt [{}] ", prompt);

    let cpus = web_sys::window()
        .expect("no global `window` exists")
        .navigator()
        .hardware_concurrency() as usize;
    log::info!("--> [available 'CPUs' = {}]\n\n", cpus);

    // we should be able to use 75% if hardware_concurrency is available,
    // check init_thread_pool above

    let active_cpus = current_num_threads();
    log::info!("--> [Running Inference on {} 'CPUs']\n\n", active_cpus);

    let rng_seeds = (42, 54);
    let mut rng = PCG::new(rng_seeds.0, rng_seeds.1);

    let mut rdr = std::io::BufReader::new(std::io::Cursor::new(model_buffer));

    let config = Config::from_buf_reader(&mut rdr);
    log::info!("Config loaded {:?}", config);

    let weights = TransformerWeights::from_buf_reader(&mut rdr, &config);
    log::info!("weights loaded");

    let (vocab, max_token_length) = read_tokenizer(config.vocab_size as usize, tokenizer_buffer);
    log::info!("Vocab loaded");

    // parse prompt from user input
    // ensure at least "" is specified in js code
    let prompt = String::from(prompt.trim());

    let prompt_tokens = match prompt.len() {
        0 => Vec::new(),
        _ => bpe_encode(prompt.as_bytes(), &vocab, max_token_length as usize),
    };

    // Main generation loop.
    let mut state = RunState::new(&config);
    let perf = web_sys::window()
        .expect("no global `window` exists")
        .performance()
        .expect("should have performance on window");
    let now = || perf.now();
    let start = now();
    let mut next;
    let mut token = 1; // token 1 is <s> (bos) in the vocab
    let mut pos: usize = 0;
    log::info!("<s>");
    while pos < steps {
        transformer(token, pos, &config, &mut state, &weights);

        if pos < prompt_tokens.len() {
            next = prompt_tokens[pos];
        } else if temperature == 0.0 {
            // greedy decoding, choose argmax
            next = state
                .logits
                .iter()
                .enumerate()
                .reduce(|(i1, v1), (i2, v2)| if v1 > v2 { (i1, v1) } else { (i2, v2) })
                .map(|(i, _)| i)
                .unwrap();
        } else {
            // temperature scaling
            if temperature < 1.0 {
                state.logits.iter_mut().for_each(|z| *z /= temperature);
            }
            // compute probabilities
            softmax(&mut state.logits);
            next = sample(&state.logits, &mut rng);
        }

        result.push_str(&vocab[next].0);
        log::info!("{}", vocab[next].0);
        stdout().flush().unwrap();

        token = next as i32;
        pos += 1;
    }
    let elapsed = now() - start;
    log::info!("\n");
    log::info!("--------------------------------");
    log::info!(
        "elapsed: {} ms, avg tok/Second: {}",
        elapsed,
        ((steps - 1) / (elapsed as usize/1000)) 
    );
    result
}
