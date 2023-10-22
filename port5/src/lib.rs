// extern crate blas_src;
mod llama2;
use std::io::{self, BufReader, Write};

use js_sys::{Object, Reflect};
use llama2::*;
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::JsValue;

// Quantization papers:
//   * SmoothQuant - https://arxiv.org/pdf/2211.10438.pdf
//   * AWQ - https://arxiv.org/pdf/2306.00978.pdf
//   * The case for 4 bit precision - https://arxiv.org/pdf/2212.09720.pdf

#[macro_use]
extern crate num_derive;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Mode {
  Generate,
  _Chat,    /*not implemented*/
  _Compare, /*not implemented*/
}

#[wasm_bindgen]
pub fn port5_main_wasm(
  model_buffer: Vec<u8>,     // model_path
  tokenizer_buffer: Vec<u8>, // tokenizer.bin
  temperature: f32,
  steps: usize,
  prompt: String,
) -> Object {
  let _ = console_log::init_with_level(log::Level::Trace);
  std::panic::set_hook(Box::new(console_error_panic_hook::hook));

  let mut logs: Vec<String> = Vec::with_capacity((8 * 32) + steps);
  let mut result: Vec<String> = Vec::with_capacity(steps);
  let mut token_per_second: f64 = 0f64;
  logs.push("using port5".into());
  let mode = Mode::Generate;
  // Initialize config from file
  let mut file = model_buffer;
  let config =
    Config::read(&mut file, &mut logs).expect("Failed to read the config");

  // Finish reading the checkpoint file by loading weights
  let perf = web_sys::window()
    .expect("no global `window` exists")
    .performance()
    .expect("should have performance on window");
  let now = || perf.now();
  let start = now();

  logs.push("Reading model weights, takes a little while...".into());
  let mut reader = BufReader::new(std::io::Cursor::new(file));
  let weights = Weights::read(&config, false, &mut reader);

  let alog = format!("Read model weights in {:.2}s.", perf.now() - start);
  logs.push(alog);
  let steps = steps.max(1).min(config.seq_len);

  let tokenizer_path = tokenizer_buffer;
  let tok = Tokenizer::read(&tokenizer_path, config.vocab_size)
    .expect("Failed to load tokenizer");

  let sampler = Sampler::new(temperature);

  match mode {
    Mode::Generate => {
      let mut trans = Transformer::new(&config, weights);
      token_per_second = generate(
        &mut trans,
        &tok,
        &sampler,
        steps,
        Some(prompt),
        &mut result,
        &mut logs,
      ) * 1000f64;
    },
    _ => {
      panic!("not implemented in this version")
    },
  }

  let logs = JsValue::from(logs.join("\n"));
  let result = JsValue::from(
    result
      .into_iter()
      .map(|mut s| if s.starts_with(' ') { s } else { s.insert_str(0," "); s })
      .collect::<String>(),
  );

  let token_per_second = JsValue::from(token_per_second);

  let ret = Object::new();
  //@todo use define_property instead of reflect
  //Object::define_property(&ret, &logs,&Object::new());
  Reflect::set(&ret, &"logs".into(), &logs).unwrap();
  Reflect::set(&ret, &"result".into(), &result).unwrap();
  Reflect::set(&ret, &"token_per_second".into(), &token_per_second).unwrap();
  ret
}

pub fn generate(
  transformer: &mut Transformer,
  tok: &Tokenizer,
  sampler: &Sampler,
  steps: usize,
  prompt: Option<String>,
  result: &mut Vec<String>,
  logs: &mut Vec<String>,
) -> f64 {
  let prompt = prompt.unwrap_or_default();
  let prompt_toks = tok.encode(&prompt, true, false);

  let perf = web_sys::window()
    .expect("no global `window` exists")
    .performance()
    .expect("should have performance on window");
  let now = || perf.now();
  let start = now();

  let mut token = prompt_toks[0];
  result.push("<s>".into());

  for pos in 0..steps {
    let logits = transformer.forward(token, pos);

    let next = if pos < prompt_toks.len() - 1 {
      prompt_toks[pos + 1]
    } else {
      sampler.sample(logits).expect("No logits")
    };
    let got_token = tok.decode(token, next);
    let alog = format!("tok.decode : {}", &got_token);
    logs.push(alog);

    result.push(got_token);

    io::stdout().flush().unwrap();
    token = next;
  }
  let alog =
    format!("achieved tok/ms: {}", steps as f64 / (perf.now() - start));

  logs.push(alog);
  steps as f64 / (perf.now() - start)
}
