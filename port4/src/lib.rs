mod llama2;
mod ops;

use crate::llama2::*;
use js_sys::{Object, Reflect};
use wasm_bindgen::{prelude::wasm_bindgen, JsValue};

/// Runs inference with repeated experiments and gets a vec of performance (tokens/s).
#[allow(clippy::too_many_arguments)]
fn run_inference(
  prompt: String,
  model_path: Vec<u8>,
  tokenizer_path: Vec<u8>,
  temperature: f32,
  n_steps: i32,
  n_repeated_experiments: u32,
  logs: &mut Vec<String>,
  result: &mut Vec<String>,
) -> Vec<f64> {
  let config = Config::new_from_file(&model_path).unwrap();
  let alog = format!("Config: {:?}\n", config);
  logs.push(alog);
  let weights = TransformerWeights::new(&model_path, &config).unwrap();
  let vocab = Vocabulary::new_from_file(tokenizer_path, &config).unwrap();

  let mut speeds = vec![];
  for _ in 0..n_repeated_experiments {
    let state = &mut RunState::new(&config);
    speeds.push(state.run(
      &prompt,
      temperature,
      n_steps,
      &config,
      &weights,
      &vocab,
      logs,
      result,
    ));
  }
  speeds
}

// need to be called before main_wasm
// takes num_threads:usize, returns promise
pub use wasm_bindgen_rayon::init_thread_pool;

#[wasm_bindgen]
pub fn port4_main_wasm(
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
  let token_per_second: f64 = 0f64;

  logs.push("using port4".into());

  let _speeds = run_inference(
    prompt,
    model_buffer,
    tokenizer_buffer,
    temperature,
    steps as i32,
    1,
    &mut logs,
    &mut result,
  );
  let alog = format!("sampled speed is {:?}", _speeds);
  logs.push(alog);
  let logs = JsValue::from(logs.join("\n"));
  let result = JsValue::from(
    result
      .into_iter()
      .map(|mut s| if s.starts_with(' ') { s } else { s.insert_str(0," "); s })
      .collect::<String>(),
  );

  let sum: f64 = _speeds.iter().sum();
  let count = _speeds.len() as f64;

  let token_per_second = if count > 0.0 {
    sum / count
  } else {
    token_per_second // Handle the case of an empty vector (optional)
  };

  let token_per_second = JsValue::from(token_per_second);

  let ret = Object::new();
  //@todo use define_property instead of reflect
  //Object::define_property(&ret, &logs,&Object::new());
  Reflect::set(&ret, &"logs".into(), &logs).unwrap();
  Reflect::set(&ret, &"result".into(), &result).unwrap();
  Reflect::set(&ret, &"token_per_second".into(), &token_per_second).unwrap();
  ret
}
