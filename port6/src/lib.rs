mod error;
mod llama2;
mod tensor;
use js_sys::{Object, Reflect};
use rayon::current_num_threads;
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::JsValue;


use crate::llama2::{
  Llama2CheckpointLoader, Llama2Loader, Llama2Runner, Llama2Sampler,
};
use std::io::Write;

#[wasm_bindgen]
pub fn port6_main_wasm(
  model_buffer: Vec<u8>,     // model_path
  tokenizer_buffer: Vec<u8>, // tokenizer.bin
  temperature: f32,
  steps: usize,
  prompt: String,
) -> Object {
  let probability = 0.9;
  let _ = console_log::init_with_level(log::Level::Trace);
  std::panic::set_hook(Box::new(console_error_panic_hook::hook));

  let mut logs: Vec<String> = Vec::with_capacity((8 * 32) + steps);
  let mut result: Vec<String> = Vec::with_capacity(steps);
  let mut token_per_second: f32 = 0.0_f32;
  logs.push("using port6".into());

  let checkpoint_loader =
    Llama2CheckpointLoader::new(model_buffer, tokenizer_buffer).unwrap();

  let (conf, weights, tokenizer) = checkpoint_loader.load().unwrap();
  let mut sampler =
    Llama2Sampler::new(conf.vocab_size, temperature, probability);
  let mut runner = Llama2Runner::new(&conf, weights, tokenizer);
  let mut output = runner.generate(&prompt, steps, &mut sampler).unwrap();
  for token in output.by_ref() {
    let got_token = token.unwrap();
    let alog = format!("token.unwrap() = {}", &got_token);
    logs.push(alog);
    result.push(got_token);
    std::io::stdout().flush().unwrap();
  }

  let threads = current_num_threads();

  token_per_second = output.average_tokens_per_seconds()*1000_f32;
  let alog = format!("{} tokens/s, {} threads", token_per_second, threads);
  logs.push(alog);
  let logs = JsValue::from(logs.join("\n"));
  let result = JsValue::from(
    result
      .into_iter()
      .map(|mut s| {
        if s.starts_with(' ') {
          s
        } else {
          s.insert(0, ' ');
          s
        }
      })
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
