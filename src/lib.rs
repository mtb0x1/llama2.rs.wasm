use js_sys::Object;
use port1::port1_main_wasm;
use port2::port2_main_wasm;
use port3::port3_main_wasm;
use port4::port4_main_wasm;
use port5::port5_main_wasm;
use port6::port6_main_wasm;
use wasm_bindgen::prelude::wasm_bindgen;

#[derive(Debug)]
enum WhichImpl {
  Port1,
  Port2,
  Port3,
  Port4,
  Port5,
  Port6,
}

impl TryFrom<String> for WhichImpl {
  type Error = String;
  fn try_from(value: String) -> Result<Self, Self::Error> {
    match value.as_str() {
      "port1" => Ok(WhichImpl::Port1),
      "port2" => Ok(WhichImpl::Port2),
      "port3" => Ok(WhichImpl::Port3),
      "port4" => Ok(WhichImpl::Port4),
      "port5" => Ok(WhichImpl::Port5),
      "port6" => Ok(WhichImpl::Port6),
      _ => Err(format!("Invalid impl_type: {}", value)),
    }
  }
}

impl TryFrom<WhichImpl> for String {
  type Error = String;
  fn try_from(value: WhichImpl) -> Result<Self, Self::Error> {
    match value {
      WhichImpl::Port1 => Ok("port1".into()),
      WhichImpl::Port2 => Ok("port2".into()),
      WhichImpl::Port3 => Ok("port3".into()),
      WhichImpl::Port4 => Ok("port4".into()),
      WhichImpl::Port5 => Ok("port5".into()),
      WhichImpl::Port6 => Ok("port6".into()),
      //_ => Err(format!("Invalid impl_type: {:?}", value)),
    }
  }
}

pub use wasm_bindgen_rayon::init_thread_pool;

#[wasm_bindgen]
pub fn main_wasm(
  model_buffer: Vec<u8>,     // model_path
  tokenizer_buffer: Vec<u8>, // tokenizer.bin
  temperature: f32,
  steps: usize,
  prompt: String,
  impl_type: String,
) -> Object {
  let impl_type: Result<WhichImpl, String> = impl_type.try_into();
  match impl_type {
    Ok(WhichImpl::Port1) => {
      port1_main_wasm(model_buffer, tokenizer_buffer, temperature, steps)
    },
    Ok(WhichImpl::Port2) => port2_main_wasm(
      model_buffer,
      tokenizer_buffer,
      temperature,
      steps,
      prompt,
    ),
    Ok(WhichImpl::Port3) => port3_main_wasm(
      model_buffer,
      tokenizer_buffer,
      temperature,
      steps,
      prompt,
    ),
    Ok(WhichImpl::Port4) => port4_main_wasm(
      model_buffer,
      tokenizer_buffer,
      temperature,
      steps,
      prompt,
    ),
    Ok(WhichImpl::Port5) => port5_main_wasm(
      model_buffer,
      tokenizer_buffer,
      temperature,
      steps,
      prompt,
    ),
    Ok(WhichImpl::Port6) => port6_main_wasm(
      model_buffer,
      tokenizer_buffer,
      temperature,
      steps,
      prompt,
    ),
    Err(e) => {
      panic!("Error: {:?}", e)
    },
  }
}
