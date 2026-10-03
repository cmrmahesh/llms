// Model registry. Any HuggingFace repo with a LlamaForCausalLM config, a BPE
// tokenizer.json and a single model.safetensors file works with this engine.
// To add one, append an entry here.

export const MODELS = [
  {
    id: 'smollm2-135m-instruct',
    label: 'SmolLM2-135M-Instruct',
    repo: 'HuggingFaceTB/SmolLM2-135M-Instruct',
    chat: true,
    systemPrompt: 'You are a helpful AI assistant named SmolLM, trained by Hugging Face',
    downloadMB: 269,
    note: '135M parameters, 30 layers. Small enough to run in a browser tab.',
  },
  {
    id: 'smollm2-135m',
    label: 'SmolLM2-135M (base)',
    repo: 'HuggingFaceTB/SmolLM2-135M',
    chat: false,
    downloadMB: 269,
    note: 'Same architecture without instruction tuning: it continues text instead of answering.',
  },
  {
    id: 'smollm2-360m-instruct',
    label: 'SmolLM2-360M-Instruct',
    repo: 'HuggingFaceTB/SmolLM2-360M-Instruct',
    chat: true,
    systemPrompt: 'You are a helpful AI assistant named SmolLM, trained by Hugging Face',
    downloadMB: 724,
    note: '360M parameters, 32 layers. Better answers, about 3× slower, needs ~1.5 GB of memory.',
  },
];

export const fileUrl = (m, f) => `https://huggingface.co/${m.repo}/resolve/main/${f}`;

export function chatPrompt(model, system, user) {
  return `<|im_start|>system\n${system}<|im_end|>\n<|im_start|>user\n${user}<|im_end|>\n<|im_start|>assistant\n`;
}
