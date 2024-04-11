from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments, Trainer
from utils import *


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(
        default = -1, metadata = {"help": "Used for multi-gpu"}
    )

    per_device_train_batch_size: Optional[int] = field(default = 4)
    per_device_eval_batch_size: Optional[int] = field(default = 1)
    gradient_accumulation_steps: Optional[int] = field(default = 4)
    learning_rate: Optional[float] = field(default = 2e-4)
    max_grad_norm: Optional[float] = field(default = 0.3)
    weight_decay: Optional[float] = field(default = 0.001)
    max_seq_length: Optional[int] = field(default = 512)
    model_name: Optional[str] = field(
        default = "Salesforce/codegen25-7b-multi",
        metadata = {
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    dataset_name: Optional[str] = field(
        default = "timdettmers/openassistant-guanaco",
        metadata = {"help": "The preference dataset to use."},
    )
    use_nested_quant: Optional[bool] = field(
        default = False,
        metadata = {"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default = "float16",
        metadata = {"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default = "nf4",
        metadata = {"help": "Quantization type fp4 or nf4"},
    )
    num_train_epochs: Optional[int] = field(
        default = 1,
        metadata = {"help": "The number of training epochs for the reward model."},
    )
    fp16: Optional[bool] = field(
        default = False,
        metadata = {"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default = False,
        metadata = {"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default = False,
        metadata = {"help": "Use packing dataset creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default = True,
        metadata = {"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default = "paged_adamw_32bit",
        metadata = {"help": "The optimizer to use."},
    )
    lr_scheduler_type: str = field(
        default = "constant",
        metadata = {
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    max_steps: int = field(
        default = 10000, metadata = {"help": "How many optimizer update steps to take"}
    )
    warmup_ratio: float = field(
        default = 0.03, metadata = {"help": "Fraction of steps to do a warmup for"}
    )
    save_steps: int = field(
        default = 10, metadata = {"help": "Save checkpoint every X updates steps."}
    )
    eval_steps: int = field(default = 10, metadata = {"help": "Eval model every X steps."})
    logging_steps: int = field(
        default = 10, metadata = {"help": "Log every X updates steps."}
    )
    output_dir: str = field(
        default = "results", metadata = {"help": "Where to store the final model."}
    )
    use_flash_attn: Optional[bool] = field(
        default = False,
        metadata = {"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default = False,
        metadata = {"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_qunatization: Optional[bool] = field(
        default = False,
        metadata = {"help": "Enables loading model in 8bit."},
    )
    use_4bit_qunatization: Optional[bool] = field(
        default = False,
        metadata = {"help": "Enables loading model in 4bit."},
    )
    use_gradient_checkpointing: Optional[bool] = field(
        default = False,
        metadata = {"help": "Enables Gradient Checkpointing."},
    )
    dataset_text_field: str = field(
        default = "text", metadata = {"help": "Dataset field to use as input text."}
    )
    push_to_hub: Optional[bool] = field(
        default = False,
        metadata = {"help": "If True, pushes the model to the HF Hub"},
    )
    num_workers: int = field(
        default = 4, metadata = {"help": "Number of dataset workers to use."}
    )
    debug: Optional[bool] = field(
        default = False,
        metadata = {
            "help": "If True, tests things like proper saving/loading/logging of model"
        },
    )


def main(args):
    # model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        load_in_8bit = args.use_8bit_qunatization,
        quantization_config = None,
        device_map = None,
        use_cache = not args.use_gradient_checkpointing,
        trust_remote_code = True,
        use_flash_attention_2 = args.use_flash_attn
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code = True)
    tokenizer.pad_token = tokenizer.eos_token

    # for param in model.parameters():
    #     param.requires_grad = False
    # model.get_input_embeddings().requires_grad = True

    # datasets
    train_dataset, eval_dataset = create_datasets(tokenizer, args)

    # training arguments
    training_arguments = TrainingArguments(
        output_dir = args.output_dir,
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = args.gradient_accumulation_steps,
        learning_rate = args.learning_rate,
        fp16 = args.fp16,
        bf16 = args.bf16,
        max_grad_norm = args.max_grad_norm,
        warmup_ratio = args.warmup_ratio,
        lr_scheduler_type = args.lr_scheduler_type,
        num_train_epochs = args.num_train_epochs,
        evaluation_strategy = "steps",
        max_steps = args.max_steps,
        eval_steps = args.eval_steps,
        save_steps = args.save_steps,
        logging_steps = args.logging_steps,
        push_to_hub = args.push_to_hub,
        gradient_checkpointing = args.use_gradient_checkpointing,
        include_tokens_per_second = True,
        report_to = None
    )
    # trainer
    trainer = Trainer(
        model = model,
        args = training_arguments,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
    )
    # train
    trainer.train()


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    main(args)
