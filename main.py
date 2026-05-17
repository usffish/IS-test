from transformers import pipeline,AutoConfig

qwen_config = AutoConfig.from_pretrained("Qwen/Qwen2.5-1.5B")
print(qwen_config)
'''
question_answerer = pipeline("text-generation", model="Qwen/Qwen2.5-1.5B")
result=question_answerer("question:Where do I work?\n"
                         "Context:My name is Sylvain and I work at Hugging Face in Brooklyn\n"
                         "Answer:"
                         )
print(result)
'''