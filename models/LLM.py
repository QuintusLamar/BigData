from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from openai import OpenAI
class LLAMA():
    def __init__(self):
        self.model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
        self.model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"
        self.model_path = None
        self.model_path = hf_hub_download(repo_id=self.model_name_or_path, filename=self.model_basename)

        self.lcpp_llm = None
        self.lcpp_llm = Llama(
            model_path=self.model_path,
            n_threads=2, # CPU cores
            n_batch=512, # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
            n_gpu_layers=43, # Change this value based on your model and your GPU VRAM pool.
            n_ctx=4096, # Context window
        )

    def getFiveQuestions(self, caption):
        prompt = f"Given the following caption, generate 5 questions about the picture that you can ask a Visual Question and Answer Model. |{caption}|"
        prompt_template=f'''SYSTEM: You are a helpful, respectful and honest assistant. Always answer as helpfully.

        USER: {prompt}

        ASSISTANT:
        '''
        print("Getting response from LLaMA...")
        response = self.lcpp_llm(
            prompt=prompt_template,
            max_tokens=128,
            temperature=0.5,
            top_p=0.95,
            repeat_penalty=1.2,
            top_k=50,
            stop = ['USER:'], # Dynamic stopping when such token is detected.
            echo=True # return the prompt
        )
        
        # print(response["choices"][0]["text"])
        res = response["choices"][0]["text"]
        questionsArr = res.split("ASSISTANT:")[1].splitlines()
        questionsArr = [question[3:] for question in questionsArr if len(question) >3]
        
        return questionsArr

class GPT():
    def __init__(self):
        self.api_key = "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"

        self.client = OpenAI(api_key=self.api_key)

    def getFiveQuestions(self, caption):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"Generate 5 questions that can be asked to a Visual Question And Answer model given the following caption of an image: {caption}",
                }
            ],
            model="gpt-3.5-turbo",
        )
        retString = chat_completion.choices[0].message.content
        retArr = retString.splitlines()

        retArr = [re[3:] for re in retArr]
        return retArr


if __name__ == "__main__":
    llama = LLAMA()
    gpt = GPT()
    caption = "A dog playing with a football in a field"
    print(llama.getFiveQuestions(caption))
    print(gpt.getFiveQuestions(caption))